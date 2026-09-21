"""Test suite for ARIA Phase 4 — Real Memory (Working / Semantic / Episodic).

Verifies:
1. WorkingMemory: session-scoped conversational buffer, turn limits, topic/question tracking.
2. SemanticMemory: persistent facts, procedural preferences, dirty-state tracking, case-insensitive load.
3. EpisodicMemory: dated discrete episodes, chronological recency, keyword search.
4. retrieve(): recency + person-scoped filtering, stranger isolation, prompt snippet formatting.
5. MemoryWriter: stranger barrier (unknown/anon_* blocked), dirty-state disk write-gate.
6. PersonMemory adapter in context_memory.py: backwards compatibility with Phase 1-3.
7. Two-session continuity: Session 1 records episode -> simulated restart -> Session 2 unprompted recall.
"""
import os
import shutil
import tempfile
import pytest

from memory.working import WorkingMemory
from memory.semantic import SemanticMemory
from memory.episodic import EpisodicMemory, Episode
from memory.retrieval import retrieve, retrieve_greeting_context
from memory.writer import MemoryWriter
from memory.context_memory import PersonMemory


@pytest.fixture
def temp_persona_dir(monkeypatch):
    """Isolate memory files in a clean temporary directory."""
    tmp = tempfile.mkdtemp(prefix="aria_test_mem_")
    import memory.semantic
    import memory.episodic
    import memory.context_memory
    monkeypatch.setattr(memory.semantic, "PERSONA_DIR", tmp)
    monkeypatch.setattr(memory.episodic, "PERSONA_DIR", tmp)
    monkeypatch.setattr(memory.context_memory, "PERSONA_DIR", tmp)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------
# 1. WorkingMemory Unit Tests
# ----------------------------------------------------------------------

def test_working_memory_turn_buffer_and_limit():
    wm = WorkingMemory(max_turns=3)
    wm.add_turn("user", "Hello")
    wm.add_turn("assistant", "Hi there!")
    wm.add_turn("user", "How are you?")
    assert len(wm.turns) == 3

    # Adding a 4th turn evicts the oldest
    wm.add_turn("assistant", "I am doing well.")
    assert len(wm.turns) == 3
    assert wm.turns[0]["content"] == "Hi there!"
    assert wm.turns[-1]["content"] == "I am doing well."


def test_working_memory_topic_and_pending_question():
    wm = WorkingMemory()
    wm.set_topic("Computer Vision")
    wm.set_pending_question("Have you trained YOLOv5 before?")
    assert wm.current_topic == "Computer Vision"
    assert wm.pending_question == "Have you trained YOLOv5 before?"

    recent = wm.recent_turns(2)
    assert isinstance(recent, list)

    wm.clear()
    assert len(wm.turns) == 0
    assert wm.current_topic is None
    assert wm.pending_question is None


# ----------------------------------------------------------------------
# 2. SemanticMemory Unit Tests
# ----------------------------------------------------------------------

def test_semantic_memory_facts_and_procedural_preferences(temp_persona_dir):
    sm = SemanticMemory(identity="Alice", name="Alice Wonderland")
    sm.relationship_tier = "familiar"
    sm.traits = ["inquisitive", "punctual"]

    # Facts
    sm.add_fact("project", "Robotics Navigation")
    sm.add_fact("hardware", "Raspberry Pi 5")
    assert sm.get_fact("project") == "Robotics Navigation"
    assert sm.get_fact("missing", "default_val") == "default_val"

    # Procedural preferences
    sm.update_preference("verbosity", "concise")
    sm.update_preference("formality", "casual")
    assert sm.get_preference("verbosity") == "concise"

    # Dirty tracking
    assert sm.is_dirty() is True
    assert sm.save() is True
    assert sm.is_dirty() is False

    # Reload from disk
    loaded = SemanticMemory.load("Alice")
    assert loaded.name == "Alice Wonderland"
    assert loaded.relationship_tier == "familiar"
    assert "inquisitive" in loaded.traits
    assert loaded.get_fact("project") == "Robotics Navigation"
    assert loaded.get_preference("verbosity") == "concise"


def test_semantic_memory_case_insensitive_load(temp_persona_dir):
    """Enrolled as 'MuzammilCK', loaded as 'muzammilck' or 'MuzammilCk'."""
    sm = SemanticMemory(identity="MuzammilCK", name="Muzammil")
    sm.add_fact("role", "Lead Architect")
    sm.save()

    # Load with different case
    loaded = SemanticMemory.load("muzammilck")
    assert loaded.identity == "MuzammilCK"
    assert loaded.name == "Muzammil"
    assert loaded.get_fact("role") == "Lead Architect"


# ----------------------------------------------------------------------
# 3. EpisodicMemory Unit Tests
# ----------------------------------------------------------------------

def test_episodic_memory_discrete_episodes_and_search(temp_persona_dir):
    em = EpisodicMemory(identity="Bob")
    em.add_episode(
        summary="Discussed OpenCV DNN integration and YOLOv5 detection.",
        topics=["OpenCV", "YOLOv5", "Vision"],
        key_events=["model_loaded"],
    )
    em.add_episode(
        summary="Set up microphone VAD speech recognition with 200ms debounce.",
        topics=["Audio", "VAD", "Speech"],
        key_events=["vad_configured"],
    )
    assert len(em.episodes) == 2
    assert em.is_dirty() is True

    # Recency
    recent = em.get_recent(limit=1)
    assert len(recent) == 1
    assert "microphone" in recent[0].summary

    # Keyword search
    search_yolo = em.search("YOLOv5")
    assert len(search_yolo) == 1
    assert "OpenCV DNN" in search_yolo[0].summary

    search_audio = em.search("speech recognition")
    assert len(search_audio) == 1
    assert "microphone" in search_audio[0].summary

    # Persistence
    assert em.save("Bob") is True
    assert em.is_dirty() is False

    loaded = EpisodicMemory.load("Bob")
    assert len(loaded.episodes) == 2
    assert loaded.episodes[0].summary.startswith("Discussed OpenCV")


# ----------------------------------------------------------------------
# 4. Selective Retrieval & Isolation Unit Tests
# ----------------------------------------------------------------------

def test_retrieval_isolation_between_persons(temp_persona_dir):
    """Data for Alice must never leak when retrieving for Bob."""
    alice_sm = SemanticMemory(identity="Alice", name="Alice")
    alice_sm.add_fact("secret_project", "Project Manhattan")
    alice_sm.save()

    bob_sm = SemanticMemory(identity="Bob", name="Bob")
    bob_sm.add_fact("secret_project", "Project Apollo")
    bob_sm.save()

    res_alice = retrieve("project", "Alice")
    assert "Project Manhattan" in res_alice["context_snippet"]
    assert "Project Apollo" not in res_alice["context_snippet"]

    res_bob = retrieve("project", "Bob")
    assert "Project Apollo" in res_bob["context_snippet"]
    assert "Project Manhattan" not in res_bob["context_snippet"]


def test_retrieval_stranger_never_leaks(temp_persona_dir):
    """Unknown or anonymous queries return zero persistent facts/episodes."""
    res_unknown = retrieve("anything", "unknown")
    assert res_unknown["semantic"] == {}
    assert res_unknown["episodic"] == []
    assert res_unknown["context_snippet"] == ""

    res_anon = retrieve("anything", "anon_face_1")
    assert res_anon["semantic"] == {}
    assert res_anon["episodic"] == []
    assert res_anon["context_snippet"] == ""


# ----------------------------------------------------------------------
# 5. MemoryWriter & Stranger Barrier Unit Tests
# ----------------------------------------------------------------------

def test_memory_writer_stranger_barrier(temp_persona_dir):
    """Unenrolled visitors ('unknown' or 'anon_*') must never write to disk."""
    assert MemoryWriter.is_persistent_identity("unknown") is False
    assert MemoryWriter.is_persistent_identity("anon_42") is False
    assert MemoryWriter.is_persistent_identity("") is False
    assert MemoryWriter.is_persistent_identity("Alice") is True

    # Attempting to save unknown PersonMemory returns False and writes no file
    mem_unknown = PersonMemory(identity="unknown")
    mem_unknown.semantic.add_fact("note", "stranger in lobby")
    saved = MemoryWriter.save_person(mem_unknown)
    assert saved is False
    assert not os.path.exists(os.path.join(temp_persona_dir, "unknown.json"))

    # Attempting to record episode for unknown returns False
    recorded = MemoryWriter.record_episode(mem_unknown, "Talked about weather")
    assert recorded is False


def test_memory_writer_dirty_state_gate(temp_persona_dir):
    """Skips disk write if data has not changed."""
    alice = PersonMemory(identity="Alice")
    alice.semantic.name = "Alice"
    alice.semantic.relationship_tier = "familiar"

    # Dirty: should save
    assert MemoryWriter.save_person(alice) is True

    # Not dirty: should skip write
    assert MemoryWriter.save_person(alice) is False

    # Add an episode: marks episodic dirty -> should save
    assert MemoryWriter.record_episode(alice, "Visited the lab to test cameras.", topics=["camera"]) is True
    assert MemoryWriter.save_person(alice) is False  # Clean again


# ----------------------------------------------------------------------
# 6. PersonMemory Backwards-Compatibility Adapter Unit Tests
# ----------------------------------------------------------------------

def test_person_memory_adapter_compatibility(temp_persona_dir):
    mem = PersonMemory(identity="Charlie")
    # Legacy persona properties
    mem.persona.name = "Charlie"
    mem.persona.purpose = "testing ARIA memory"
    mem.persona.traits = ["diligent", "tech-savvy"]

    # Working context turns
    mem.add_interaction("user", "Hello ARIA")
    mem.add_interaction("assistant", "Hello Charlie!")
    assert len(mem.working_context) == 2
    assert mem.working.turns[0]["content"] == "Hello ARIA"

    # Legacy add_fact updates semantic memory
    mem.add_fact("favorite_editor", "Neovim")
    assert mem.semantic.get_fact("favorite_editor") == "Neovim"

    # Serialization roundtrip
    d = mem.to_dict()
    assert d["identity"] == "Charlie"
    assert d["persona"]["name"] == "Charlie"
    assert "favorite_editor" in d["persona"]["facts"]

    restored = PersonMemory.from_dict(d)
    assert restored.identity == "Charlie"
    assert restored.persona.name == "Charlie"
    assert restored.semantic.get_fact("favorite_editor") == "Neovim"
    assert len(restored.working_context) == 3  # 2 interactions + 1 fact


# ----------------------------------------------------------------------
# 7. Multi-Session Unprompted Continuity (Phase 4 DoD Live Simulation)
# ----------------------------------------------------------------------

def test_multi_session_unprompted_callback_across_restart(temp_persona_dir):
    """Simulates:
    Session 1: User 'Dan' interacts with ARIA, discusses an autonomous drone.
               Departure records an episode. Episodic memory persists to disk.
    App Restart: In-memory state is wiped; agent reloads from disk.
    Session 2: Dan returns. ARIA unpromptedly references the drone discussion
               in greeting without any user prompt.
    """
    # --- SESSION 1 ---
    dan_mem = PersonMemory(identity="Dan")
    dan_mem.persona.name = "Dan"
    dan_mem.persona.purpose = "developing autonomous drone navigation"
    dan_mem.working.add_turn("user", "I spent the day tuning the Kalman filter for the autonomous drone.")
    dan_mem.working.add_turn("assistant", "Kalman filters can be tricky with noisy IMU data.")

    # User departs -> run.py _record_session_episode logic
    user_turns = [t["content"] for t in dan_mem.working.turns if t["role"] == "user"]
    summary = f"Visited and discussed: {user_turns[0][:80]}."
    topics = ["Kalman filter", "autonomous drone"]
    saved_ep = MemoryWriter.record_episode(dan_mem, summary=summary, topics=topics)
    assert saved_ep is True
    dan_mem.working.clear()
    dan_mem.semantic.save()

    # --- SIMULATE APP RESTART ---
    del dan_mem

    # --- SESSION 2 ---
    # Agent loads Dan fresh from disk
    dan_reloaded = PersonMemory.load("Dan")
    assert dan_reloaded.identity == "Dan"
    assert len(dan_reloaded.episodic.episodes) == 1

    # Unprompted callback for greeting
    callback = retrieve_greeting_context("Dan", dan_reloaded)
    assert callback is not None
    assert "autonomous drone" in callback or "Kalman filter" in callback

    # Test VisionAgent._fallback greeting incorporates this unprompted recall
    from cognition.agent import VisionAgent
    agent = VisionAgent()
    greeting = agent._fallback("Dan", None, dan_reloaded)
    assert "Dan" in greeting
    assert ("autonomous drone" in greeting) or ("Kalman filter" in greeting) or ("drone" in greeting)
