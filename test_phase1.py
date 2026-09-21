import os
import unittest.mock as mock
import pytest

from cognition.agent import VisionAgent, VisionContext, WorldState
from memory.context_memory import PersonMemory, PersonaGraph, PERSONA_DIR


def test_think_and_stream_response_write_gate():
    """Definition of Done: calling think()/stream_response() twice with no new
    information does not call save() a second time.
    """
    agent = VisionAgent()
    agent.llm = mock.MagicMock()
    agent.llm.available = True
    agent.llm.complete.return_value = "Hello there"
    agent.llm.stream.return_value = iter(["Hello", " there"])
    identity = "test_user_p1"
    face_data = {"authorized": True, "score": 0.95}

    # Ensure clean slate for test user
    path = os.path.join(PERSONA_DIR, f"{identity}.json")
    if os.path.exists(path):
        os.remove(path)

    mem = agent._memory_for(identity)

    # First turn: may save if initial state is dirty / fact learned
    with mock.patch.object(mem, "save", wraps=mem.save) as spy_save:
        agent.think(identity, face_data, user_input="hello")
        first_call_count = spy_save.call_count

        # Second turn with no new persistent information (small talk):
        agent.think(identity, face_data, user_input="hello again")
        assert spy_save.call_count == first_call_count, (
            f"save() was called on think() with no new info (calls: {spy_save.call_count})"
        )

        # Third turn via stream_response with no new persistent info:
        list(agent.stream_response(identity, face_data, user_input="still here"))
        assert spy_save.call_count == first_call_count, (
            f"save() was called on stream_response() with no new info (calls: {spy_save.call_count})"
        )

    # Cleanup
    if os.path.exists(path):
        os.remove(path)


def test_two_unrecognized_visitors_isolated_and_nonpersistent():
    """Definition of Done: two different unrecognized visitors in one session
    never see each other's learned facts, and stranger memory is never written to disk.
    """
    agent = VisionAgent()
    agent.llm = mock.MagicMock()
    agent.llm.available = False

    # Snapshot existing files and mtimes before stranger turns
    files_before = set(os.listdir(PERSONA_DIR))
    mtimes_before = {f: os.path.getmtime(os.path.join(PERSONA_DIR, f)) for f in files_before}

    # Visitor 1: unrecognized stranger on track face_0
    face_0 = {"track_id": "face_0", "authorized": False, "score": 0.45}
    agent.think(
        "unknown",
        face_0,
        user_input="My name is Bob",
    )

    mem_0 = agent._memory_for("unknown", face_0)
    assert mem_0.persona.name == "bob"
    mem_0.add_fact("purpose", "delivery")

    # Visitor 2: distinct unrecognized stranger on track face_1
    face_1 = {"track_id": "face_1", "authorized": False, "score": 0.40}
    resp_1, _ = agent.think("unknown", face_1, user_input="hello")

    mem_1 = agent._memory_for("unknown", face_1)
    assert mem_1.persona.name is None, "Visitor 2 inherited Visitor 1's name"
    assert mem_1.persona.purpose == "", "Visitor 2 inherited Visitor 1's purpose"
    assert "bob" not in resp_1.lower(), "ARIA greeted Visitor 2 with Visitor 1's name"

    # Verify no persistent files were created or modified on disk for unknown visitors
    files_after = set(os.listdir(PERSONA_DIR))
    assert files_after == files_before, f"New files created: {files_after - files_before}"
    for f in files_after:
        assert os.path.getmtime(os.path.join(PERSONA_DIR, f)) == mtimes_before[f], (
            f"File {f} in PERSONA_DIR was modified during stranger interaction"
        )

    # Test stranger track eviction when track ends
    agent.clear_stranger_memory("face_0")
    assert "unknown_face_0" not in agent.memory
    assert "unknown_face_1" in agent.memory


def test_stranger_memory_never_persists_even_if_explicitly_called():
    """Stranger / unknown PersonaGraph and PersonMemory instances refuse to persist."""
    mem = PersonMemory("unknown")
    mem.add_fact("name", "EphemeralStranger")
    saved = mem.save()
    assert saved is False, "PersonMemory('unknown').save() should return False"

    pg = PersonaGraph("unknown")
    pg.name = "EphemeralStranger"
    saved_pg = pg.save()
    assert saved_pg is False, "PersonaGraph('unknown').save() should return False"

    # Loading unknown always yields a fresh instance
    loaded = PersonMemory.load("unknown")
    assert loaded.persona.name is None


def test_world_state_structured_seed_in_vision_context():
    """VisionContext populates structured WorldState without changing rendered output."""
    vc = VisionContext()
    assert isinstance(vc.world_state, WorldState)
    assert vc.context_text() == "No vision input right now."

    # Update with 1 recognized, 1 stranger, 1 background person, and objects
    class DummyDet:
        def __init__(self, label):
            self.label = label

    face_results = [
        {"identity": "Alice", "authorized": True},
        {"identity": "unknown", "authorized": False},
    ]
    detections = [
        DummyDet("person"),
        DummyDet("person"),
        DummyDet("person"),  # 3 people total, 2 faces -> 1 extra in background
        DummyDet("laptop"),
    ]

    vc.update(face_results, detections)

    ws = vc.world_state
    assert ws.known_people == ["Alice"]
    assert ws.unrecognized_count == 1
    assert ws.background_people_count == 1
    assert ws.objects == ["laptop"]
    assert ws.last_known_name == "Alice"

    text = vc.context_text()
    assert "Recognized: Alice." in text
    assert "1 unrecognized visitor(s) here." in text
    assert "1 other person(s) in the background." in text
    assert "Nearby objects: laptop." in text

    # Event logging through WorldState
    vc.add_event("Alice opened a notebook.")
    assert "Alice opened a notebook." in vc.world_state.recent_events
    assert "Recent: Alice opened a notebook." in vc.context_text()

    # Identity clearance
    vc.clear_identity("Alice")
    assert vc.world_state.last_known_name is None
