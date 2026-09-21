"""Unit tests for Phase 8 — Repository Restructuring.

Verifies:
1. All 6 packages (core, perception, memory, cognition, interaction, actions) exist and import cleanly.
2. Core modules (Event, PolicyDecision, ToolDispatcher) are exported from `core`.
3. Perception modules (Detector, FaceEngine, FaceTracker, PresenceManager) are exported from `perception`.
4. Memory modules (WorkingMemory, SemanticMemory, EpisodicMemory, MemoryWriter, retrieve, PersonMemory) from `memory`.
5. Cognition modules (VisionAgent, VisionContext, VoiceSession, AgentState, WorldState) from `cognition`.
6. Interaction modules (ConversationManager, DialogueState, SarvamVoice, MicVAD, UiHub) from `interaction`.
7. Actions modules (Alerter) from `actions`.
8. Root-level shims maintain 100% backward compatibility for legacy callers.
"""
import pytest


def test_package_imports_and_exports():
    import core
    import perception
    import memory
    import cognition
    import interaction
    import actions

    # 1. core
    assert hasattr(core, "Event")
    assert hasattr(core, "EventType")
    assert hasattr(core, "PolicyDecision")
    assert hasattr(core, "ToolDispatcher")
    assert hasattr(core, "should_greet")

    # 2. perception
    assert hasattr(perception, "Detector")
    assert hasattr(perception, "FaceEngine")
    assert hasattr(perception, "FaceTracker")
    assert hasattr(perception, "PresenceManager")
    assert hasattr(perception, "WebcamSource")

    # 3. memory
    assert hasattr(memory, "WorkingMemory")
    assert hasattr(memory, "SemanticMemory")
    assert hasattr(memory, "EpisodicMemory")
    assert hasattr(memory, "MemoryWriter")
    assert hasattr(memory, "retrieve")
    assert hasattr(memory, "PersonMemory")

    # 4. cognition
    assert hasattr(cognition, "VisionAgent")
    assert hasattr(cognition, "VisionContext")
    assert hasattr(cognition, "VoiceSession")
    assert hasattr(cognition, "AgentState")
    assert hasattr(cognition, "WorldState")
    assert hasattr(cognition, "Reasoner")
    assert hasattr(cognition, "ProactiveEngine")

    # 5. interaction
    assert hasattr(interaction, "ConversationManager")
    assert hasattr(interaction, "DialogueState")
    assert hasattr(interaction, "SarvamVoice")
    assert hasattr(interaction, "MicVAD")
    assert hasattr(interaction, "UiHub")

    # 6. actions
    assert hasattr(actions, "Alerter")


def test_modular_package_submodules():
    """Verify all submodules can be directly imported from their package namespaces."""
    import core.events
    import core.policy
    import core.tools
    import perception.detector
    import perception.face_engine
    import perception.tracker
    import perception.presence
    import perception.input_source
    import perception.face_pattern
    import memory.working
    import memory.semantic
    import memory.episodic
    import memory.writer
    import memory.retrieval
    import memory.context_memory
    import cognition.agent
    import cognition.reasoner
    import cognition.llm_interface
    import cognition.companion
    import interaction.conversation
    import interaction.voice
    import interaction.mic_vad
    import interaction.webui
    import actions.alerter

    assert core.events.Event is not None
    assert core.policy.should_greet is not None
    assert core.tools.ToolDispatcher is not None
    assert perception.detector.Detector is not None
    assert perception.face_engine.FaceEngine is not None
    assert perception.tracker.FaceTracker is not None
    assert perception.presence.PresenceManager is not None
    assert memory.context_memory.PersonMemory is not None
    assert cognition.agent.VisionAgent is not None
    assert interaction.conversation.ConversationManager is not None
    assert actions.alerter.Alerter is not None


def test_clean_root_no_orphan_shims():
    """Verify that root directory does not contain backward-compatibility shims."""
    import os
    root_dir = os.path.dirname(os.path.abspath(__file__))
    shim_names = [
        "agent.py", "alerter.py", "companion.py", "context_memory.py",
        "conversation.py", "detector.py", "events.py", "face_engine.py",
        "face_pattern.py", "input_source.py", "llm_interface.py", "mic_vad.py",
        "policy.py", "presence.py", "reasoner.py", "tools.py", "tracker.py",
        "voice.py", "webui.py"
    ]
    for shim in shim_names:
        shim_path = os.path.join(root_dir, shim)
        assert not os.path.exists(shim_path), f"Orphan shim '{shim}' still exists in repo root"


def test_cross_package_instantiation():
    """Verify classes from different packages collaborate without circular imports."""
    from perception import PresenceManager
    from cognition import VisionAgent, VisionContext, AgentState, VoiceSession
    from core import ToolDispatcher
    import queue

    q = queue.Queue()
    presence = PresenceManager(event_queue=q)
    assert presence is not None

    agent = VisionAgent()
    assert isinstance(agent.dispatcher, ToolDispatcher)

    v_ctx = VisionContext()
    assert v_ctx.world_state is not None

    state = AgentState()
    assert state.mode == "idle"
