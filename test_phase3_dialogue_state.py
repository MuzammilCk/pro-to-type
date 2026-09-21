import time
import pytest
import numpy as np

from interaction.conversation import (
    ConversationManager,
    DialogueState,
    InvalidTransitionError,
    can_transition,
    TRANSITION_TABLE,
)
from cognition.agent import AgentState, VoiceSession


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------

class FakeVoice:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text, interrupt=True):
        self.spoken.append(text)

    def speak_async(self, text):
        self.spoken.append(text)

    def enqueue_speech(self, text):
        self.spoken.append(text)

    def listen(self, timeout=10, phrase_limit=8, mic=None):
        return ""


class FakeAgent:
    def __init__(self):
        self.memory = {}

    def _memory_for(self, identity):
        if identity not in self.memory:
            m = type("M", (), {})()
            m.persona = type("P", (), {})()
            m.persona.name = None
            m.persona.purpose = None
            m.interaction_count = 0
            m.add_interaction = lambda *args: None
            m.save = lambda: None
            self.memory[identity] = m
        return self.memory[identity]

    def stream_response(self, identity, face_data, user_input=None, frame=None):
        yield "Hello there! What brings you here?", "ask"

    def think(self, identity, face_data, user_input=None):
        return f"Welcome, {identity}!", "ask"


class FakeFaceEngine:
    def __init__(self):
        self.known_faces = {}


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_greeting_to_enrolling_and_verify_to_enrolling_with_unknown_succeeds():
    """Named requirement: GREETING -> ENROLLING and VERIFY -> ENROLLING with

    active_person="unknown" must succeed. This is the single most common
    real-world enrollment path and must be explicitly tested.
    """
    # 1. Pure function assertion
    ok, reason = can_transition(DialogueState.GREETING, DialogueState.ENROLLING, active_person="unknown")
    assert ok is True, f"GREETING -> ENROLLING failed: {reason}"

    ok, reason = can_transition(DialogueState.VERIFY, DialogueState.ENROLLING, active_person="unknown")
    assert ok is True, f"VERIFY -> ENROLLING failed: {reason}"

    # 2. ConversationManager path: GREETING -> ENROLLING
    conv = ConversationManager(FakeAgent(), FakeVoice(), FakeFaceEngine())
    greeting = conv.start_for("unknown", {"authorized": False, "score": 0.5})
    assert greeting.strip()
    assert conv.state == DialogueState.GREETING
    assert conv.current_identity == "unknown"
    assert conv.agent_state.active_person == "unknown"
    assert conv.agent_state.active_goal == "greet"

    success = conv.transition_to(DialogueState.ENROLLING)
    assert success is True
    assert conv.state == DialogueState.ENROLLING
    assert conv.agent_state.active_goal == "enroll"
    assert conv.agent_state.mode == "conversing"

    # 3. ConversationManager path: GREETING -> VERIFY -> ENROLLING
    conv.reset()
    assert conv.state == DialogueState.IDLE
    conv.start_for("unknown", {"authorized": False, "score": 0.5})
    assert conv.state == DialogueState.GREETING

    ok_verify = conv.transition_to(DialogueState.VERIFY)
    assert ok_verify is True
    assert conv.state == DialogueState.VERIFY
    assert conv.agent_state.active_goal == "verify"

    ok_enroll = conv.transition_to(DialogueState.ENROLLING)
    assert ok_enroll is True
    assert conv.state == DialogueState.ENROLLING
    assert conv.agent_state.active_goal == "enroll"


def test_illegal_transition_idle_to_enrolling_no_person_rejected_and_logged(capsys):
    """Roadmap Definition of Done:

    Assert an illegal transition (e.g. IDLE -> ENROLLING with no person present)
    is rejected and logged, not silently allowed.
    """
    conv = ConversationManager(FakeAgent(), FakeVoice(), FakeFaceEngine())
    assert conv.state == DialogueState.IDLE
    assert conv.current_identity is None

    # (a) Raising error when requested (default behavior)
    with pytest.raises(InvalidTransitionError) as exc_info:
        conv.transition_to(DialogueState.ENROLLING)

    assert "Cannot transition to ENROLLING with no person present" in str(exc_info.value)
    assert conv.state == DialogueState.IDLE, "State must remain IDLE after rejected transition"

    # (b) Raising error via property setter
    with pytest.raises(InvalidTransitionError):
        conv.state = DialogueState.ENROLLING
    assert conv.state == DialogueState.IDLE

    # (c) Graceful non-raising mode logs warning and returns False
    res = conv.transition_to(DialogueState.ENROLLING, raise_error=False)
    assert res is False
    assert conv.state == DialogueState.IDLE

    captured = capsys.readouterr().out
    assert "[DialogueState] Illegal transition" in captured
    assert "rejected" in captured
    assert "Cannot transition to ENROLLING with no person present" in captured


def test_illegal_transitions_table_enforcement():
    """Assert invalid transitions across states are rejected."""
    conv = ConversationManager(FakeAgent(), FakeVoice(), FakeFaceEngine())

    # IDLE cannot jump directly to VERIFY or DECIDE even if a name is forced
    with pytest.raises(InvalidTransitionError):
        conv.transition_to(DialogueState.VERIFY, identity="alice")
    assert conv.state == DialogueState.IDLE

    with pytest.raises(InvalidTransitionError):
        conv.transition_to(DialogueState.DECIDE, identity="alice")
    assert conv.state == DialogueState.IDLE

    # Transition to non-IDLE states with empty person is always rejected
    with pytest.raises(InvalidTransitionError):
        conv.transition_to(DialogueState.GREETING, identity="")
    assert conv.state == DialogueState.IDLE

    # Start conversation
    conv.start_for("unknown", {"authorized": False, "score": 0.5})
    assert conv.state == DialogueState.GREETING

    # Advance to ENROLLING
    conv.transition_to(DialogueState.ENROLLING)
    assert conv.state == DialogueState.ENROLLING

    # ENROLLING cannot transition backward to GREETING, VERIFY, or DECIDE
    with pytest.raises(InvalidTransitionError):
        conv.transition_to(DialogueState.GREETING)
    assert conv.state == DialogueState.ENROLLING

    with pytest.raises(InvalidTransitionError):
        conv.transition_to(DialogueState.VERIFY)
    assert conv.state == DialogueState.ENROLLING

    with pytest.raises(InvalidTransitionError):
        conv.transition_to(DialogueState.DECIDE)
    assert conv.state == DialogueState.ENROLLING

    # But ENROLLING can transition to IDLE (e.g. upon completion or cancel)
    conv.transition_to(DialogueState.IDLE)
    assert conv.state == DialogueState.IDLE


def test_full_valid_lifecycle():
    """Verify standard happy-path progression through all dialogue states."""
    conv = ConversationManager(FakeAgent(), FakeVoice(), FakeFaceEngine())
    assert conv.state == DialogueState.IDLE

    # 1. IDLE -> GREETING
    conv.start_for("unknown", {"authorized": False, "score": 0.5})
    assert conv.state == DialogueState.GREETING

    # 2. GREETING -> VERIFY
    assert conv.transition_to(DialogueState.VERIFY) is True
    assert conv.state == DialogueState.VERIFY

    # 3. VERIFY -> DECIDE
    assert conv.transition_to(DialogueState.DECIDE) is True
    assert conv.state == DialogueState.DECIDE

    # 4. DECIDE -> ENROLLING
    assert conv.transition_to(DialogueState.ENROLLING) is True
    assert conv.state == DialogueState.ENROLLING

    # 5. ENROLLING -> IDLE (via reset)
    conv.reset()
    assert conv.state == DialogueState.IDLE
    assert conv.agent_state.mode == "idle"
    assert conv.agent_state.active_person is None


def test_agent_state_wiring_and_synchronization():
    """Verify AgentState is shared and synchronized between ConversationManager and VoiceSession."""
    shared_state = AgentState()
    voice = FakeVoice()
    agent = FakeAgent()

    conv = ConversationManager(agent, voice, FakeFaceEngine(), agent_state=shared_state)
    session = VoiceSession(agent, voice, mic=None, vision=None, agent_state=shared_state)

    assert conv.agent_state is shared_state
    assert session.agent_state is shared_state
    assert shared_state.mode == "idle"
    assert shared_state.active_person is None
    assert shared_state.active_goal is None

    # conv starts conversation
    conv.start_for("unknown", {"authorized": False, "score": 0.5})
    assert shared_state.mode == "conversing"
    assert shared_state.active_person == "unknown"
    assert shared_state.active_goal == "greet"

    # session touches activity
    t_prev = shared_state.last_activity
    time.sleep(0.01)
    session.touch_activity()
    assert shared_state.last_activity > t_prev

    # conv resets
    conv.reset()
    assert shared_state.mode == "idle"
    assert shared_state.active_person is None
    assert shared_state.active_goal is None


def test_dialogue_state_backwards_compatibility():
    """Ensure DialogueState maintains backwards compatibility with str."""
    assert DialogueState.IDLE == "IDLE"
    assert DialogueState.GREETING == "GREETING"
    assert DialogueState.VERIFY == "VERIFY"
    assert DialogueState.DECIDE == "DECIDE"
    assert DialogueState.ENROLLING == "ENROLLING"

    assert isinstance(DialogueState.IDLE, str)
    assert DialogueState.IDLE in ("IDLE", "GREETING")

    # can convert from string cleanly
    assert DialogueState("IDLE") is DialogueState.IDLE
    assert DialogueState("ENROLLING") is DialogueState.ENROLLING
