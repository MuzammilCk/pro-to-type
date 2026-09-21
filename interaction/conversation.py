import time
from enum import Enum
from memory.context_memory import PersonMemory, SessionManager
from cognition.agent import AgentState


class DialogueState(str, Enum):
    """Explicit conversation states for the ARIA dialogue state machine."""
    IDLE = "IDLE"           # No active dialogue
    GREETING = "GREETING"   # Initial hello + identity inquiry
    VERIFY = "VERIFY"       # Inquire purpose / visitor credentials
    DECIDE = "DECIDE"       # Evaluate next step (enroll vs alert vs escalate)
    ENROLLING = "ENROLLING" # Biometric facial sample capture


class InvalidTransitionError(ValueError):
    """Raised when an illegal dialogue state transition is attempted."""
    pass


TRANSITION_TABLE: dict[DialogueState, set[DialogueState]] = {
    DialogueState.IDLE: {
        DialogueState.IDLE,
        DialogueState.GREETING,
    },
    DialogueState.GREETING: {
        DialogueState.GREETING,
        DialogueState.VERIFY,
        DialogueState.DECIDE,
        DialogueState.ENROLLING,
        DialogueState.IDLE,
    },
    DialogueState.VERIFY: {
        DialogueState.VERIFY,
        DialogueState.DECIDE,
        DialogueState.ENROLLING,
        DialogueState.IDLE,
    },
    DialogueState.DECIDE: {
        DialogueState.DECIDE,
        DialogueState.ENROLLING,
        DialogueState.IDLE,
    },
    DialogueState.ENROLLING: {
        DialogueState.ENROLLING,
        DialogueState.IDLE,
    },
}

TRANSITIONS = TRANSITION_TABLE


def can_transition(
    current_state: DialogueState | str,
    target_state: DialogueState | str,
    active_person: str | None = None,
) -> tuple[bool, str]:
    """Validate whether transitioning from current_state to target_state is legal.

    Returns (allowed: bool, reason: str).
    """
    try:
        curr = DialogueState(current_state)
        tgt = DialogueState(target_state)
    except ValueError as e:
        return False, f"Invalid state value: {e}"

    # Target non-IDLE states require an active person present
    if tgt != DialogueState.IDLE and not active_person:
        return False, f"Cannot transition to {tgt.name} with no person present"

    allowed_targets = TRANSITION_TABLE.get(curr, set())
    if tgt not in allowed_targets:
        return False, f"Illegal transition: {curr.name} -> {tgt.name} not permitted in transition table"

    return True, "ok"


class ConversationManager:
    """Manages dialogue flow for an unknown visitor encounter.

    States:
      1. IDLE      — no active conversation
      2. GREETING  — initial hello + name request
      3. VERIFY    — ask purpose / business
      4. DECIDE    — agent decides enroll vs alert
      5. ENROLLING — capturing face samples for enrollment

    Integration patterns from research:
    - Full-duplex: voice I/O and vision run on separate threads sharing state via queues
    - Streaming STT yields partial transcripts for barge-in support
    - Event-driven vision: triggers face capture on linguistic cues, not per-frame
    - Session checkpointing: state saved/restored between runs
    """

    STATES = tuple(DialogueState)
    TRANSITION_TABLE = TRANSITION_TABLE
    TRANSITIONS = TRANSITION_TABLE

    def __init__(
        self,
        agent,
        voice,
        face_engine,
        vision_queue=None,
        frame_provider=None,
        agent_state: AgentState | None = None,
    ):
        self.agent = agent
        self.voice = voice
        self.face_engine = face_engine
        self.vision_queue = vision_queue
        self.frame_provider = frame_provider  # Callable[[], np.ndarray | None]
        self.agent_state = agent_state if agent_state is not None else AgentState()
        self._state = DialogueState.IDLE
        self.current_identity = None
        self.turn_count = 0
        self._state_just_changed = False

    @property
    def state(self) -> DialogueState:
        return self._state

    @state.setter
    def state(self, new_state: DialogueState | str):
        self.transition_to(new_state, raise_error=True)

    def transition_to(
        self,
        new_state: DialogueState | str,
        identity: str | None = None,
        raise_error: bool = True,
    ) -> bool:
        """Advance dialogue state following the explicit transition table.

        If raise_error is True, raises InvalidTransitionError on illegal transitions.
        If raise_error is False, logs a warning, keeps current state, and returns False.
        """
        active = (
            identity
            or self.current_identity
            or (self.agent_state.active_person if self.agent_state else None)
        )
        ok, reason = can_transition(self._state, new_state, active_person=active)
        if not ok:
            msg = f"[DialogueState] Illegal transition: {self._state} -> {new_state} rejected ({reason})"
            print(msg)
            if raise_error:
                raise InvalidTransitionError(msg)
            return False

        self._state = DialogueState(new_state)
        if self.agent_state:
            self.agent_state.last_activity = time.time()
            if self._state == DialogueState.IDLE:
                self.agent_state.mode = "idle"
                self.agent_state.active_goal = None
            else:
                self.agent_state.mode = "conversing"
                if self._state == DialogueState.ENROLLING:
                    self.agent_state.active_goal = "enroll"
                elif self._state == DialogueState.VERIFY:
                    self.agent_state.active_goal = "verify"
                elif self._state == DialogueState.GREETING:
                    self.agent_state.active_goal = "greet"
                elif self._state == DialogueState.DECIDE:
                    self.agent_state.active_goal = "decide"
        return True

    def start_for(self, identity: str, face_data: dict) -> str:
        """Begin a new unknown-visitor conversation.

        Returns the greeting text — the caller owns voice playback (B1 fix:
        speaking here AND in the caller made the greeting play twice).
        """
        if identity != "unknown":
            self.transition_to(DialogueState.IDLE, identity=identity)
            return ""

        self.current_identity = identity
        if self.agent_state:
            self.agent_state.active_person = identity
            self.agent_state.mode = "conversing"
            self.agent_state.active_goal = "greet"
        self.transition_to(DialogueState.GREETING, identity=identity)
        self.turn_count = 0
        self._state_just_changed = True

        # Use streaming response for lowest latency
        response = ""
        if hasattr(self.agent, "stream_response"):
            frame = self.frame_provider() if self.frame_provider else None
            for token, action in self.agent.stream_response(identity, face_data, None, frame):
                response += token
            response = response.strip()
        else:
            response, action = self.agent.think(identity, face_data)

        self._trigger_vision_event("capture_face", face_data)
        return response

    def handle_response(self, user_input: str, identity: str, face_data: dict) -> tuple[str, str]:
        """Process user speech input and advance the state machine."""
        if self._state == DialogueState.IDLE:
            return "", "none"

        self._state_just_changed = False
        response = ""
        action = "ask"

        # Streaming response with event detection during token yield
        if hasattr(self.agent, "stream_response"):
            token_stream = self.agent.stream_response(identity, face_data, user_input)
            for token, detected_action in token_stream:
                response += token
                if detected_action != "ask":
                    action = detected_action
            response = response.strip()
        else:
            response, action = self.agent.think(identity, face_data, user_input)

        self.turn_count += 1

        # State transitions
        if action == "enroll":
            if self._state != DialogueState.ENROLLING:
                self.transition_to(DialogueState.ENROLLING, identity=identity)
                self._state_just_changed = True
                self._trigger_vision_event("enroll_face", face_data)
        elif action == "alert":
            if self._state != DialogueState.DECIDE:
                self.transition_to(DialogueState.DECIDE, identity=identity)
                self._trigger_vision_event("escalate", face_data)
        elif self.turn_count >= 6:
            action = "escalate"
            self.transition_to(DialogueState.DECIDE, identity=identity)

        # Auto-transition after name is learned
        if not self._state_just_changed:
            mem = self.agent._memory_for(identity) if hasattr(self.agent, "_memory_for") else None
            if mem and mem.persona.name and self._state == DialogueState.GREETING:
                self.transition_to(DialogueState.VERIFY, identity=identity)

        # Checkpoint session state
        self._checkpoint()

        return response, action

    def reset(self):
        self._state = DialogueState.IDLE
        self.current_identity = None
        self.turn_count = 0
        self._state_just_changed = False
        if self.agent_state:
            self.agent_state.mode = "idle"
            self.agent_state.active_person = None
            self.agent_state.active_goal = None
            self.agent_state.last_activity = time.time()

    def _trigger_vision_event(self, event_type: str, face_data: dict):
        """Signal vision thread to perform an action (event-driven vision).

        Per research: don't run vision continuously — trigger on conversation
        events like name learned, decision to enroll, suspicious input.
        """
        if self.vision_queue is not None:
            self.vision_queue.put({
                "type": event_type,
                "face_data": face_data,
                "timestamp": self._now_iso(),
            })

    def _checkpoint(self):
        """Save session state for resume capability."""
        if hasattr(self.agent, "_memory_for"):
            mem = self.agent._memory_for(self.current_identity or "unknown")
            SessionManager.save_current_state(
                {self.current_identity or "unknown": mem}
            )

    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime
        return datetime.now().isoformat()
