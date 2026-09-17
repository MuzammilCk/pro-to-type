from context_memory import PersonMemory, SessionManager


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

    STATES = ("IDLE", "GREETING", "VERIFY", "DECIDE", "ENROLLING")

    def __init__(self, agent, voice, face_engine, vision_queue=None, frame_provider=None):
        self.agent = agent
        self.voice = voice
        self.face_engine = face_engine
        self.vision_queue = vision_queue
        self.frame_provider = frame_provider  # Callable[[], np.ndarray | None]
        self.state = "IDLE"
        self.current_identity = None
        self.turn_count = 0
        self._state_just_changed = False

    def start_for(self, identity: str, face_data: dict) -> str:
        """Begin a new unknown-visitor conversation.

        Returns the greeting text — the caller owns voice playback (B1 fix:
        speaking here AND in the caller made the greeting play twice).
        """
        if identity != "unknown":
            self.state = "IDLE"
            return ""

        self.state = "GREETING"
        self.current_identity = identity
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
        if self.state == "IDLE":
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
            if self.state != "ENROLLING":
                self.state = "ENROLLING"
                self._state_just_changed = True
                self._trigger_vision_event("enroll_face", face_data)
        elif action == "alert":
            if self.state != "DECIDE":
                self.state = "DECIDE"
                self._trigger_vision_event("escalate", face_data)
        elif self.turn_count >= 6:
            action = "escalate"
            self.state = "DECIDE"

        # Auto-transition after name is learned
        if not self._state_just_changed:
            mem = self.agent._memory_for(identity) if hasattr(self.agent, "_memory_for") else None
            if mem and mem.persona.name and self.state == "GREETING":
                self.state = "VERIFY"

        # Checkpoint session state
        self._checkpoint()

        return response, action

    def reset(self):
        self.state = "IDLE"
        self.current_identity = None
        self.turn_count = 0
        self._state_just_changed = False

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
