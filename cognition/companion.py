"""ProactiveEngine — ARIA starts conversations on her own (Phase 3).

GPT-Live's conversational doctrine applied to a *companion*: the system
doesn't just answer — it notices, follows up, and keeps light chatter going
when the human goes quiet. This engine runs on the conversation thread's
idle time and produces spontaneous spoken lines grounded in:

  * what ARIA currently sees (VisionContext — people, objects, strangers)
  * what she remembers (PersonMemory — names, past purposes, tags)
  * cooldowns so she never becomes an annoying chatterbox

Everything here is speech-only; it never mutates recognition/presence state.
"""
import random
import threading
import time

OBSERVATIONS = [
    "You know, I've been watching the room — it's nice having some company.",
    "Fun fact: I can see {count} thing(s) in the room right now. I feel like a very nosey houseplant.",
    "If you're wondering what else I can do — ask me anything. I love a challenge.",
    "You've got a good energy about you today.",
    "I noticed you're still around — thanks for keeping me company.",
]


class ProactiveEngine:
    """Occasional spontaneous remarks, grounded in vision + memory."""

    MIN_INTERVAL = 90.0     # never chatter more often than this
    MAX_PER_HOUR = 6
    ONLY_WHEN_SEEN = True   # stay silent if nobody is in view

    def __init__(self, agent, voice, vision, session=None,
                 followup_prob: float = 0.4):
        self.agent = agent
        self.voice = voice
        self.vision = vision
        self.session = session            # VoiceSession, if running
        self.followup_prob = followup_prob  # memory follow-up probability
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_spoke = 0.0
        self._hour_count: list[float] = []

    # ------------------------------------------------------------------

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="ProactiveEngine")
        self._thread.start()

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------

    def _loop(self):
        while not self._stop.is_set():
            self._stop.wait(timeout=self._next_delay())
            if self._stop.is_set():
                return
            line = self.make_remark()
            if line:
                self.voice.enqueue_speech(line)
                self._last_spoke = time.time()
                self._hour_count = [t for t in self._hour_count
                                    if time.time() - t < 3600]
                self._hour_count.append(time.time())
                if self.session is not None:
                    self.session.history.append(
                        {"role": "assistant", "content": line})
                    self.session.history = self.session.history[-12:]

    def _next_delay(self) -> float:
        base = self.MIN_INTERVAL + random.uniform(0, 60)
        return max(30.0, base)

    def _allowed(self) -> bool:
        from core.policy import should_proactive_remark
        is_busy = bool(self.session is not None and getattr(self.session, "busy", False))
        agent_state = getattr(self.session, "agent_state", None) if self.session else None
        world_state = getattr(self.vision, "world_state", None) if self.vision else None

        # Ensure faces representation matches vision context
        faces = []
        if self.vision is not None:
            summary = self.vision.context_text()
            if "No one is in view" not in summary and "No vision input" not in summary:
                faces = [{"track_id": "face_0"}]
        if world_state is not None:
            if not getattr(world_state, "faces", []):
                world_state.faces = faces
        else:
            class _WS:
                def __init__(self, f):
                    self.faces = f
            world_state = _WS(faces)

        decision = should_proactive_remark(
            world_state=world_state,
            agent_state=agent_state,
            last_spoke_time=self._last_spoke,
            hour_timestamps=self._hour_count,
            min_interval_sec=self.MIN_INTERVAL,
            max_per_hour=self.MAX_PER_HOUR,
            only_when_seen=self.ONLY_WHEN_SEEN,
            is_busy=is_busy,
        )
        if decision.allowed:
            now = time.time()
            self._hour_count = [t for t in self._hour_count if now - t < 3600]
            return True
        return False

    # ------------------------------------------------------------------

    def make_remark(self) -> str | None:
        """Produce one grounded spontaneous line, or None if not allowed."""
        if not self._allowed():
            return None

        # Prefer a memory-driven follow-up occasionally
        try:
            known = [n for n in self.agent.memory
                     if n not in ("unknown",) and self.agent.memory[n].interaction_count > 0]
        except Exception:
            known = []
        if known and random.random() < self.followup_prob:
            name = known[0]
            mem = self.agent.memory[name]
            if mem.persona.purpose:
                return (f"Hey {mem.persona.name or name}, earlier you mentioned "
                        f"{str(mem.persona.purpose)[:40]} — how did that go?")

        if self.agent.llm.available and self.vision is not None:
            try:
                prompt = (
                    "Make ONE brief, warm observational remark about the room "
                    "or the visitor, in a companion's voice. Max 2 short "
                    "sentences, no questions, no lists. Context: "
                    + self.vision.context_text()
                )
                msgs = [{"role": "user", "content": prompt}]
                out = []
                for tok in self.agent.llm.stream(msgs):
                    out.append(tok)
                text = "".join(out).strip()
                if text and "delegate" not in text.lower():
                    return text[:220]
            except Exception:
                pass

        count = 0
        if self.vision is not None:
            summary = self.vision.summary
            digits = "".join(c for c in summary if c.isdigit())
            count = int(digits) if digits else 1
        return random.choice(OBSERVATIONS).format(count=count)
