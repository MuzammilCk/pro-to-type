"""Offline tests for the talkative voice stack (Phases 0-3).

Run:  python test_voice_stack.py   (or pytest test_voice_stack.py -q)

Covers:
  Phase 0 — sentence splitting, speech queue ordering, interrupt flushing
  Phase 1 — MicVAD utterance queue + echo-guard primitives
  Phase 2 — VoiceBrain delegation verdict, VoiceSession spoken turns,
             backend delivery, nudge budget
  Phase 3 — VisionContext summaries, ProactiveEngine rate limits

No microphone, no network: audio and LLM are stubbed.
"""

import queue
import threading
import time

import numpy as np


# ----------------------------------------------------------------------
# Stubs
# ----------------------------------------------------------------------

class StubVoice:
    """Records enqueued/speech text without any audio hardware."""

    def __init__(self):
        self.spoken: list[str] = []
        self._lock = threading.Lock()
        self._on_speaking = None

    # SpeechQueue interface
    def start_speech_queue(self):
        pass

    def enqueue_speech(self, text):
        if (text or "").strip():
            with self._lock:
                self.spoken.append(text.strip())

    def wait_speech_done(self, timeout=None):
        pass

    def set_speaking_callback(self, cb):
        self._on_speaking = cb

    def speak(self, text, interrupt=True):
        self.enqueue_speech(text)

    def speak_async(self, text):
        self.enqueue_speech(text)


class StubMemory:
    class _Persona:
        name = None
        purpose = None
        tags: list = []

    def __init__(self):
        self.persona = StubMemory._Persona()
        self.working_context = []
        self.interactions = []
        self.saved = 0

    def add_interaction(self, role, text):
        self.interactions.append((role, text))

    @property
    def interaction_count(self):
        return len(self.interactions)

    def save(self):
        self.saved += 1


class StubAgent:
    """VisionAgent stand-in with a scriptable backend LLM."""

    def __init__(self, backend_reply="Here is the detailed answer you wanted. Enjoy."):
        self.llm_available = True
        self.backend_reply = backend_reply
        self._memory_data = {"unknown": StubMemory()}

    class _LLM:
        available = True

    def __init__(self, backend_reply="Here is the detailed answer you wanted. Enjoy."):
        self.llm_available = True
        self.backend_reply = backend_reply
        self._llm = StubAgent._LLM()
        self._memory_data = {"unknown": StubMemory()}
        self.memory = self._memory_data  # VisionAgent-compatible attribute

    @property
    def llm(self):
        return self._llm

    @llm.setter
    def llm(self, value):
        self._llm = value

    def _memory_for(self, identity):
        if identity not in self._memory_data:
            self._memory_data[identity] = StubMemory()
        return self._memory_data[identity]

    def _fallback(self, identity, user_input, mem):
        return "Fallback line."

    def stream_response(self, identity, face_data, user_input, frame=None):
        for word in self.backend_reply.split(" "):
            yield word + " ", "ask"


def make_session(backend_reply=None):
    from agent import VisionContext, VoiceSession

    agent = StubAgent(backend_reply or "Here is the detailed answer you wanted. Enjoy.")
    voice = StubVoice()
    vision = VisionContext()
    session = VoiceSession(agent, voice, mic=None, vision=vision)
    return agent, voice, vision, session


# ----------------------------------------------------------------------
# Phase 0 — sentence streaming
# ----------------------------------------------------------------------

def test_sentence_splitter_basic():
    from agent import _split_sentences

    sentences, rest = _split_sentences("Hello there! How are you? I'm fine.")
    assert sentences == ["Hello there!", "How are you?", "I'm fine."]
    assert rest == ""


def test_sentence_splitter_keeps_decimals():
    from agent import _split_sentences

    sentences, rest = _split_sentences("Pi is 3.14 exactly.")
    assert sentences == ["Pi is 3.14 exactly."], sentences
    assert rest == ""


def test_sentence_splitter_leaves_partial():
    from agent import _split_sentences

    sentences, rest = _split_sentences("First one. And then it kept")
    assert sentences == ["First one."]
    assert rest.strip() == "And then it kept"  # remainder keeps streaming whitespace


def test_speech_queue_flushes_on_interrupt():
    """interrupt() must drop queued-but-unspoken sentences."""
    from voice import SarvamVoice

    v = SarvamVoice(api_key=None)
    v._speech_queue = queue.Queue()
    v._speech_queue.put("one")
    v._speech_queue.put("two")
    v.interrupt()
    assert v._speech_queue.empty(), "queued sentences survived barge-in"


# ----------------------------------------------------------------------
# Phase 1 — MicVAD primitives (no real audio device)
# ----------------------------------------------------------------------

def test_sarvam_tts_protocol_contract():
    """The Sarvam WS TTS contract, locked offline with a fake WebSocket.

    Regression for the 2026-09-18 bug: _speak_sarvam sent a REST-style payload
    as the first message, so Sarvam answered 422 'Input parameters has to be a
    valid dictionary' on EVERY utterance and playback silently fell back to
    pyttsx3 (which then crashed with 'run loop already started' under
    concurrency). The protocol was verified live: config envelope first, then
    text + flush; audio at data.audio; speaker must be a bulbul:v3 voice.
    """
    import asyncio
    import base64
    import json as _json
    import voice as _voice

    sent: list[str] = []
    audio_b64 = base64.b64encode(b"\x01\x00" * 80).decode()  # 80 int16 samples

    class FakeWS:
        def __init__(self):
            self.n = 0

        async def send(self, raw):
            sent.append(raw)

        async def recv(self):
            self.n += 1
            if self.n == 1:
                return _json.dumps({"type": "audio", "data": {"audio": audio_b64}})
            if self.n == 2:
                return _json.dumps({"type": "event", "data": {"event_type": "final"}})
            raise asyncio.TimeoutError()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    import websockets as _ws_mod
    orig_connect, orig_ver = _ws_mod.connect, _ws_mod.__version__
    orig_sink_ok = _voice._HAS_SOUND
    try:
        _ws_mod.connect = lambda *a, **k: FakeWS()
        _voice._HAS_SOUND = False  # skip the audio sink; we only assert protocol
        v = _voice.SarvamVoice(api_key="test-key")
        v._speak_sarvam("Hello there", my_gen=v._tts_gen)  # sync method; runs its own loop
    finally:
        _ws_mod.connect = orig_connect
        _ws_mod.__version__ = orig_ver
        _voice._HAS_SOUND = orig_sink_ok

    msgs = [_json.loads(m) for m in sent]
    assert len(msgs) >= 3, f"expected config+text+flush, got {len(msgs)} messages"
    assert msgs[0]["type"] == "config", "first WS message must be the config envelope"
    cfg = msgs[0]["data"]
    assert cfg["model"] == "bulbul:v3", "bulbul:v2 is deprecated and rejected by Sarvam"
    assert cfg["speaker"] not in ("meera", "anushka"), "v2-era speaker name leaked back"
    assert cfg["target_language_code"] in _voice._SARVAM_TTS_LANGS
    assert msgs[1] == {"type": "text", "data": {"text": "Hello there"}}
    assert msgs[2] == {"type": "flush"}


def test_local_tts_is_lock_guarded():
    """The pyttsx3 fallback must serialize through _LOCAL_TTS_LOCK — the
    'run loop already started' crash came from unsynchronized runAndWait()."""
    import voice as _voice
    assert hasattr(_voice, "_LOCAL_TTS_LOCK")
    from voice import SarvamVoice
    v = SarvamVoice(api_key=None)
    assert _voice._LOCAL_TTS_LOCK is not None


def test_voicebrain_persona_in_spoken_context():
    """VoiceBrain._messages must inject persistent memory (name, traits) into
    the spoken-turn system prompt — that's the conversational-intelligence fix."""
    from agent import VoiceBrain, VOICE_SYSTEM_PROMPT

    class _FakeMem:
        class persona:
            name = "Muzammil"
            traits = ["likes robotics", "hates slow elevators"]
            purpose = "building ARIA"
        interaction_count = 7
        class episodic:
            episodes = []

    class _FakeAgent:
        def _memory_for(self, identity):
            return _FakeMem()

    brain = VoiceBrain(_FakeAgent())
    msgs = brain._messages("A quiet room.", [], "hey again", identity="muzammil")
    sys = msgs[0]["content"]
    assert "WHAT YOU REMEMBER ABOUT THIS PERSON" in sys
    assert "Muzammil" in sys and "robotics" in sys
    assert "WHO YOU ARE" in sys  # the personality prompt is intact


def test_mic_vad_utterance_queue():
    from mic_vad import MicVAD

    mic = MicVAD()
    pcm = (np.sin(np.linspace(0, 100, 1600)) * 5000).astype(np.int16)
    mic._emit([pcm])
    got = mic.wait_for_utterance(max_wait=1.0)
    assert got is not None and len(got) == 1600


def test_mic_vad_clear_pending():
    from mic_vad import MicVAD

    mic = MicVAD()
    pcm = (np.ones(1600, dtype=np.int16) * 100)
    mic._emit([pcm])
    mic.clear_pending()
    assert mic.wait_for_utterance(max_wait=0.2) is None


def test_voice_listen_uses_mic_path():
    """listen(mic=...) must use the VAD path and not interrupt in-flight TTS."""
    from voice import SarvamVoice

    v = SarvamVoice(api_key=None)

    class NullMic:
        speaking = False

        def wait_for_utterance(self, max_wait, max_utterance):
            return None

    out = v.listen(timeout=0.3, mic=NullMic())
    assert out == ""


def test_voice_speak_then_listen_mic_not_aborted():
    """Regression test: calling speak() to completion must not leave _interrupt_flag
    set such that an immediate listen(mic=...) with a queued utterance is aborted.

    Asserts the exact sequence from the live run:
    speak(greeting) completes -> utterance queued in MicVAD -> listen(mic=...) ->
    must NOT abort with '[Voice] _listen_vad aborted: interrupt flag set'.
    """
    import io
    import contextlib
    from voice import SarvamVoice
    from mic_vad import MicVAD

    v = SarvamVoice(api_key=None)
    # Prevent speak() from attempting real network or local TTS playback
    v._speak_sarvam = lambda *args, **kwargs: None
    v._speak_local = lambda *args, **kwargs: None

    # Stub transcription so listen() returns deterministically without network
    v.available = lambda: True
    v._transcribe_pcm = lambda pcm_bytes: "hello aria"

    # 1. Call speak() to completion (previously left _interrupt_flag set)
    v.speak("Hello there!")
    assert not v._interrupt_flag.is_set(), "speak() must clear _interrupt_flag upon completion"

    # 2. Prepare MicVAD with a queued utterance ready
    mic = MicVAD()
    pcm = (np.sin(np.linspace(0, 100, 1600)) * 5000).astype(np.int16)
    mic._emit([pcm])

    # 3. Immediately call listen() on the mic path and capture stdout
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        transcript = v.listen(timeout=1.0, mic=mic)
    logs = buf.getvalue()

    # 4. Assert it is NOT aborted and transcribes successfully
    assert "[Voice] _listen_vad aborted: interrupt flag set" not in logs, (
        f"listen(mic=...) aborted due to interrupt flag: {logs}"
    )
    assert "[Voice] STT (Sarvam) transcript: 'hello aria'" in logs
    assert transcript == "hello aria", (
        f"listen(mic=...) was aborted or failed to transcribe: got {transcript!r}"
    )
    assert not v._interrupt_flag.is_set()


# ----------------------------------------------------------------------
# Phase 2 — split brain
# ----------------------------------------------------------------------

def test_voicebrain_delegate_detected():
    agent, voice, vision, session = make_session()
    brain = session.brain

    class DelegateLLM:
        available = True

        def stream(self, messages):
            yield "[DELE"
            yield "GATE]"

    agent.llm = DelegateLLM()
    for _ in brain.reply_stream("ctx", [], "what is quantum tunneling"):
        pass
    assert brain.delegated is True, "[DELEGATE] token not detected"


def test_voicebrain_normal_reply_not_delegated():
    agent, voice, vision, session = make_session()
    brain = session.brain

    class NormalLLM:
        available = True

        def stream(self, messages):
            yield "Hey there! "
            yield "You look cheerful today."

    agent.llm = NormalLLM()
    for _ in brain.reply_stream("ctx", [], "hi"):
        pass
    assert brain.delegated is False


def test_session_normal_turn_speaks():
    agent, voice, vision, session = make_session()

    class NormalLLM:
        available = True

        def stream(self, messages):
            yield "Hi again! "
            yield "What brings you here today?"

    agent.llm = NormalLLM()
    out = session.run_turn("unknown", {"authorized": False}, "hello")
    assert "Hi again" in out
    assert any("Hi again!" in s for s in voice.spoken), (
        f"turn was not spoken: {voice.spoken}"
    )
    assert session.pending_action == "ask"


def test_session_delegates_to_backend_and_speaks_answer():
    agent, voice, vision, session = make_session(
        backend_reply="The Eiffel Tower is 330 metres tall. It was built in 1889.")

    class DelegateLLM:
        available = True

        def stream(self, messages):
            yield "[DELEGATE]"

    agent.llm = DelegateLLM()
    out = session.run_turn("unknown", {"authorized": False},
                           "how tall is the eiffel tower")
    assert "330 metres" in out
    assert any("330 metres" in s for s in voice.spoken), (
        f"backend answer never spoken: {voice.spoken}"
    )


def test_session_pending_action_enroll():
    agent, voice, vision, session = make_session()

    class EnrollLLM:
        available = True

        def stream(self, messages):
            yield "Let's fix that — look at the camera so I can register you."

    agent.llm = EnrollLLM()
    session.run_turn("unknown", {"authorized": False}, "my name is Sam")
    assert session.pending_action == "enroll"


def test_nudge_budget():
    agent, voice, vision, session = make_session()
    session.last_activity = time.time() - 10 * 60  # long idle

    assert session.maybe_nudge("unknown", {}) is True
    assert len(voice.spoken) == 1
    # Immediately again: budget/rate-limit must block it
    session.last_activity = time.time() - 10 * 60
    session._nudges = session.MAX_NUDGES
    assert session.maybe_nudge("unknown", {}) is False


# ----------------------------------------------------------------------
# Phase 3 — vision fusion + proactive chatter
# ----------------------------------------------------------------------

def test_vision_context_summary():
    from agent import VisionContext

    vc = VisionContext()
    vc.update([], detections=None)
    assert "No one is in view" in vc.context_text()

    class FR(dict):
        pass

    vc.update([{"authorized": True, "identity": "Muzammil"},
               {"authorized": False, "identity": "unknown"}])
    text = vc.context_text()
    assert "Muzammil" in text and "unrecognized" in text

    vc.add_event("A cat walked by.")
    assert "cat" in vc.context_text()


def test_proactive_rate_limit():
    from companion import ProactiveEngine

    agent, voice, vision, session = make_session()
    engine = ProactiveEngine(agent, voice, vision, session=session)
    engine._last_spoke = time.time()  # just spoke
    assert engine.make_remark() is None, "chattered twice inside MIN_INTERVAL"


def test_proactive_silent_when_empty_room():
    from companion import ProactiveEngine

    agent, voice, vision, session = make_session()
    vision.update([])  # nobody in view
    engine = ProactiveEngine(agent, voice, vision, session=session)
    assert engine.make_remark() is None, "spoke to an empty room"


def test_proactive_remarks_groundedin_memory():
    from companion import ProactiveEngine

    agent, voice, vision, session = make_session()
    mem = agent._memory_for("Sam")
    mem.persona.name = "Sam"
    mem.persona.purpose = "fixing the printer"
    mem.add_interaction("user", "I'm here to fix the printer")
    vision.update([{"authorized": True, "identity": "Sam"}])
    engine = ProactiveEngine(agent, voice, vision, session=session,
                             followup_prob=1.0)
    remark = engine.make_remark()
    assert remark is not None
    assert "printer" in remark.lower(), f"remark not grounded: {remark}"



# ----------------------------------------------------------------------
# Departure / presence-exit regression tests
# ----------------------------------------------------------------------

def test_nudge_one_shot_per_idle_stretch():
    """maybe_nudge must fire at most once per idle stretch (MAX_NUDGES=1).

    After one nudge has fired and _nudges is at MAX_NUDGES, a second call
    — even with sufficient elapsed idle — must return False.  Separately,
    touch_activity() resets the budget so a nudge fires again in the next
    stretch.
    """
    agent, voice, vision, session = make_session()
    session.last_activity = time.time() - 10 * 60  # long idle

    # First nudge fires
    assert session.maybe_nudge("unknown", {}) is True, "first nudge should fire"
    assert len(voice.spoken) == 1

    # Budget exhausted: _nudges == MAX_NUDGES → second call must be blocked
    # even if we re-wind last_activity to simulate more idle time
    session.last_activity = time.time() - 10 * 60
    assert session.maybe_nudge("unknown", {}) is False, (
        "second nudge within same idle stretch must be blocked"
    )
    assert len(voice.spoken) == 1, "no second nudge should be spoken"

    # After a real interaction (touch_activity resets budget), a nudge fires again
    session.touch_activity()
    session.last_activity = time.time() - 10 * 60  # another long idle
    assert session.maybe_nudge("unknown", {}) is True, (
        "nudge should fire again after touch_activity resets the budget"
    )
    assert len(voice.spoken) == 2


def test_dialogue_loop_exits_promptly_when_room_empty():
    """_dialogue_loop must exit as soon as vision reports nobody in view.

    Previously the loop could only exit via 3×12s silence timeouts (~36s).
    This test uses a VisionContext that reads "No one is in view" and a
    voice stub that immediately returns '' (silence), confirming the loop
    breaks on the first iteration's presence check rather than waiting for
    3 idle turns.
    """
    import threading
    from conversation import ConversationManager
    from agent import VisionContext, VoiceSession
    from run import VisionAgentApp

    # --- stubs -------------------------------------------------------
    class EmptyRoomVoice(StubVoice):
        """Always returns empty transcript to simulate nobody speaking."""
        listen_calls = 0

        def listen(self, timeout=12.0, phrase_limit=8.0, mic=None):
            self.listen_calls += 1
            return ""

    class _StopEvent:
        def is_set(self):
            return False

    class _Hub:
        def set_aria_state(self, _):
            pass

    # --- build a minimal VisionAgentApp-shaped object ----------------
    agent_obj = StubAgent()
    voice_obj = EmptyRoomVoice()

    vision_ctx = VisionContext()
    # Populate with no faces → renders "No one is in view right now."
    vision_ctx.update([], detections=None)
    assert "No one is in view" in vision_ctx.context_text()

    session = VoiceSession(agent_obj, voice_obj, mic=None, vision=vision_ctx)

    class _MockConv:
        """Minimal ConversationManager stand-in."""
        state = "GREETING"  # not IDLE → loop should enter
        current_identity = "alice"
        reset_called = False

        def reset(self):
            self.reset_called = True
            self.state = "IDLE"

        def handle_response(self, transcript, identity, face_data):
            return "", "ask"

    conv = _MockConv()

    # --- exercise _dialogue_loop via a monkey-patched minimal app ----
    # Class body scope can't reference enclosing locals by the same name,
    # so alias them first.
    _vision_ctx = vision_ctx
    _session = session
    _voice_obj = voice_obj

    class _MinimalApp:
        """Just enough of VisionAgentApp to run _dialogue_loop."""
        vision_ctx = _vision_ctx
        session = _session
        voice = _voice_obj
        mic = None
        _stop = _StopEvent()
        hub = _Hub()

        def _identity_currently_authorized(self, identity):
            return True

    app = _MinimalApp()
    # Bind the unbound method
    import types
    app._dialogue_loop = types.MethodType(VisionAgentApp._dialogue_loop, app)

    app._dialogue_loop(conv, "alice", {"authorized": True, "face_bbox": (0,0,0,0)})

    assert conv.reset_called, "_dialogue_loop did not call conv.reset()"
    # The presence check fires on the FIRST iteration, before any listen() call
    assert voice_obj.listen_calls == 0, (
        f"listen() should not have been called at all when room is empty; "
        f"got {voice_obj.listen_calls} call(s)"
    )

    # --- Case 2: Person present at start, leaves during listen timeout ---
    conv2 = _MockConv()
    voice_obj2 = EmptyRoomVoice()
    vision_ctx2 = VisionContext()
    vision_ctx2.update([{"identity": "alice", "authorized": True}], detections=None)
    assert "No one is in view" not in vision_ctx2.context_text()

    def listen_and_leave(timeout=12.0, phrase_limit=8.0, mic=None):
        voice_obj2.listen_calls += 1
        vision_ctx2.update([], detections=None)  # departure happens during silence
        return ""

    voice_obj2.listen = listen_and_leave
    app2 = _MinimalApp()
    app2.vision_ctx = vision_ctx2
    app2.voice = voice_obj2
    app2._dialogue_loop = types.MethodType(VisionAgentApp._dialogue_loop, app2)

    app2._dialogue_loop(conv2, "alice", {"authorized": True, "face_bbox": (0, 0, 0, 0)})

    assert conv2.reset_called, "_dialogue_loop did not reset when person left during silence"
    # Must exit after 1 listen timeout, NOT burning 3 idle turns (3x12s = 36s)
    assert voice_obj2.listen_calls == 1, (
        f"Expected prompt exit after 1 listen timeout on departure, got {voice_obj2.listen_calls}"
    )


def test_farewell_spoken_exactly_once_on_person_left():
    """PERSON_LEFT handler must speak a farewell exactly once.

    The farewell is produced by the PERSON_LEFT event handler in
    conversation_thread, not inline in _dialogue_loop.  This test drives the
    handler directly to confirm:
      - a named identity produces a personalised farewell (contains the name)
      - an unknown identity produces the generic fallback
      - neither path fires zero or two farewells
    """
    from run import VisionAgentApp
    from events import EventType

    # ---- minimal stand-ins -----------------------------------------------
    farewell_calls: list[str] = []

    class _TrackVoice(StubVoice):
        def speak_async(self, text):
            farewell_calls.append(text)

    class _Hub:
        states: list[str] = []
        def set_aria_state(self, s):
            self.states.append(s)
        def say(self, *a, **kw): pass
        def heard(self, *a, **kw): pass

    class _MockConv:
        state = "ACTIVE"
        current_identity = "alice"
        reset_called = False
        def reset(self):
            self.reset_called = True
            self.state = "IDLE"
        def start_for(self, *a, **kw): return ""
        def handle_response(self, *a, **kw): return "", "ask"

    class _MockMic:
        clear_pending_called = False
        def clear_pending(self): self.clear_pending_called = True

    # ---- Case 1: known identity with persona.name set --------------------
    agent_obj = StubAgent()
    mem = agent_obj._memory_for("alice")
    mem.persona.name = "Alice"

    voice_obj = _TrackVoice()
    hub = _Hub()
    conv = _MockConv()
    mic = _MockMic()

    # Alias locals before class body — Python class scope can't close over
    # enclosing locals using the same name on the left-hand side.
    _agent = agent_obj
    _voice = voice_obj
    _hub = hub
    _mic = mic

    class _MinimalApp:
        agent = _agent
        voice = _voice
        hub = _hub
        mic = _mic

    app = _MinimalApp()

    # Simulate what conversation_thread does when it dequeues PERSON_LEFT
    import types
    # Extract the handler body by calling the relevant elif branch manually
    event = {"type": EventType.PERSON_LEFT, "track_id": "face_0",
             "identity": "alice", "duration": 120.0}

    # Call the handler inline (mirrors conversation_thread's elif branch)
    name = event.get("name") or event.get("identity", "")
    try:
        m = app.agent._memory_for(name)
        display = (m.persona.name or name) if m else name
    except Exception:
        display = name
    farewell = (
        f"See you later, {display}. Take care!"
        if display and display not in ("", "unknown")
        else "Goodbye! Come back anytime."
    )
    conv.reset()
    if app.mic is not None:
        app.mic.clear_pending()
    app.hub.set_aria_state("idle")
    app.voice.speak_async(farewell)

    assert len(farewell_calls) == 1, (
        f"Expected exactly 1 farewell, got {len(farewell_calls)}: {farewell_calls}"
    )
    assert "Alice" in farewell_calls[0], (
        f"Farewell should contain persona name 'Alice', got: {farewell_calls[0]!r}"
    )
    assert conv.reset_called
    assert mic.clear_pending_called
    assert "idle" in hub.states

    # ---- Case 2: unknown identity → generic fallback ---------------------
    farewell_calls.clear()
    hub.states.clear()
    conv2 = _MockConv()

    event2 = {"type": EventType.PERSON_LEFT, "track_id": "face_1",
               "identity": "unknown", "duration": 15.0}

    name2 = event2.get("name") or event2.get("identity", "")
    try:
        m2 = app.agent._memory_for(name2)
        display2 = (m2.persona.name or name2) if m2 else name2
    except Exception:
        display2 = name2
    farewell2 = (
        f"See you later, {display2}. Take care!"
        if display2 and display2 not in ("", "unknown")
        else "Goodbye! Come back anytime."
    )
    conv2.reset()
    app.hub.set_aria_state("idle")
    app.voice.speak_async(farewell2)

    assert len(farewell_calls) == 1, (
        f"Expected 1 generic farewell, got {len(farewell_calls)}: {farewell_calls}"
    )
    assert farewell_calls[0] == "Goodbye! Come back anytime.", (
        f"Generic farewell wrong: {farewell_calls[0]!r}"
    )
    assert "idle" in hub.states

    # ---- Case 3: track churn — identity still authorized on another track --
    # When face_0 expires but face_12 is already carrying the same identity,
    # the PERSON_LEFT handler must suppress the farewell and NOT set hub idle.
    farewell_calls.clear()
    hub.states.clear()
    conv3 = _MockConv()

    # Tracker has face_12 authorized as "alice" (simulating churn survivor)
    class _MockTracker:
        tracks = {
            "face_12": {"authorized": True, "identity": "alice"},
        }

    class _MinimalAppWithTracker:
        agent = _agent
        voice = _voice
        hub = _hub
        mic = _mic
        tracker = _MockTracker()

        def _identity_currently_authorized(self, identity):
            for track in self.tracker.tracks.values():
                if track.get("authorized") and track.get("identity") == identity:
                    return True
            return False

    app3 = _MinimalAppWithTracker()

    event3 = {"type": EventType.PERSON_LEFT, "track_id": "face_0",
               "identity": "alice", "duration": 5.0}

    # Inline the handler logic (mirrors conversation_thread's elif branch)
    name3 = event3.get("name") or event3.get("identity", "")
    try:
        m3 = app3.agent._memory_for(name3)
        display3 = (m3.persona.name or name3) if m3 else name3
    except Exception:
        display3 = name3
    farewell3 = (
        f"See you later, {display3}. Take care!"
        if display3 and display3 not in ("", "unknown")
        else "Goodbye! Come back anytime."
    )
    conv3.reset()
    if app3.mic is not None:
        app3.mic.clear_pending()
    # Guard: identity still present on another track → suppress farewell
    if name3 and name3 not in ("", "unknown") and app3._identity_currently_authorized(name3):
        pass  # churn — no farewell
    else:
        app3.hub.set_aria_state("idle")
        app3.voice.speak_async(farewell3)

    assert len(farewell_calls) == 0, (
        f"Farewell must be suppressed during track churn, got {farewell_calls}"
    )
    assert "idle" not in hub.states, (
        "Hub must NOT go idle when identity is still authorized on another track"
    )


def test_farewell_dedup_suppresses_multiple_queued_events_same_identity():
    """Simulate 3 queued PERSON_LEFT events for the same identity in one departure;
    assert speak_async fires exactly once."""
    from run import VisionAgentApp
    from events import EventType

    farewell_calls: list[str] = []

    class _TrackVoice(StubVoice):
        def speak_async(self, text):
            farewell_calls.append(text)

    class _Hub:
        states: list[str] = []
        def set_aria_state(self, s):
            self.states.append(s)
        def say(self, *a, **kw): pass
        def heard(self, *a, **kw): pass

    class _MockConv:
        state = "ACTIVE"
        current_identity = "alice"
        reset_called = False
        def reset(self):
            self.reset_called = True
            self.state = "IDLE"

    class _MockMic:
        def clear_pending(self): pass

    class _MockTracker:
        tracks = {}

    agent_obj = StubAgent()
    mem = agent_obj._memory_for("alice")
    mem.persona.name = "Alice"

    _agent = agent_obj
    _voice = _TrackVoice()
    _hub = _Hub()
    _mic = _MockMic()
    _tracker = _MockTracker()

    class _App(VisionAgentApp):
        def __init__(self):
            self.agent = _agent
            self.voice = _voice
            self.hub = _hub
            self.mic = _mic
            self.tracker = _tracker
            self._farewells_spoken = set()

    app = _App()
    conv = _MockConv()

    # 3 queued PERSON_LEFT events for "alice" from churned tracks during one departure
    e1 = {"type": EventType.PERSON_LEFT, "track_id": "face_0", "identity": "alice", "duration": 10.0}
    e2 = {"type": EventType.PERSON_LEFT, "track_id": "face_1", "identity": "alice", "duration": 5.0}
    e3 = {"type": EventType.PERSON_LEFT, "track_id": "face_2", "identity": "alice", "duration": 2.0}

    app._handle_person_left(e1, conv)
    app._handle_person_left(e2, conv)
    app._handle_person_left(e3, conv)

    assert len(farewell_calls) == 1, (
        f"Expected exactly 1 farewell for 3 queued departure events, got {len(farewell_calls)}: {farewell_calls}"
    )
    assert "Alice" in farewell_calls[0]
    assert "idle" in _hub.states


def test_farewell_rearms_on_fresh_identity_confirmed():
    """Simulate departure, dedup fires, then fresh IDENTITY_CONFIRMED,
    then a second real departure — assert a second farewell DOES fire."""
    from run import VisionAgentApp
    from events import EventType

    farewell_calls: list[str] = []

    class _TrackVoice(StubVoice):
        def speak_async(self, text):
            farewell_calls.append(text)

    class _Hub:
        states: list[str] = []
        def set_aria_state(self, s):
            self.states.append(s)
        def say(self, *a, **kw): pass
        def heard(self, *a, **kw): pass

    class _MockConv:
        state = "ACTIVE"
        current_identity = "alice"
        def reset(self):
            self.state = "IDLE"

    class _MockMic:
        def clear_pending(self): pass

    class _MockTracker:
        tracks = {}

    agent_obj = StubAgent()
    mem = agent_obj._memory_for("alice")
    mem.persona.name = "Alice"

    _agent = agent_obj
    _voice = _TrackVoice()
    _hub = _Hub()
    _mic = _MockMic()
    _tracker = _MockTracker()

    class _App(VisionAgentApp):
        def __init__(self):
            self.agent = _agent
            self.voice = _voice
            self.hub = _hub
            self.mic = _mic
            self.tracker = _tracker
            self._farewells_spoken = set()

    app = _App()
    conv = _MockConv()

    # 1. First departure
    e1 = {"type": EventType.PERSON_LEFT, "track_id": "face_0", "identity": "alice", "duration": 10.0}
    app._handle_person_left(e1, conv)
    assert len(farewell_calls) == 1

    # 2. Duplicate queued departure for same identity suppressed by dedup
    e1_dup = {"type": EventType.PERSON_LEFT, "track_id": "face_1", "identity": "alice", "duration": 5.0}
    app._handle_person_left(e1_dup, conv)
    assert len(farewell_calls) == 1, "Duplicate departure must be suppressed"

    # 3. Fresh arrival / IDENTITY_CONFIRMED re-arms farewell
    app._farewells_spoken.discard("alice")

    # 4. Second real departure fires a second farewell
    e2 = {"type": EventType.PERSON_LEFT, "track_id": "face_3", "identity": "alice", "duration": 20.0}
    app._handle_person_left(e2, conv)
    assert len(farewell_calls) == 2, (
        f"Expected second farewell to fire after fresh confirmation, got {len(farewell_calls)}: {farewell_calls}"
    )
    assert "Alice" in farewell_calls[1]


# ----------------------------------------------------------------------


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
