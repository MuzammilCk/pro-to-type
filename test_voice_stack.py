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
