"""Regression tests for the P0 bug-cluster fixes.

Run:  python test_p0_fixes.py   (or pytest test_p0_fixes.py -q)

Covers:
  B1 — greeting is built once and spoken once (ConversationManager no longer speaks)
  B3 — enrollment name extraction never takes filler words as identity
  B4/B8 — tracker result rebuild without a second detect pass
  B9 — tracker authorization hysteresis + presence identity-level greeting cooldown
  voice — generation counter: interrupt() invalidates in-flight speakers
"""

import threading
import time
import queue as queue_mod

import numpy as np


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------

class FakeVoice:
    """Records speak() calls; listen() returns scripted utterances."""

    def __init__(self, replies=None):
        self.spoken: list[str] = []
        self.replies = list(replies or [])
        self.interrupt_count = 0

    def speak(self, text, interrupt=True):
        self.spoken.append(text)

    def speak_async(self, text):
        self.spoken.append(text)
        return threading.Thread(target=lambda: None, daemon=True)

    def listen(self, timeout=10, phrase_limit=8):
        return self.replies.pop(0) if self.replies else ""

    def interrupt(self):
        self.interrupt_count += 1


class FakeAgent:
    """Minimal stand-in for VisionAgent."""

    def __init__(self):
        from context_memory import PersonMemory
        self.memory = {}

    def _memory_for(self, identity):
        if identity not in self.memory:
            self.memory[identity] = type("M", (), {})()
            self.memory[identity].persona = type("P", (), {})()
            self.memory[identity].persona.name = None
            self.memory[identity].working_context = []
            self.memory[identity].interaction_count = 0
        return self.memory[identity]

    def stream_response(self, identity, face_data, user_input, frame=None):
        yield "Hello there! What brings you here?", "ask"

    def think(self, identity, face_data, user_input=None):
        return f"Welcome back, {identity}!", "recognized"


class FakeFaceEngine:
    """Face engine stub with just what tracker/conversation touch."""

    MATCH_THRESHOLD = 0.36

    def __init__(self):
        self.known_faces = {}

    def detect(self, frame):
        return np.empty((0, 0))

    def identify(self, frame, face):
        return "unknown", 0.0, {"quality": 0.5, "threshold": self.MATCH_THRESHOLD}


def make_face_result(track_id="face_0", identity="alice", authorized=True,
                     score=0.6, bbox=(100, 100, 80, 80)):
    return {
        "track_id": track_id,
        "identity": identity,
        "authorized": authorized,
        "distance": score,
        "face_bbox": bbox,
        "landmarks": [],
        "quality": 0.8,
        "needs_recognition": False,
    }


# ----------------------------------------------------------------------
# B1 — single greeting
# ----------------------------------------------------------------------

def test_b1_greeting_spoken_exactly_once():
    from conversation import ConversationManager
    from context_memory import SessionManager

    # Keep the checkpoint hermetic — no persona/ writes during the test
    SessionManager.save_current_state = staticmethod(lambda identities: None)

    voice = FakeVoice()
    conv = ConversationManager(FakeAgent(), voice, FakeFaceEngine())

    greeting = conv.start_for("unknown", {"authorized": False, "score": 0.5})
    conv.handle_response("I'm here to fix the printer", "unknown",
                         {"authorized": False, "score": 0.5})

    assert greeting.strip(), "start_for must still return the greeting text"
    assert voice.spoken == [], (
        f"B1 regression: ConversationManager spoke {len(voice.spoken)} time(s); "
        "playback must belong to the caller only"
    )


# ----------------------------------------------------------------------
# B3 — enrollment name extraction
# ----------------------------------------------------------------------

def test_b3_name_extraction():
    from run import VisionAgentApp

    cases = {
        "my name is Praveen": "Praveen",
        "My Name Is  Sara .": "Sara",
        "i'm Dan": "Dan",
        "Im Ravi": "Ravi",
        "I am called Jean": "Jean",
        "call me Bob": "Bob",
        "hey": None,           # the old code enrolled "Hey" here
        "hi there": None,
        "my name is": None,
        "yes please": None,    # consent without a name
        "sure": None,          # lone filler word
        "Alice": "Alice",
        "": None,
    }
    for utterance, expected in cases.items():
        got = VisionAgentApp._extract_name(utterance)
        assert got == expected, f"_extract_name({utterance!r}) = {got!r}, expected {expected!r}"


# ----------------------------------------------------------------------
# B8/B4 — tracker rebuild + pending-recognition carry-over
# ----------------------------------------------------------------------

def test_b8_update_results_no_second_detection():
    from tracker import FaceTracker

    calls = {"detect": 0}

    class CountingEngine(FakeFaceEngine):
        def detect(self, frame):
            calls["detect"] += 1
            return np.array([[10.0, 10.0, 60.0, 60.0, 0.9] + [0.0] * 10])

    eng = CountingEngine()
    tr = FaceTracker(eng)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    tr.update(frame)
    n_after_update = calls["detect"]

    results = tr.update_results()  # must NOT re-run detection
    assert calls["detect"] == n_after_update, "update_results() re-ran detection"
    assert len(results) == 1
    assert results[0]["track_id"].startswith("face_")

    # Pending recognition must survive the rebuild (old double-update wiped it)
    tr.tracks[results[0]["track_id"]]["needs_recognition"] = True
    tr.update(frame)
    assert tr.tracks[results[0]["track_id"]]["needs_recognition"] is True, (
        "pending recognition lost across update()"
    )


# ----------------------------------------------------------------------
# B9a — tracker authorization hysteresis
# ----------------------------------------------------------------------

def _tracker_with_track(authorized=True):
    from tracker import FaceTracker

    eng = FakeFaceEngine()
    tr = FaceTracker(eng, miss_limit=2, hysteresis_margin=0.08)
    tr.tracks["face_0"] = {
        "bbox": (10, 10, 60, 60),
        "last_seen": time.time(),
        "last_recognize": time.time(),
        "skip_count": 0,
        "identity": "alice" if authorized else "unknown",
        "score": 0.6 if authorized else 0.1,
        "meta": {},
        "landmarks": [],
        "needs_recognition": True,
        "authorized": authorized,
    }
    return tr, eng


def test_b9_single_bad_frame_keeps_authorization():
    tr, eng = _tracker_with_track(authorized=True)
    # One clear miss (score 0.1 << threshold) — must stay authorized
    eng.identify = lambda frame, face: ("unknown", 0.1, {"threshold": 0.36})
    tr.recognize_faces(np.zeros((240, 320, 3), dtype=np.uint8))
    assert tr.tracks["face_0"]["authorized"] is True, "single miss revoked authorization"
    assert tr._miss_streak["face_0"] == 1


def test_b9_miss_streak_revokes_authorization():
    tr, eng = _tracker_with_track(authorized=True)
    eng.identify = lambda frame, face: ("unknown", 0.1, {"threshold": 0.36})
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    tr.recognize_faces(frame)   # miss 1 -> sticky
    tr.tracks["face_0"]["needs_recognition"] = True
    tr.recognize_faces(frame)   # miss 2 -> streak reached, revoke
    assert tr.tracks["face_0"]["authorized"] is False, "streak of misses failed to revoke"
    assert tr.tracks["face_0"]["identity"] == "unknown"


def test_b9_near_miss_keeps_identity():
    tr, eng = _tracker_with_track(authorized=True)
    # 0.32 is below the 0.36 threshold but within the 0.08 margin
    eng.identify = lambda frame, face: ("alice", 0.32, {"threshold": 0.36})
    tr.recognize_faces(np.zeros((240, 320, 3), dtype=np.uint8))
    assert tr.tracks["face_0"]["authorized"] is True
    assert tr.tracks["face_0"]["identity"] == "alice"


def test_b9_rematch_clears_streak():
    tr, eng = _tracker_with_track(authorized=True)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    eng.identify = lambda f, face: ("unknown", 0.1, {"threshold": 0.36})
    tr.recognize_faces(frame)                       # miss 1
    tr.tracks["face_0"]["needs_recognition"] = True
    eng.identify = lambda f, face: ("alice", 0.55, {"threshold": 0.36})
    tr.recognize_faces(frame)                       # rematch
    assert tr.tracks["face_0"]["authorized"] is True
    assert "face_0" not in tr._miss_streak, "streak not cleared after rematch"


# ----------------------------------------------------------------------
# B9b — presence identity-level greeting cooldown
# ----------------------------------------------------------------------

def test_b9_identity_cooldown_suppresses_reentry_greeting():
    from presence import PresenceManager

    q = queue_mod.Queue()
    pm = PresenceManager(q)

    pm.update([make_face_result(track_id="face_A", identity="alice")])
    pm.mark_greeted("face_A", now=time.time())

    # Drain the FIRST greeting's events — we only care about what re-entry emits
    while not q.empty():
        q.get_nowait()

    # Same person walks out and back in -> NEW track_id, same identity
    pm.update([])  # face_A goes away (left event after 2s; force with large now)
    pm.update([], now=time.time() + 10)  # flush left/purge
    pm.tracks.clear()  # simulate full track-table reset (new session of the person)

    pm.update([make_face_result(track_id="face_B", identity="alice")])

    events = []
    while not q.empty():
        events.append(q.get_nowait()["type"])
    assert "person_recognized" not in events, (
        "identity-level cooldown failed: re-entering known person re-greeted"
    )


def test_b9_identity_cooldown_expires():
    from presence import PresenceManager

    q = queue_mod.Queue()
    pm = PresenceManager(q)

    pm.update([make_face_result(track_id="face_A", identity="alice")])
    pm.mark_greeted("face_A", now=time.time())

    while not q.empty():  # drain first-greeting events
        q.get_nowait()

    pm.tracks.clear()
    pm._identity_greeted["alice"] = time.time() - pm.GREETING_COOLDOWN - 1

    pm.update([make_face_result(track_id="face_B", identity="alice")])

    events = []
    while not q.empty():
        events.append(q.get_nowait()["type"])
    assert "person_recognized" in events, "cooldown never expired for known person"


def test_presence_unknown_reentry_still_greeted():
    """Unknown visitors must still trigger a conversation on re-entry."""
    from presence import PresenceManager

    q = queue_mod.Queue()
    pm = PresenceManager(q)

    pm.update([make_face_result(track_id="face_A", identity="unknown", authorized=False)])
    pm.mark_greeted("face_A", now=time.time())
    pm.tracks.clear()
    pm.update([make_face_result(track_id="face_B", identity="unknown", authorized=False)])

    events = []
    while not q.empty():
        events.append(q.get_nowait()["type"])
    assert "person_unrecognized" in events, "unknown re-entry no longer triggers greeting"


# ----------------------------------------------------------------------
# voice — generation counter makes barge-in race-safe
# ----------------------------------------------------------------------

def test_voice_interrupt_bumps_generation():
    from voice import SarvamVoice

    v = SarvamVoice(api_key=None)  # forces the print-TTS path, no network
    gen_before = v._tts_gen
    v.interrupt()
    assert v._tts_gen == gen_before + 1, "interrupt() must bump the generation counter"
    assert v._interrupt_flag.is_set()


def test_voice_speak_captures_generation_after_interrupt():
    from voice import SarvamVoice

    v = SarvamVoice(api_key=None)
    captured = {}

    def fake_sarvam(text, my_gen=None, voice="meera", speed=1.0):
        captured["gen"] = my_gen

    v._speak_sarvam = fake_sarvam
    v.speak("hello")            # interrupt() inside speak() bumps gen, then captures
    gen_after_speak = v._tts_gen
    assert captured["gen"] == gen_after_speak, (
        "speak() must capture the generation AFTER its own interrupt bump, "
        "or a listen() race can kill the new utterance"
    )
    # A later interrupt invalidates the captured speaker
    v.interrupt()
    assert captured["gen"] != v._tts_gen


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
