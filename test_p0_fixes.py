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
import os
from core.events import EventType

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
        from memory.context_memory import PersonMemory
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
    from interaction.conversation import ConversationManager
    from memory.context_memory import SessionManager

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
    from perception.tracker import FaceTracker

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
    from perception.tracker import FaceTracker

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
        # Phase-4 state machine: an authorized track IS in the AUTHORIZED
        # state — provisional-miss grace applies only from there.
        "state": "AUTHORIZED" if authorized else "UNKNOWN",
        "positives": 2 if authorized else 0,
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


def test_p1_near_miss_counts_as_miss_provisional():
    """Phase-4 contract: a near-miss on an authorized track keeps the identity
    PROVISIONALLY for this round, but counts toward revocation — the old
    near-miss-preserved-identity-forever behavior is gone."""
    tr, eng = _tracker_with_track(authorized=True)
    # 0.32 is below the 0.36 threshold — a miss, near or not
    eng.identify = lambda frame, face: ("alice", 0.32,
                                        {"threshold": 0.36, "authorized": False,
                                         "reason": "threshold_rejection"})
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    tr.recognize_faces(frame)
    # Provisional this round...
    assert tr.tracks["face_0"]["authorized"] is True
    assert tr.tracks["face_0"]["identity"] == "alice"
    # ...but the miss was counted
    assert tr._miss_streak["face_0"] == 1, "near-miss did not count toward revocation"

    tr.tracks["face_0"]["needs_recognition"] = True
    eng.identify = lambda frame, face: ("", 0.1,
                                        {"threshold": 0.36, "authorized": False,
                                         "reason": "threshold_rejection"})
    tr.recognize_faces(frame)
    # streak reached miss_limit=2 -> revoked
    assert tr.tracks["face_0"]["authorized"] is False
    assert tr.tracks["face_0"]["identity"] == "unknown"


def test_p1_engine_verdict_is_authoritative():
    """A high score with engine-authorized=False (e.g. ambiguous_match) must
    NOT authorize the track just because score >= threshold."""
    tr, eng = _tracker_with_track(authorized=False)
    eng.identify = lambda frame, face: ("", 0.9,
                                        {"threshold": 0.36, "authorized": False,
                                         "reason": "ambiguous_match", "score": 0.9})
    tr.recognize_faces(np.zeros((240, 320, 3), dtype=np.uint8))
    assert tr.tracks["face_0"]["authorized"] is False, (
        "track authorized despite the engine's explicit rejection"
    )
    assert tr.tracks["face_0"]["identity"] == "unknown"


def test_p1_rejected_identity_normalized_to_unknown():
    """Engine returns "" on rejection; downstream must see "unknown"."""
    tr, eng = _tracker_with_track(authorized=False)
    eng.identify = lambda frame, face: ("", 0.2,
                                        {"threshold": 0.36, "authorized": False})
    tr.recognize_faces(np.zeros((240, 320, 3), dtype=np.uint8))
    assert tr.tracks["face_0"]["identity"] == "unknown"


# ----------------------------------------------------------------------
# Phase 1 — face engine decision contract
# ----------------------------------------------------------------------

_ENGINE = None


def _shared_engine():
    """One real FaceEngine for all contract tests (model load is slow)."""
    global _ENGINE
    if _ENGINE is None:
        from perception.face_engine import FaceEngine
        _ENGINE = FaceEngine()
    return _ENGINE


def _probe_row(x=10.0, y=10.0, w=160.0, h=160.0, conf=0.9):
    """A 15-float YuNet-style face row with plausible landmarks."""
    cx, cy = x + w / 2, y + h / 2
    return np.array([x, y, w, h,
                     cx - w * 0.18, cy - h * 0.15,   # right eye
                     cx + w * 0.18, cy - h * 0.15,   # left eye
                     cx, cy + h * 0.02,              # nose
                     cx - w * 0.10, cy + h * 0.22,   # mouth r
                     cx + w * 0.10, cy + h * 0.22,   # mouth l
                     conf], dtype=np.float32)


def _mix(vec, other, cos_target):
    """Vector with cosine similarity = cos_target to vec (both normalized)."""
    ortho = other - np.dot(other, vec) * vec
    n = np.linalg.norm(ortho)
    assert n > 1e-6
    ortho = ortho / n
    sin_t = (1.0 - cos_target ** 2) ** 0.5
    out = cos_target * vec + sin_t * ortho
    return (out / np.linalg.norm(out)).astype(np.float32)


def test_p1_identify_meta_contract_and_margin_pass():
    eng = _shared_engine()
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    row = _probe_row()
    probe = eng._extract_aligned(frame, row)
    assert probe is not None, "probe embedding failed — landmarks/model issue"

    rng = np.random.default_rng(7)
    other = rng.normal(size=probe.shape).astype(np.float32)
    other /= np.linalg.norm(other)
    eng.known_faces = {"alice": [probe],
                       "bob": [_mix(probe, other, 0.50)]}
    try:
        identity, score, meta = eng.identify(frame, row)
        assert identity == "alice"
        assert meta["authorized"] is True and meta["reason"] == "match"
        for key in ("quality", "threshold", "score", "second_best", "margin",
                    "margin_required", "template_count", "authorized", "reason"):
            assert key in meta, f"meta missing contract key {key}"
        assert meta["margin"] >= 0.05, "clear match must clear MARGIN_MIN"
        assert abs(meta["second_best"] - 0.50) < 0.02, (
            f"second_best should reflect bob's constructed similarity, got {meta['second_best']}"
        )
    finally:
        eng.known_faces = {}


def test_p1_identify_rejects_ambiguous_match():
    eng = _shared_engine()
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    row = _probe_row()
    probe = eng._extract_aligned(frame, row)
    rng = np.random.default_rng(11)
    other = rng.normal(size=probe.shape).astype(np.float32)
    other /= np.linalg.norm(other)
    # bob sits 0.98 from the probe — inside MARGIN_MIN (0.05) of alice's 1.0
    eng.known_faces = {"alice": [probe],
                       "bob": [_mix(probe, other, 0.98)]}
    try:
        identity, score, meta = eng.identify(frame, row)
        assert identity == "", "ambiguous match must not return an identity"
        assert meta["authorized"] is False
        assert meta["reason"] == "ambiguous_match"
        assert meta["score"] >= meta["threshold"], "score cleared the bar — margin must be why"
        assert meta["margin"] < meta["margin_required"]
    finally:
        eng.known_faces = {}


def test_p1_identify_low_quality_rejects_with_threshold_intact():
    eng = _shared_engine()
    # Tiny face on a big frame + flat (blurred) background -> quality < 0.30
    frame = np.full((720, 960, 3), 128, dtype=np.uint8)
    row = _probe_row(x=100.0, y=100.0, w=40.0, h=40.0)
    rng = np.random.default_rng(3)
    tmpl = rng.normal(size=128).astype(np.float32)
    tmpl /= np.linalg.norm(tmpl)
    eng.known_faces = {"alice": [tmpl]}
    try:
        identity, score, meta = eng.identify(frame, row)
        assert identity == ""
        assert meta["authorized"] is False
        assert meta["reason"] == "low_quality"
        # THE contract: the bar was never lowered
        assert meta["threshold"] == eng.MATCH_THRESHOLD
    finally:
        eng.known_faces = {}


def test_p1_identify_threshold_and_edge_cases():
    eng = _shared_engine()
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    row = _probe_row()

    # No identities enrolled
    eng.known_faces = {}
    identity, score, meta = eng.identify(frame, row)
    assert identity == "" and meta["reason"] == "no_identities_enrolled"

    # Single identity, orthogonal template -> below threshold
    rng = np.random.default_rng(5)
    tmpl = rng.normal(size=128).astype(np.float32)
    tmpl /= np.linalg.norm(tmpl)
    eng.known_faces = {"alice": [tmpl]}
    try:
        identity, score, meta = eng.identify(frame, row)
        assert identity == "" and meta["reason"] == "threshold_rejection"
        assert meta["score"] < meta["threshold"]
        # <2 identities: margin must not gate
        assert meta["margin_required"] == 0.0
        assert meta["second_best"] is None and meta["margin"] is None

        # Alignment failure: box-only row (no landmarks)
        identity, score, meta = eng.identify(frame, np.array([10, 10, 60, 60], dtype=np.float32))
        assert identity == "" and meta["reason"] == "alignment_failed"
        assert meta["authorized"] is False
    finally:
        eng.known_faces = {}


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
    from perception.presence import PresenceManager

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
    assert EventType.IDENTITY_CONFIRMED not in events, (
        "identity-level cooldown failed: re-entering known person re-greeted"
    )


def test_b9_identity_cooldown_expires():
    from perception.presence import PresenceManager

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
    assert EventType.IDENTITY_CONFIRMED in events, "cooldown never expired for known person"


def test_presence_unknown_reentry_still_greeted():
    """Unknown visitors must still trigger a conversation on re-entry."""
    from perception.presence import PresenceManager

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
    from interaction.voice import SarvamVoice

    v = SarvamVoice(api_key=None)  # forces the print-TTS path, no network
    gen_before = v._tts_gen
    v.interrupt()
    assert v._tts_gen == gen_before + 1, "interrupt() must bump the generation counter"
    assert v._interrupt_flag.is_set()


def test_voice_speak_captures_generation_after_interrupt():
    from interaction.voice import SarvamVoice

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
# B10 — downscaled landmarks must be rescaled before recognition
# ----------------------------------------------------------------------

def test_b10_tracker_rescales_landmarks_to_full_resolution():
    """Landmarks from the downscaled detection frame must be scaled back up.

    Regression: tracker.update() rescaled only the bbox, leaving landmarks in
    downscaled coords. SFace then aligned on garbage landmark positions and
    enrolled faces scored ~0.28 against their own templates (threshold 0.36)
    — authorized faces were never recognized live.
    """
    from perception.tracker import FaceTracker

    LM = [30.0, 40.0, 50.0, 40.0, 40.0, 50.0, 30.0, 60.0, 50.0, 60.0]

    class FixedEngine(FakeFaceEngine):
        def detect(self, frame):
            # Same coords regardless of input scale — the tracker must
            # undo its own downscaling.
            return np.array([[20.0, 20.0, 60.0, 60.0] + LM])

    tr = FaceTracker(FixedEngine())
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    tr.update(frame, downscale_factor=0.5)

    track = next(iter(tr.tracks.values()))
    got = track["landmarks"]
    expected = [int(v * 2) for v in LM]
    assert got == expected, (
        f"landmarks not rescaled to full resolution: {got} != {expected}"
    )
    # bbox must be rescaled too (regression guard for both halves of the fix)
    assert track["bbox"] == (40, 40, 120, 120)


def test_b10_recognize_faces_pads_missing_landmarks():
    """A track with no landmarks must not produce a 4-element face array.

    Regression: [x,y,w,h] + [] built a 4-element row, and SFace's alignCrop
    silently mis-cropped it, producing a non-face embedding.
    """
    from perception.tracker import FaceTracker

    seen = {}

    class SpyEngine(FakeFaceEngine):
        def identify(self, frame, face):
            seen["face"] = np.asarray(face, dtype=np.float32).copy()
            return "alice", 0.5, {"threshold": 0.36, "quality": 0.8}

    tr = FaceTracker(SpyEngine())
    tr.tracks["face_0"] = {
        "bbox": (10, 10, 60, 60),
        "last_seen": time.time(),
        "last_recognize": time.time(),
        "skip_count": 0,
        "identity": "unknown",
        "score": 0.0,
        "meta": {},
        "landmarks": [],
        "needs_recognition": True,
    }
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    tr.recognize_faces(frame)
    assert seen["face"].size == 14, (
        f"face array has {seen["face"].size} elements — landmarks not padded"
    )
    # Phase-4 temporal confirmation: one positive arms VERIFYING; a second
    # strong positive AUTHORIZES. Identity follows current evidence.
    assert tr.tracks["face_0"]["identity"] == "alice"
    assert tr.tracks["face_0"]["state"] == "VERIFYING"
    tr.tracks["face_0"]["needs_recognition"] = True
    tr.recognize_faces(frame)
    assert tr.tracks["face_0"]["authorized"] is True
    assert tr.tracks["face_0"]["state"] == "AUTHORIZED"


def test_b10_engine_rejects_garbage_landmark_rows():
    """FaceEngine._extract_aligned must refuse to embed non-face alignments."""
    import os

    if not (os.path.isfile("models/face_detection_yunet_2023mar.onnx")
            and os.path.isfile("models/face_recognition_sface_2021dec.onnx")):
        print("      (models not present — skipping)")
        return

    from perception.face_engine import FaceEngine

    fe = FaceEngine()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    # Box-only row (the old 4-element rebuild) must be rejected
    assert fe._extract_aligned(frame, np.array([10, 10, 60, 60], dtype=np.float32)) is None
    # Zero landmarks must be rejected
    assert fe._extract_aligned(
        frame, np.array([10, 10, 60, 60] + [0] * 10, dtype=np.float32)) is None


def test_b10_quality_uses_detection_score_not_eye_x():
    """_face_quality must read YuNet confidence from index 14, not 4.

    face[4] is the right-eye X coordinate (an int, often > 1.0), which made
    conf_score clamp to 1.0 for every face and quality worthless.
    """
    from perception.face_engine import FaceEngine

    fe = FaceEngine()
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    # Landmarks with a huge eye-x coordinate, confidence at index 14 = 0.6
    face = np.array([10, 10, 60, 60, 300, 40, 320, 40, 310, 55,
                     295, 65, 315, 65, 0.6], dtype=np.float32)
    q = fe._face_quality(face, frame)
    # 60x60 on 240x320 = 4.7% of frame -> size_score = 1.0 (0.4 pts).
    # conf 0.6 -> 0.3*0.6 = 0.18 pts. Flat frame -> blur 0 (0.3 pts lost).
    # Correct total: ~0.58. The old bug read eye-x=300 as confidence -> 1.0,
    # giving 0.7. Assert the conf term reflects face[14], not face[4].
    assert q < 0.7, f"quality saturated ({q}) — still reading face[4] as confidence"
    assert q >= 0.55, f"quality too low ({q}) — confidence index change broke scoring"


# ----------------------------------------------------------------------
# B11 — one face must never produce two overlapping tracks (duplicate mesh)
# ----------------------------------------------------------------------

def _scripted_engine(positions):
    """Engine stub whose detect() yields a scripted list of face rows.

    Each entry is a list of (x, y, w, h) boxes for one frame; [] = no faces.
    """
    class ScriptedEngine(FakeFaceEngine):
        def __init__(self):
            super().__init__()
            self.frames = list(positions)

        def detect(self, frame):
            boxes = self.frames.pop(0) if self.frames else []
            rows = []
            for (x, y, w, h) in boxes:
                cx, cy = x + w // 2, y + h // 2
                rows.append([float(x), float(y), float(w), float(h),
                             float(cx - 10), float(cy - 10), float(cx + 10), float(cy - 10),
                             float(cx), float(cy), float(cx - 8), float(cy + 12),
                             float(cx + 8), float(cy + 12), 0.9])
            return np.array(rows) if rows else np.empty((0, 0))

    return ScriptedEngine()


def test_b11_ghost_track_absorbed_no_duplicate_pattern():
    """A stale track overlapping a fresh track of the SAME face must be
    merged, not kept alongside it.

    Regression: when detection flickered (blur) and a large close-up face
    moved further than dist_threshold, the old track was kept as a ghost AND
    a new track spawned — the landmark mesh drew twice for one person.
    """
    from perception.tracker import FaceTracker

    # 300x300 face (typical close-up webcam box), moved 120px between
    # frames: centroid dist 120 > dist_threshold (100) so centroid matching
    # misses, but IoU = 0.43 > 0.30 so the boxes are obviously the same face.
    script = [
        [(10, 10, 300, 300)],   # frame 1: track created
        [],                      # frame 2: detection flickers -> ghost kept
        [(130, 10, 300, 300)],  # frame 3: same face re-detected, moved
    ]
    tr = FaceTracker(_scripted_engine(script), dist_threshold=100.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    tr.update(frame, downscale_factor=1.0)
    tr.recognize_faces(frame)          # face_0 recognized as unknown
    tr.update(frame, downscale_factor=1.0)   # flicker
    results = tr.update(frame, downscale_factor=1.0)  # re-detection after move

    assert len(tr.tracks) == 1, (
        f"duplicate tracks for one face: {list(tr.tracks)}"
    )
    assert len(results) == 1, "results draw more than one pattern for one face"
    tid = next(iter(tr.tracks))
    assert tid == "face_0", "older track id must survive for state continuity"
    assert tr.tracks[tid]["bbox"] == (130, 10, 300, 300), (
        "merged track must take the fresh geometry"
    )
    assert tr.tracks[tid]["needs_recognition"] is True, (
        "merged track must re-identify (the new geometry was never recognized)"
    )


def test_b11_ghost_dropped_when_twin_is_matched_track():
    """A ghost overlapping a track that was matched this frame is dropped."""
    from perception.tracker import FaceTracker

    script = [
        [(10, 10, 300, 300)],  # face_0 created
    ]
    tr = FaceTracker(_scripted_engine(script))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tr.update(frame, downscale_factor=1.0)
    tr.recognize_faces(frame)

    # Inject a ghost (e.g. left over from a previous flicker) on the same face
    tr.tracks["face_9"] = {
        "bbox": (40, 20, 300, 300),
        "last_seen": time.time(),
        "last_recognize": time.time(),
        "skip_count": 0,
        "identity": "unknown",
        "score": 0.0,
        "meta": {},
        "landmarks": [],
        "needs_recognition": False,
    }

    script2 = [[(12, 12, 300, 300)]]  # face_0 matched again this frame
    tr.face_engine.frames = script2
    tr.update(frame, downscale_factor=1.0)

    assert "face_9" not in tr.tracks, "ghost overlapping a matched track survived"
    assert "face_0" in tr.tracks


def test_b12_anon_persona_roundtrip():
    """Phase 1: unknown identity personas are session-scoped and non-persistent.
    save(identity="unknown") does NOT write anon_<date>.json; load("unknown")
    always returns a fresh unpersisted instance.
    """
    import memory.context_memory as context_memory

    today = context_memory.datetime.datetime.now().strftime("%Y%m%d")
    anon_path = os.path.join(context_memory.PERSONA_DIR, f"anon_{today}.json")
    backup = None
    if os.path.exists(anon_path):
        with open(anon_path) as f:
            backup = f.read()
        os.remove(anon_path)
    try:
        pg = context_memory.PersonaGraph("unknown")
        pg.traits = ["wearing glasses", "mentioned printer repair"]
        pg.purpose = "fixing the coffee machine"
        saved = pg.save()
        assert saved is False, "save(unknown) should refuse to persist"
        assert not os.path.exists(anon_path), "save(unknown) should NOT write anon_<date>.json"

        loaded = context_memory.PersonaGraph.load("unknown")
        assert loaded.traits == [], "unknown persona should always load fresh, not from disk"
        assert loaded.purpose == ""
    finally:
        if backup is not None:
            with open(anon_path, "w") as f:
                f.write(backup)
        elif os.path.exists(anon_path):
            os.remove(anon_path)


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
