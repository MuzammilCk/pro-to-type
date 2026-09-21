"""Tests for the MediaPipe face-pattern renderer (ARIA_FACE_PATTERN_UPGRADE.md).

Run:  python test_face_pattern.py   (or pytest test_face_pattern.py -q)

Covers:
  Step 4/5 — landmark extraction on a real face + pure rendering function
  Step 6   — ROI-scoped, throttled analysis wired against tracker-shaped input
  Step 7   — BOTH states explicitly:
             - pattern appears and persists on an unrecognized face
             - pattern disappears the moment the face is recognized

Face boxes come from the REAL YuNet detector (same as production), so the
ROIs fed to the landmarker match what run.py's tracker produces.
"""

import numpy as np
import cv2

from perception.face_pattern import FacePatternEngine


def yunet_face_bbox(frame) -> tuple[int, int, int, int] | None:
    """Largest face bbox in the frame via the production YuNet detector."""
    det = cv2.FaceDetectorYN_create(
        "models/face_detection_yunet_2023mar.onnx", "", (frame.shape[1], frame.shape[0]),
        score_threshold=0.5, nms_threshold=0.3, top_k=5000)
    faces = det.detect(frame)
    if faces[1] is None:
        return None
    x, y, w, h = [int(v) for v in max(faces[1], key=lambda f: f[2] * f[3])[:4]]
    return (x, y, w, h)


def load_frame_with_face():
    frame = cv2.imread("samples/face_test.jpg")
    assert frame is not None, "samples/face_test.jpg missing"
    bbox = yunet_face_bbox(frame)
    assert bbox is not None, "YuNet found no face in the sample image"
    return frame, bbox


def make_face_result(track_id="face_0", authorized=False,
                     bbox=(160, 130, 190, 190)):
    """Tracker-output-shaped dict (same keys tracker._build_results emits)."""
    return {
        "track_id": track_id,
        "identity": "unknown" if not authorized else "alice",
        "authorized": authorized,
        "distance": 0.0 if not authorized else 0.6,
        "face_bbox": bbox,
        "landmarks": [],
        "quality": 0.8,
        "needs_recognition": False,
    }


def test_real_face_landmarks_extracted():
    """FaceLandmarker returns a dense mesh for the real face sample."""
    frame, bbox = load_frame_with_face()
    eng = FacePatternEngine()
    assert eng.available, "FacePatternEngine unavailable — check models/face_landmarker.task"

    fr = make_face_result(bbox=bbox)
    eng.analyze(frame, [fr], frame_idx=0)
    cache = eng._cache.get("face_0")
    assert cache is not None and cache["points"] is not None, (
        "no landmarks extracted for the sample face"
    )
    pts = cache["points"]
    assert pts.ndim == 2 and pts.shape[1] == 2
    assert len(pts) >= 468, f"expected >=468 landmarks, got {len(pts)}"
    # Absolute coordinates must land inside the frame
    h, w = frame.shape[:2]
    assert pts[:, 0].min() >= 0 and pts[:, 0].max() < w
    assert pts[:, 1].min() >= 0 and pts[:, 1].max() < h


def test_pattern_draws_on_unrecognized():
    """Pattern pixels must appear on an unrecognized face's frame."""
    frame, bbox = load_frame_with_face()
    eng = FacePatternEngine()

    eng.analyze(frame, [make_face_result(bbox=bbox)], frame_idx=0)
    canvas = frame.copy()
    drawn = eng.draw(canvas, "face_0")
    assert drawn, "draw() reported nothing drawn for unrecognized face"
    assert not np.array_equal(canvas, frame), "canvas unchanged — nothing rendered"


def test_pattern_persists_across_frames():
    """Unregistered face keeps its pattern on subsequent frames (throttled reuse)."""
    frame, bbox = load_frame_with_face()
    eng = FacePatternEngine()

    eng.analyze(frame, [make_face_result(bbox=bbox)], frame_idx=0)
    assert eng.draw(frame.copy(), "face_0")
    # Next frame, throttled — cache must still serve the pattern
    eng.analyze(frame, [make_face_result(bbox=bbox)], frame_idx=1)
    assert eng.draw(frame.copy(), "face_0"), "pattern did not persist to next frame"


def test_pattern_hides_when_recognized():
    """Step 7 before/after: register/recognize the SAME face -> pattern gone."""
    frame, bbox = load_frame_with_face()
    eng = FacePatternEngine()

    eng.analyze(frame, [make_face_result(authorized=False, bbox=bbox)], frame_idx=0)
    assert eng.draw(frame.copy(), "face_0"), "precondition: unrecognized face draws"

    # Presence flips the status (e.g. right after enrollment/registration)
    eng.analyze(frame, [make_face_result(authorized=True, bbox=bbox)], frame_idx=1)
    assert "face_0" not in eng._cache or eng._cache["face_0"]["points"] is None, (
        "cache survived recognition"
    )
    assert not eng.draw(frame.copy(), "face_0"), (
        "pattern still drawn after the face became recognized"
    )


def test_prune_drops_gone_tracks():
    eng = FacePatternEngine()
    eng._cache["face_0"] = {"points": np.zeros((478, 2), np.float32), "frame": 0}
    eng._cache["face_9"] = {"points": np.zeros((478, 2), np.float32), "frame": 0}
    eng.prune({"face_0"})
    assert "face_0" in eng._cache and "face_9" not in eng._cache


def test_small_face_roi_skipped_safely():
    """A far-away face below MIN_ROI must not crash or fabricate landmarks."""
    eng = FacePatternEngine()
    frame = np.zeros((240, 320, 3), np.uint8)
    eng.analyze(frame, [make_face_result(track_id="tiny", bbox=(10, 10, 20, 20))], 0)
    assert not eng.draw(frame.copy(), "tiny")


def test_full_pipeline_render_unrecognized_vs_recognized():
    """End-to-end visual diff: same frame, two statuses, different canvases."""
    frame, bbox = load_frame_with_face()

    eng = FacePatternEngine()
    eng.analyze(frame, [make_face_result(authorized=False, bbox=bbox)], frame_idx=0)
    unrec = frame.copy()
    assert eng.draw(unrec, "face_0")

    eng2 = FacePatternEngine()
    eng2.analyze(frame, [make_face_result(authorized=True, bbox=bbox)], frame_idx=0)
    rec = frame.copy()
    assert not eng2.draw(rec, "face_0")

    diff = cv2.absdiff(unrec, rec)
    changed = int((diff.sum(axis=2) > 0).sum())
    assert changed > 500, (
        f"only {changed} px differ between states — pattern is not visibly rendering"
    )
    cv2.imwrite("samples/output_pattern_unrecognized.jpg", unrec)
    cv2.imwrite("samples/output_pattern_recognized.jpg", rec)


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
