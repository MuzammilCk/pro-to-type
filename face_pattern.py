"""Facial landmark pattern renderer — MediaPipe Face Landmarker (Tasks API).

Per ARIA_FACE_PATTERN_UPGRADE.md this is a NEW component alongside the
existing identity pipeline, not a replacement:

    Identity pipeline (unchanged):  YuNet detect -> SFace recognize -> Presence
    New: pattern renderer:          FaceLandmarker -> 468-pt mesh -> draw

The renderer reads each tracked face's existing recognition status to decide
visibility:
    unrecognized -> draw the landmark pattern (contours or tesselation style)
    recognized   -> draw nothing

No new "authorization" concept is introduced — the pattern simply becomes
color/state-aware of the status the Presence Manager already computes.

Performance (CPU-only hardware):
- Runs only on each tracked face's ROI (from the tracker), never the full frame
- Re-infers only every Nth frame per face (`ARIA_PATTERN_EVERY`, default 2);
  in between it reuses the last known landmark positions — imperceptible at
  webcam rates, roughly half the landmarker cost
- Purely visual: nothing here feeds back into recognition logic

Threading contract: analyze() and draw() are called from the vision thread
only (run.py's _vision_loop), so the cache needs no locks.

MediaPipe version notes (verified on mediapipe 1.0.1 / Python 3.14):
- Built against the Tasks API (FaceLandmarker, VIDEO running mode); the legacy
  `mp.solutions.face_mesh` API is NOT present in 1.0.1 and must not be used.
- Connection topology comes from vision.FaceLandmarksConnections; if a future
  install drops it, we fall back to an embedded face-oval ring so the jawline
  still renders.
"""
import os
import time

import cv2
import numpy as np

PATTERN_MODEL_PATH = "models/face_landmarker.task"

# Env-tunable config (defaults chosen for the demo look described in the doc)
STYLE = os.getenv("ARIA_PATTERN_STYLE", "contours")   # "contours" | "tesselation"
EVERY_N_FRAMES = int(os.getenv("ARIA_PATTERN_EVERY", "2"))
MAX_FACES = int(os.getenv("ARIA_PATTERN_MAX_FACES", "4"))

# Unrecognized-visitor scan color (BGR): bright cyan with a darker glow pass
COLOR_MAIN = (255, 255, 0)
COLOR_GLOW = (110, 110, 0)

# Face-oval ring (FACEMESH_FACE_OVAL) — embedded fallback topology, used only
# if vision.FaceLandmarksConnections is unavailable in the installed mediapipe.
_FACE_OVAL_FALLBACK = frozenset([
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
])

# ROI padding around the tracker bbox — headroom for jaw/hairline movement
# between the tracker's detection and the landmarker run.
ROI_PAD = 0.30
MIN_ROI = 48  # px; smaller ROIs are below landmarker's useful resolution


def _edges(conn_set) -> list[tuple[int, int]]:
    """Normalize a connection set to (start, end) index pairs.

    Tasks API entries expose .start/.end; tolerate plain tuples as well.
    """
    out = []
    for e in conn_set:
        a = getattr(e, "start", None)
        b = getattr(e, "end", None)
        if a is None or b is None:
            a, b = e[0], e[1]
        out.append((int(a), int(b)))
    return out


class FacePatternEngine:
    """Landmark extraction + pattern rendering for tracked faces.

    analyze() refreshes the per-track landmark cache (throttled), draw()
    renders whatever is cached. A track's pattern is drawn only while it is
    unrecognized; run.py's face loop decides by simply not calling draw()
    for recognized tracks.
    """

    def __init__(self, model_path: str = PATTERN_MODEL_PATH):
        self.available = False
        self.style = STYLE if STYLE in ("contours", "tesselation") else "contours"
        self._landmarker = None
        self._contour_groups: list[tuple[list[tuple[int, int]], int]] = []
        self._tess_edges: list[tuple[int, int]] = []
        # track_id -> {"points": np.ndarray Nx2 (abs coords), "frame": int, "ts": int}
        self._cache: dict[str, dict] = {}
        self._ts_ms = 0  # strictly-increasing VIDEO-mode timestamp

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
        except ImportError:
            print("[Pattern] mediapipe not installed — rectangle fallback in use "
                  "(pip install mediapipe)")
            return

        if not os.path.isfile(model_path):
            print(f"[Pattern] {model_path} missing — run `python setup_models.py` "
                  "to fetch it; rectangle fallback in use")
            return

        try:
            options = vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=MAX_FACES,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:  # noqa: BLE001 — never let cosmetics kill vision
            print(f"[Pattern] FaceLandmarker init failed: {e}")
            return

        self._build_topology(vision)
        self.available = True
        print(f"[Pattern] Face Landmarker ready ({self.style} style, "
              f"every {EVERY_N_FRAMES} frame(s), up to {MAX_FACES} faces)")

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def _build_topology(self, vision):
        """Contour/tesselation edge lists from the Tasks API connection sets."""
        conn = getattr(vision, "FaceLandmarksConnections", None)

        def group(name: str, width: int, fallback) -> tuple[list, int]:
            raw = getattr(conn, name, None) if conn is not None else None
            return (_edges(raw) if raw else _edges(fallback)), width

        self._contour_groups = [
            group("FACE_LANDMARKS_FACE_OVAL", 2, _FACE_OVAL_FALLBACK),
            group("FACE_LANDMARKS_LEFT_EYE", 1, ()),
            group("FACE_LANDMARKS_RIGHT_EYE", 1, ()),
            group("FACE_LANDMARKS_LEFT_EYEBROW", 1, ()),
            group("FACE_LANDMARKS_RIGHT_EYEBROW", 1, ()),
            group("FACE_LANDMARKS_LIPS", 1, ()),
            group("FACE_LANDMARKS_NOSE", 1, ()),
        ]
        tess = getattr(conn, "FACE_LANDMARKS_TESSELATION", None) if conn else None
        self._tess_edges = _edges(tess) if tess else []
        if self.style == "contours" and not any(g[0] for g in self._contour_groups):
            print("[Pattern] WARNING: no contour topology available")

    # ------------------------------------------------------------------
    # Landmark extraction
    # ------------------------------------------------------------------

    def analyze(self, frame: np.ndarray, face_results: list[dict], frame_idx: int):
        """Refresh landmark caches for all unrecognized tracked faces.

        face_results: tracker output dicts (track_id, face_bbox, authorized...).
        Landmarks are extracted from the face's ROI only, every Nth frame per
        track; on skipped frames (or a transiently empty inference) the last
        known positions are reused.
        """
        if not self.available:
            return
        h, w = frame.shape[:2]
        for fr in face_results:
            if fr.get("authorized"):
                self._cache.pop(fr.get("track_id"), None)
                continue
            track_id = fr.get("track_id")
            if track_id is None:
                continue

            cached = self._cache.get(track_id)
            if cached is not None and frame_idx - cached["frame"] < EVERY_N_FRAMES:
                continue  # throttle: reuse last known landmarks

            x, y, fw, fh = fr["face_bbox"]
            cx, cy = x + fw / 2, y + fh / 2
            pad_x, pad_y = int(fw * ROI_PAD), int(fh * ROI_PAD)
            x1, y1 = max(0, int(x) - pad_x), max(0, int(y) - pad_y)
            x2, y2 = min(w, int(x + fw) + pad_x), min(h, int(y + fh) + pad_y)
            if x2 - x1 < MIN_ROI or y2 - y1 < MIN_ROI:
                continue  # face too small for the landmarker to mean anything

            roi = frame[y1:y2, x1:x2]
            points = self._infer_roi(roi, (x1, y1), (cx, cy))
            if points is not None:
                self._cache[track_id] = {"points": points, "frame": frame_idx}
            elif cached is None:
                # First attempt found nothing (profile angle, motion blur) —
                # nothing to draw yet; retry on the next throttle tick.
                self._cache[track_id] = {"points": None, "frame": frame_idx}

    def _infer_roi(self, roi: np.ndarray, origin: tuple[int, int],
                   track_center: tuple[float, float]) -> np.ndarray | None:
        """Run FaceLandmarker on one ROI, return best-matching Nx2 abs points."""
        import mediapipe as mp

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires strictly increasing timestamps per instance
        now_ms = int(time.perf_counter() * 1000)
        self._ts_ms = max(self._ts_ms + 1, now_ms)

        try:
            result = self._landmarker.detect_for_video(image, self._ts_ms)
        except Exception as e:  # noqa: BLE001
            print(f"[Pattern] inference error: {e}")
            return None
        if not result.face_landmarks:
            return None

        ox, oy = origin
        tcx, tcy = track_center
        roi_w = max(1, roi.shape[1])
        roi_h = max(1, roi.shape[0])
        best, best_dist = None, float("inf")
        for pts in result.face_landmarks:
            # Mesh centroid (first 468; ignore appended iris points) in abs coords
            arr = np.array([(p.x, p.y) for p in pts[:468]], dtype=np.float32)
            gx = ox + float(arr[:, 0].mean())
            gy = oy + float(arr[:, 1].mean())
            # Normalized distance (face-width units): only rejects gross
            # mismatches, e.g. a neighbor's face bleeding into this ROI
            dx = (gx - tcx) / roi_w
            dy = (gy - tcy) / roi_h
            d = dx * dx + dy * dy
            if d < best_dist:
                best, best_dist = arr, d

        # Reject a wild mismatch (centroid far outside this track's box)
        if best is None or best_dist > 1.0:
            return None
        # Scale normalized coords by the ROI size, then offset to full-frame
        abs_x = ox + best[:, 0] * roi_w
        abs_y = oy + best[:, 1] * roi_h
        return np.column_stack((abs_x, abs_y)).astype(np.float32)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray, track_id: str) -> bool:
        """Draw the cached pattern for a track. Returns True if drawn.

        Pure rendering: no recognition decisions — callers only invoke this
        for tracks whose status is unrecognized.
        """
        cache = self._cache.get(track_id)
        if not self.available or cache is None or cache["points"] is None:
            return False
        pts = cache["points"]
        h, w = frame.shape[:2]
        px = np.clip(pts[:, 0].astype(int), 0, w - 1)
        py = np.clip(pts[:, 1].astype(int), 0, h - 1)

        if self.style == "tesselation" and self._tess_edges:
            for a, b in self._tess_edges:
                cv2.line(frame, (px[a], py[a]), (px[b], py[b]), COLOR_GLOW, 1)

        for edges, width in self._contour_groups:
            if not edges:
                continue
            # glow underlay then bright core — cheap HUD look, ~150 lines
            for a, b in edges:
                cv2.line(frame, (px[a], py[a]), (px[b], py[b]), COLOR_GLOW, width + 2)
            for a, b in edges:
                cv2.line(frame, (px[a], py[a]), (px[b], py[b]), COLOR_MAIN, width)
        return True

    def forget(self, track_id: str):
        """Drop a track's cache (call when the tracker cleans it up)."""
        self._cache.pop(track_id, None)

    def prune(self, active_track_ids):
        """Drop caches for tracks the tracker no longer reports."""
        for tid in list(self._cache.keys()):
            if tid not in active_track_ids:
                del self._cache[tid]
