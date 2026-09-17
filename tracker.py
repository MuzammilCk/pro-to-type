"""Face tracking with centroid-distance matching + frame-skipping.

Based on ARIA_ARCHITECTURE.md: 'Track faces frame-to-frame with simple
centroid-distance matching; only trigger a fresh recognition call when
a track is new or hasn't matched in a while. Downscale before detection.'
"""
import time
import cv2
import numpy as np
from collections import OrderedDict


class FaceTracker:
    """Tracks detected faces across frames using centroid matching.

    - Reuses face ROIs between frames to avoid redundant recognition
    - Only triggers recognition on new tracks or stale tracks
    - Frame-skipping: skips detection every N frames on tracked faces
    """

    def __init__(self, face_engine, max_frames_to_skip: int = 5,
                 max_track_age: float = 1.0, dist_threshold: float = 100.0,
                 hysteresis_margin: float = 0.08, miss_limit: int = 3):
        self.face_engine = face_engine
        self.max_frames_to_skip = max_frames_to_skip
        self.max_track_age = max_track_age  # seconds before a face is "left"
        self.dist_threshold = dist_threshold  # pixels for centroid matching
        # B9 hysteresis: authorization flips need more evidence than one frame
        self.hysteresis_margin = hysteresis_margin  # extra similarity needed to REVOKE authorized
        self.miss_limit = miss_limit  # consecutive clear misses before revoking
        self._miss_streak: dict[str, int] = {}  # track_id -> consecutive clear-miss count

        # Track state: track_id -> dict(bbox, last_seen, last_recognize, skip_count, identity, score, meta)
        self.tracks: OrderedDict[str, dict] = OrderedDict()
        self._next_id = 0

    @property
    def track_count(self) -> int:
        return len(self.tracks)

    def update(self, frame: np.ndarray, downscale_factor: float = 0.5) -> list[dict]:
        """Process a frame, return active face results.

        Args:
            frame: Full-resolution frame
            downscale_factor: Fraction to downscale before detection (performance)

        Returns:
            List of face result dicts with keys:
            track_id, identity, authorized, score, face_bbox, landmarks,
            quality, meta, needs_recognition
        """
        h, w = frame.shape[:2]

        # Run detection on downscaled frame for speed
        if downscale_factor < 1.0:
            small = cv2.resize(frame, (int(w * downscale_factor), int(h * downscale_factor)))
        else:
            small = frame

        small_faces = self.face_engine.detect(small)

        # Map downscaled face boxes back to full resolution
        scale = 1.0 / downscale_factor if downscale_factor < 1.0 else 1.0
        faces = []
        for face in small_faces:
            x, y, fw, fh = [float(v) for v in face[:4]]
            face_scaled = face.copy()
            face_scaled[0] = x * scale
            face_scaled[1] = y * scale
            face_scaled[2] = fw * scale
            face_scaled[3] = fh * scale
            faces.append(face_scaled)

        now = time.time()

        # Match existing tracks to new detections via centroid distance
        matched = set()
        new_tracks = OrderedDict()

        for face in faces:
            cx, cy = face[0] + face[2] / 2, face[1] + face[3] / 2

            best_track_id = None
            best_dist = float("inf")
            for tid, track in self.tracks.items():
                tx, ty, tw, th = track["bbox"]
                t_cx, t_cy = tx + tw / 2, ty + th / 2
                dist = np.sqrt((cx - t_cx) ** 2 + (cy - t_cy) ** 2)
                if dist < self.dist_threshold and dist < best_dist:
                    best_dist = dist
                    best_track_id = tid

            if best_track_id is not None:
                matched.add(best_track_id)
                track = self.tracks[best_track_id]
                track["bbox"] = (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
                track["last_seen"] = now
                track["skip_count"] += 1
                if track.get("needs_recognition"):
                    # Recognition pending — detection already re-confirmed the
                    # face, don't also bump skip_count toward a re-recognize
                    track["skip_count"] = 0

                # Re-recognize if track is stale or never recognized
                time_since_rec = now - track.get("last_recognize", 0)
                if track["skip_count"] >= self.max_frames_to_skip or time_since_rec > self.max_track_age:
                    track["skip_count"] = 0
                    track["last_recognize"] = now
                    track["needs_recognition"] = True
                else:
                    track["needs_recognition"] = False

                new_tracks[best_track_id] = track
            else:
                # New track
                tid = f"face_{self._next_id}"
                self._next_id += 1
                x, y, fw, fh = [int(v) for v in face[:4]]
                landmarks = face[4:14].astype(int).tolist() if len(face) > 14 else []
                new_tracks[tid] = {
                    "bbox": (x, y, fw, fh),
                    "last_seen": now,
                    "last_recognize": 0,  # Force immediate recognition
                    "skip_count": 0,
                    "identity": "unknown",
                    "score": 0.0,
                    "meta": {},
                    "landmarks": landmarks,
                    "needs_recognition": True,
                }

        # Check for lost tracks (not seen for a while)
        for tid in list(self.tracks.keys()):
            if tid not in new_tracks:
                track = self.tracks[tid]
                if now - track["last_seen"] < self.max_track_age:
                    # Keep briefly — might reappear
                    new_tracks[tid] = track
                    track["skip_count"] += 1

        # Carry pending-recognition state across the rebuild (B8: a second
        # update() used to wipe it, forcing recognition to run every frame)
        for tid, track in new_tracks.items():
            old = self.tracks.get(tid)
            if old is not None and old.get("needs_recognition") and tid not in matched:
                track["needs_recognition"] = True

        self.tracks = new_tracks
        return self._build_results()

    def update_results(self) -> list[dict]:
        """Rebuild results from current track state WITHOUT re-running detection.

        Used after recognize_faces() so the vision loop doesn't pay for a
        second full detect+match pass on the same frame (B8 fix).
        """
        return self._build_results()

    def recognize_faces(self, frame: np.ndarray):
        """Run face recognition on all tracks that need it.

        Applies authorization hysteresis (B9): once a track is authorized,
        a single bad frame doesn't flip it to unknown — it takes a streak of
        clear misses. This stops greeting/re-alert chatter when a face turns,
        blurs, or is briefly occluded.
        """
        for tid, track in self.tracks.items():
            if not track["needs_recognition"]:
                continue
            x, y, w, h = track["bbox"]
            face_data = np.array([x, y, w, h] + track.get("landmarks", []))
            identity, score, meta = self.face_engine.identify(frame, face_data)
            threshold = meta.get("threshold", self.face_engine.MATCH_THRESHOLD)
            currently_authorized = track.get("authorized", False)

            if score >= threshold:
                track["identity"] = identity
                track["score"] = score
                track["authorized"] = True
                self._miss_streak.pop(tid, None)
            elif currently_authorized and score >= threshold - self.hysteresis_margin:
                # Near-miss while authorized: keep the authorized identity (sticky)
                track["score"] = score
                track["meta"] = meta
                self._miss_streak.pop(tid, None)
            elif currently_authorized:
                streak = self._miss_streak.get(tid, 0) + 1
                self._miss_streak[tid] = streak
                if streak >= self.miss_limit:
                    track["identity"] = identity
                    track["score"] = score
                    track["authorized"] = False
                    self._miss_streak.pop(tid, None)
                # else: keep previous authorized state this round
            else:
                track["identity"] = identity
                track["score"] = score
                track["authorized"] = False
                self._miss_streak.pop(tid, None)

            track["meta"] = meta
            track["needs_recognition"] = False

    def _build_results(self) -> list[dict]:
        results = []
        for tid, track in self.tracks.items():
            x, y, w, h = track["bbox"]
            results.append({
                "track_id": tid,
                "identity": track.get("identity", "unknown"),
                "authorized": track.get("authorized", False),
                "distance": round(track.get("score", 0.0), 3),
                "face_bbox": (x, y, w, h),
                "landmarks": track.get("landmarks", []),
                "quality": track.get("meta", {}).get("quality", 0.0),
                "needs_recognition": track.get("needs_recognition", False),
            })
        return results

    def get_left_faces(self, now=None) -> list[str]:
        """Return track IDs that have been gone for longer than max_track_age."""
        now = now or time.time()
        return [tid for tid, t in self.tracks.items()
                if now - t["last_seen"] > self.max_track_age]

    def cleanup_left_faces(self):
        """Remove tracks that are no longer visible."""
        now = time.time()
        for tid in self.get_left_faces(now):
            del self.tracks[tid]
            self._miss_streak.pop(tid, None)
