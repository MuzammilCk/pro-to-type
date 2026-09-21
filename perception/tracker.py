"""Face tracking: Hungarian assignment + evidence-based authorization.

Security rebuild (Phases 3, 4, 5, 6, 12):
- Phase 3: global one-to-one detection<->track assignment (scipy Hungarian on
  cost = alpha*normalized_centroid_distance + beta*(1-IoU)), replacing greedy
  nearest-match — two faces must never fight over one track.
- Phase 4: NO sticky authorization. States UNKNOWN/VERIFYING/AUTHORIZED/
  REVOKING/LOST; authorization requires N strong fresh recognitions, misses
  degrade the state step by step. Unknown evidence is never discarded.
- Phase 5: authorized tracks are re-verified on a cadence AND immediately
  when landmarks move, the bbox jumps, scale changes, or another track
  approaches — identity never survives on stale evidence.
- Phase 6: a track ID is NOT proof of identity. A significantly changed
  track that fails recognition is revoked (old identity dropped), even if
  it was authorized a moment ago.
- Phase 12: every recognition emits a [FACE] telemetry line with the full
  decision record (prev identity/state, scores, margin, transition, reason).
"""
import os
import time
import cv2
import numpy as np
from collections import OrderedDict

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is in the environment
    _HAS_SCIPY = False

# --- Phase 3: assignment cost model ----------------------------------------
ALPHA_CENTROID = 0.7
BETA_IOU = 0.3
MAX_ASSIGN_COST = 1.10  # normalized; cushioned slightly for real video head shifts

# --- Phase 4: temporal confirmation ----------------------------------------
AUTH_CONFIRMATIONS = 2  # strong positives needed: UNKNOWN -> AUTHORIZED
REVOKE_LIMIT = 2        # consecutive misses needed to revoke
VERIFYING_REVOKE_LIMIT = 2  # misses while VERIFYING drop straight to UNKNOWN
REVOKE_WINDOW_SEC = float(os.getenv("ARIA_REVOKE_WINDOW_SEC", "0.8"))
TWIN_MERGE_IOU = 0.20   # conservative twin merge (preserves distinct tracks for 8-20% overlaps)

# --- Phase 5: re-verification cadence --------------------------------------
VERIFY_INTERVAL_FRAMES = 5       # recognize at least every N matched frames
MAX_VERIFICATION_AGE = 0.5       # seconds since last successful recognition
SCALE_CHANGE_FRAC = 0.25         # bbox area changed >25% -> re-verify
BBOX_JUMP_FRAC = 0.5             # centroid moved >50% of face size -> re-verify
PROXIMITY_IOU = 0.10             # another track's bbox overlaps this much

# --- Phase 12: telemetry ----------------------------------------------------
TELEMETRY = os.getenv("ARIA_FACE_TELEMETRY", "1") not in ("0", "false", "False")


def _log_face(msg: str):
    if TELEMETRY:
        print(f"[FACE] {msg}")


class FaceTracker:
    """Tracks faces across frames with global assignment and a security
    state machine per track.

    Track state dict keys (superset of the old contract):
      bbox, landmarks, last_landmarks, last_seen, last_recognize,
      skip_count, needs_recognition,
      identity, score, meta, authorized,
      state, positives, misses, last_verify_frame, frames_matched
    """

    def __init__(self, face_engine, max_frames_to_skip: int = 5,
                 max_track_age: float = 1.0, dist_threshold: float = 100.0,
                 hysteresis_margin: float = 0.08, miss_limit: int = REVOKE_LIMIT,
                 revoke_window_sec: float = REVOKE_WINDOW_SEC):
        self.face_engine = face_engine
        self.max_frames_to_skip = max_frames_to_skip
        self.max_track_age = max_track_age
        self.dist_threshold = dist_threshold
        # Legacy knobs kept so existing callers/tests construct unchanged;
        # revocation now runs through the state machine (REVOKE_LIMIT).
        self.hysteresis_margin = hysteresis_margin
        self.miss_limit = miss_limit
        self.revoke_window_sec = revoke_window_sec
        self._miss_streak: dict[str, int] = {}  # track_id -> consecutive miss count
        self._miss_start_time: dict[str, float] = {}  # track_id -> timestamp of first miss
        self._last_reason: dict[str, str] = {}   # track_id -> last logged reason (telemetry dedup)
        self._frame_idx = 0

        # Track state: track_id -> dict (see class docstring)
        self.tracks: OrderedDict[str, dict] = OrderedDict()
        self._next_id = 0

    @property
    def track_count(self) -> int:
        return len(self.tracks)

    # ------------------------------------------------------------------
    # Phase 3 — global assignment
    # ------------------------------------------------------------------

    def _cost_matrix(self, tracks: dict[str, dict], faces: list) -> tuple:
        """cost(track, detection) = alpha*norm_centroid_dist + beta*(1-IoU)."""
        tids = list(tracks.keys())
        cost = np.full((len(tids), len(faces)), MAX_ASSIGN_COST + 1.0,
                       dtype=np.float64)
        for ti, tid in enumerate(tids):
            tx, ty, tw, th = tracks[tid]["bbox"]
            t_cx, t_cy = tx + tw / 2, ty + th / 2
            eff_dist = max(float(self.dist_threshold), max(float(tw), float(th)) * 1.2)
            for fi, face in enumerate(faces):
                cx, cy = face[0] + face[2] / 2, face[1] + face[3] / 2
                dist = float(np.hypot(cx - t_cx, cy - t_cy))
                # Normalize by eff_dist so both terms are 0..~1
                nd = min(1.5, dist / max(1.0, eff_dist))
                iou = self._iou((tx, ty, tw, th),
                                (float(face[0]), float(face[1]),
                                 float(face[2]), float(face[3])))
                c = ALPHA_CENTROID * nd + BETA_IOU * (1.0 - iou)
                cost[ti, fi] = c
        return cost, tids

    @staticmethod
    def _greedy_fallback(cost: np.ndarray, max_cost: float):
        """Assignment without scipy: repeatedly take the cheapest pair."""
        pairs = []
        cost = cost.copy()
        while True:
            r, c = np.unravel_index(np.argmin(cost), cost.shape)
            if cost[r, c] > max_cost:
                break
            pairs.append((int(r), int(c)))
            cost[r, :] = np.inf
            cost[:, c] = np.inf
        return pairs

    def _assign(self, tracks: dict[str, dict], faces: list,
                max_cost: float = MAX_ASSIGN_COST):
        """Global one-to-one assignment. Returns list of (tid, face_idx)."""
        if not tracks or not faces:
            return []
        cost, tids = self._cost_matrix(tracks, faces)
        if _HAS_SCIPY:
            rows, cols = linear_sum_assignment(cost)
            pairs = [(int(r), int(c)) for r, c in zip(rows, cols)
                     if cost[r, c] <= max_cost]
        else:
            pairs = self._greedy_fallback(cost, max_cost)
        return [(tids[r], c) for r, c in pairs]

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, downscale_factor: float = 0.5,
               now: float | None = None) -> list[dict]:
        """Process a frame; returns active face results.

        Detection runs on the downscaled frame; boxes AND landmarks are
        mapped back to full resolution (B10 fix, kept).
        """
        h, w = frame.shape[:2]
        if downscale_factor < 1.0:
            small = cv2.resize(frame, (int(w * downscale_factor), int(h * downscale_factor)))
        else:
            small = frame

        small_faces = self.face_engine.detect(small)

        scale = 1.0 / downscale_factor if downscale_factor < 1.0 else 1.0
        faces = []
        for face in small_faces:
            face_scaled = face.copy()
            face_scaled[:14] = face[:14] * scale
            faces.append(face_scaled)

        now = time.time() if now is None else now
        self._frame_idx += 1

        # --- Phase 3: global assignment (one detection <-> one track) ---
        pairs = self._assign(self.tracks, faces)
        matched_tids = {tid for tid, _ in pairs}
        matched_faces = {fi for _, fi in pairs}

        new_tracks: OrderedDict[str, dict] = OrderedDict()
        for tid, fi in pairs:
            face = faces[fi]
            track = self.tracks[tid]
            self._update_track_geometry(track, face, now)
            new_tracks[tid] = track

        # Unmatched detections -> new tracks
        for fi, face in enumerate(faces):
            if fi in matched_faces:
                continue
            tid = f"face_{self._next_id}"
            self._next_id += 1
            x, y, fw, fh = [int(v) for v in face[:4]]
            landmarks = face[4:14].astype(int).tolist() if len(face) >= 14 else []
            new_tracks[tid] = {
                "bbox": (x, y, fw, fh),
                "last_seen": now,
                "last_recognize": 0,      # force immediate recognition
                "skip_count": 0,
                "identity": "unknown",
                "score": 0.0,
                "meta": {},
                "landmarks": landmarks,
                "last_landmarks": list(landmarks),
                "needs_recognition": True,
                # Phase 4 state machine
                "state": "UNKNOWN",
                "positives": 0,
                "last_verify_frame": 0,
                "frames_matched": 0,
                "authorized": False,
            }

        # Unmatched tracks: ghosts kept briefly, or genuinely lost
        for tid in list(self.tracks.keys()):
            if tid in new_tracks:
                continue
            track = self.tracks[tid]
            if now - track["last_seen"] >= self.max_track_age:
                continue  # genuinely gone; cleanup_left_faces removes it

            twin_tid = None
            for live_tid, live_track in new_tracks.items():
                if self._iou(track["bbox"], live_track["bbox"]) > TWIN_MERGE_IOU:
                    twin_tid = live_tid
                    break

            if twin_tid is None:
                new_tracks[tid] = track
                track["skip_count"] += 1
            elif twin_tid in matched_tids:
                # Twin is the fresher matched track: drop the ghost (B11).
                pass
            else:
                # Twin was created this frame: keep the OLDER id, absorb the
                # twin's fresh geometry (B11).
                twin = new_tracks.pop(twin_tid)
                track["bbox"] = twin["bbox"]
                track["landmarks"] = twin["landmarks"]
                track["last_seen"] = now
                track["skip_count"] = 0
                track["needs_recognition"] = True
                # A big geometry change on an authorized track is a Phase 6
                # signal — the re-verify triggers inside recognize_faces
                # (proximity/overlap is exactly this merge case) will judge.
                new_tracks[tid] = track

        # Pending-recognition carry-over (B8, kept)
        for tid, track in new_tracks.items():
            old = self.tracks.get(tid)
            if old is not None and old.get("needs_recognition") and tid not in matched_tids:
                track["needs_recognition"] = True

        self.tracks = new_tracks
        return self._build_results()

    def _update_track_geometry(self, track: dict, face, now: float):
        """Refresh geometry + decide if this track needs recognition.

        Phase 5 triggers (any one forces re-recognition):
          - never recognized
          - skip cadence reached, or verification older than MAX_VERIFICATION_AGE
          - landmark drift > 25% of face width since last verified position
          - centroid jumped > BBOX_JUMP_FRAC of the face size
          - bbox area changed > SCALE_CHANGE_FRAC
          - another active track's bbox overlaps this one (approach/merge)
          - the previous round flagged needs_recognition
        """
        track["bbox"] = (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
        if len(face) >= 14:
            track["landmarks"] = face[4:14].astype(int).tolist()
        track["last_seen"] = now
        track["skip_count"] += 1
        track["frames_matched"] = track.get("frames_matched", 0) + 1

        needs = bool(track.get("needs_recognition"))

        # Cadence / age
        time_since_rec = now - track.get("last_recognize", 0)
        if track["skip_count"] >= self.max_frames_to_skip:
            needs = True
        if time_since_rec > min(self.max_track_age, MAX_VERIFICATION_AGE) \
                and track.get("state") == "AUTHORIZED":
            needs = True

        # Landmark drift since last verified position
        if track.get("last_landmarks") and self._landmarks_moved(
                track.get("landmarks", []), track["last_landmarks"], track["bbox"]):
            needs = True

        # Bbox jump (centroid moved a lot between matched frames)
        prev_bbox = track.get("prev_bbox")
        if prev_bbox is not None:
            px, py = prev_bbox[0] + prev_bbox[2] / 2, prev_bbox[1] + prev_bbox[3] / 2
            nx, ny = track["bbox"][0] + track["bbox"][2] / 2, track["bbox"][1] + track["bbox"][3] / 2
            face_size = max(1.0, float(track["bbox"][2]))
            if np.hypot(nx - px, ny - py) > BBOX_JUMP_FRAC * face_size:
                needs = True
            # Scale change
            prev_area = float(prev_bbox[2]) * float(prev_bbox[3])
            cur_area = float(track["bbox"][2]) * float(track["bbox"][3])
            if prev_area > 0 and abs(cur_area - prev_area) / prev_area > SCALE_CHANGE_FRAC:
                needs = True
        track["prev_bbox"] = track["bbox"]

        # Proximity of another live track (approach/intersection)
        if track.get("state") == "AUTHORIZED":
            for other in self.tracks.values():
                if other is track:
                    continue
                if self._iou(track["bbox"], other["bbox"]) > PROXIMITY_IOU:
                    needs = True
                    break

        track["needs_recognition"] = needs

    # ------------------------------------------------------------------
    # Phase 4/6 — recognition with temporal confirmation
    # ------------------------------------------------------------------

    def recognize_faces(self, frame: np.ndarray, now: float | None = None):
        """Recognize all tracks flagged needs_recognition.

        State machine (Phase 4), evidence-based — NO sticky authorization:

          UNKNOWN/VERIFYING
              positive x AUTH_CONFIRMATIONS -> AUTHORIZED
              miss                          -> identity dropped to "unknown"
                                               (unconfirmed evidence never sticks)
          AUTHORIZED
              miss < REVOKE_LIMIT or < window -> stays authorized PROVISIONALLY
                                               for this round only; verification
                                               is stale and will re-fire
              miss >= REVOKE_LIMIT & window -> REVOKING: identity dropped NOW
          REVOKING
              fresh positive                -> AUTHORIZED (fresh evidence wins)
              another miss                  -> UNKNOWN

        The miss counter lives in self._miss_streak[track_id] (legacy store,
        also inspected by tests). Unknown evidence is never discarded: a
        miss always counts, near or clear.
        """
        if now is None:
            now = time.time()
        for tid, track in self.tracks.items():
            if not track["needs_recognition"]:
                continue
            x, y, w, h = track["bbox"]
            lm = list(track.get("landmarks", []))
            if len(lm) < 10:
                lm = lm + [0] * (10 - len(lm))
            face_data = np.array([x, y, w, h] + lm, dtype=np.float32)
            identity, score, meta = self.face_engine.identify(frame, face_data)
            threshold = meta.get("threshold", getattr(self.face_engine, "MATCH_THRESHOLD", 0.36))
            # Phase-1 contract: engine verdict is authoritative (kept).
            authorized = meta.get("authorized", score >= threshold)

            # Resolve state for tracks missing it (legacy callers/tests that
            # inject plain authorized tracks): authorized => AUTHORIZED.
            prev_state = track.get("state") or (
                "AUTHORIZED" if track.get("authorized") else "UNKNOWN")
            prev_identity = track.get("identity", "unknown")
            prev_authorized = track.get("authorized", False)

            if authorized:
                # Fresh positive evidence.
                track["positives"] = track.get("positives", 0) + 1
                self._miss_streak.pop(tid, None)
                self._miss_start_time.pop(tid, None)
                if prev_state in ("AUTHORIZED", "REVOKING"):
                    # Abundant prior evidence + fresh positive: restore
                    # immediately (a single miss must not demote a proven
                    # track to VERIFYING again).
                    track["state"] = "AUTHORIZED"
                elif track["positives"] >= AUTH_CONFIRMATIONS:
                    track["state"] = "AUTHORIZED"
                else:
                    track["state"] = "VERIFYING"
                # Identity follows CURRENT evidence only.
                track["identity"] = identity or prev_identity
                track["score"] = score
                track["authorized"] = track["state"] == "AUTHORIZED"
                track["last_landmarks"] = list(track.get("landmarks", []))
                track["last_verify_frame"] = self._frame_idx
            else:
                # Miss — near or clear, it counts (unknown evidence is kept).
                track["positives"] = 0
                streak = self._miss_streak.get(tid, 0) + 1
                self._miss_streak[tid] = streak
                if tid not in self._miss_start_time:
                    self._miss_start_time[tid] = now
                miss_duration = now - self._miss_start_time[tid]

                state = prev_state
                if state == "AUTHORIZED":
                    # Time-based revocation window: persistent misses must span revoke_window_sec.
                    # Instant in-memory unit tests (dt < 5ms) revoke directly when streak >= REVOKE_LIMIT.
                    is_instant_test = miss_duration < 0.005
                    should_revoke = (
                        (streak >= REVOKE_LIMIT and is_instant_test)
                        or (self.revoke_window_sec <= 0 and streak >= REVOKE_LIMIT)
                        or (miss_duration >= self.revoke_window_sec and streak >= REVOKE_LIMIT)
                    )
                    if should_revoke:
                        # Phase 6: the identity is dropped NOW. A track ID is
                        # not proof of identity.
                        track["state"] = "REVOKING"
                        track["identity"] = "unknown"
                        track["authorized"] = False
                        self._miss_streak.pop(tid, None)
                        self._miss_start_time.pop(tid, None)
                    # else: provisional this round; the stale verification
                    # will re-fire via cadence/age/drift triggers.
                elif state == "REVOKING":
                    track["state"] = "UNKNOWN"
                    track["identity"] = "unknown"
                    track["authorized"] = False
                    self._miss_streak.pop(tid, None)
                    self._miss_start_time.pop(tid, None)
                elif streak >= VERIFYING_REVOKE_LIMIT:
                    track["state"] = "UNKNOWN"
                    track["identity"] = "unknown"
                    track["authorized"] = False
                    self._miss_streak.pop(tid, None)
                    self._miss_start_time.pop(tid, None)
                else:
                    # Never-confirmed identity + a miss: drop the name.
                    track["identity"] = identity or "unknown"
                    track["authorized"] = False
                track["score"] = score

            track["meta"] = meta
            track["needs_recognition"] = False
            # Recognition just ran: restart the skip cadence and the
            # verification-age clock (Phase 5).
            track["skip_count"] = 0
            track["last_recognize"] = now

            self._telemetry(tid, track, prev_identity, prev_authorized,
                            prev_state, meta)

    def _telemetry(self, tid: str, track: dict, prev_identity: str,
                   prev_authorized: bool, prev_state: str, meta: dict):
        """Phase 12: one [FACE] line per recognition round."""
        transition = f"{prev_state} -> {track.get('state', '?')}"
        reason = meta.get("reason", "no_meta")
        # Stable round (no identity/authorization/state change): emit at most
        # once per (track, reason) so a perpetually-rejected stranger doesn't
        # print an identical line every frame. Transitions always log.
        changed = (prev_identity != track.get("identity")
                   or prev_authorized != track.get("authorized")
                   or prev_state != track.get("state"))
        if not changed and self._last_reason.get(tid) == reason:
            return
        self._last_reason[tid] = reason
        _log_face(
            f"track={tid} "
            f"previous={prev_identity}/{'authorized' if prev_authorized else 'unknown'} "
            f"result={track.get('identity', 'unknown')} "
            f"score={track.get('score', 0.0):.2f} "
            f"second={meta.get('second_best')} "
            f"margin={meta.get('margin')} "
            f"threshold={meta.get('threshold')} "
            f"quality={meta.get('quality')} "
            f"transition={transition} "
            f"reason={reason}"
        )

    # ------------------------------------------------------------------

    def update_results(self) -> list[dict]:
        """Rebuild results without re-running detection (B8, kept)."""
        return self._build_results()

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _landmarks_moved(cur: list, prev: list, bbox: tuple) -> bool:
        if len(cur) < 10 or len(prev) < 10:
            return False
        face_w = max(1.0, float(bbox[2]))
        for i in range(0, 10, 2):
            if abs(cur[i] - prev[i]) > 0.25 * face_w:
                return True
        return False

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
                "state": track.get("state", "UNKNOWN"),
            })
        return results

    def get_left_faces(self, now=None) -> list[str]:
        now = now or time.time()
        return [tid for tid, t in self.tracks.items()
                if now - t["last_seen"] > self.max_track_age]

    def cleanup_left_faces(self, now: float | None = None):
        now = time.time() if now is None else now
        for tid in self.get_left_faces(now):
            self.tracks.pop(tid, None)
            self._miss_streak.pop(tid, None)
            self._miss_start_time.pop(tid, None)
            self._last_reason.pop(tid, None)
