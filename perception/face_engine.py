import os
import numpy as np
import cv2

FACE_DET_MODEL = "models/face_detection_yunet_2023mar.onnx"
FACE_REC_MODEL = "models/face_recognition_sface_2021dec.onnx"
FACES_DIR = "faces"

MIN_FACE_SIZE = 40  # Minimum face dimension in pixels for reliable recognition

# --- Security-rebuild contract (Phase 1) -----------------------------------
# All three are env-tunable; Phase 11 calibration sets the final values with
# measured FAR/FRR evidence instead of guesses.
#   RECOGNITION_THRESHOLD: min cosine similarity to accept an identity.
#     Kept at OpenCV's 0.36 default for now — baseline-recognition.json shows
#     genuine median 0.545 vs impostor max 0.173, so there is calibration room.
#   QUALITY_MIN: floor for face quality. POOR QUALITY NEVER LOWERS THE BAR —
#     the old adaptive relaxation (threshold - (1-quality)*0.08) is GONE.
#   MARGIN_MIN: best score must beat second-best identity by this much (when
#     2+ identities are enrolled) or the match is ambiguous and rejected.
RECOGNITION_THRESHOLD = float(os.getenv("ARIA_RECOGNITION_THRESHOLD", "0.36"))
QUALITY_MIN = float(os.getenv("ARIA_QUALITY_MIN", "0.30"))
MARGIN_MIN = float(os.getenv("ARIA_MARGIN_MIN", "0.05"))

# --- Phase 2: immutable enrollment contract --------------------------------
# Runtime recognition NEVER writes faces/*.npy. The only writer is
# enroll_identity(), and only after validation passes:
#   ENROLL_MIN_SAMPLES   independent captures required (plan: 3 min, 5 ideal)
#   ENROLL_DUPLICATE_SIM embeddings above this are the SAME sample, not
#                        independent evidence (blocks "one frame x5")
#   ENROLL_COHERENCE_MIN median pairwise similarity floor — below it the
#                        samples do not look like ONE person
ENROLL_MIN_SAMPLES = int(os.getenv("ARIA_ENROLL_MIN_SAMPLES", "3"))
ENROLL_TARGET_SAMPLES = 5
ENROLL_DUPLICATE_SIM = 0.995
ENROLL_COHERENCE_MIN = 0.40


class FaceEngine:
    """SFace face recognition pipeline using OpenCV 5 YuNet + SFace.

    Research-backed improvements:
    - L2-normalized embeddings for consistent cosine similarity
    - Multi-template enrollment (3-5 samples per person recommended)
    - Face quality assessment before embedding (size + landmark spread)
    - Adaptive threshold based on face quality
    - Template aggregation: matches against all stored templates, returns max score

    SFace: 37MB model, 128-D embeddings, 95% LFW accuracy on CPU.
    Threshold 0.36 (OpenCV default) tuned for YuNet + 5-landmark alignment.

    Recognition contract (Phase 1 security rebuild):
      identify() returns (identity, score, meta) where meta carries the FULL
      decision record:
        {quality, threshold, score, second_best, margin, margin_required,
         template_count, authorized, reason}
      Authorization requires ALL of:
        valid embedding AND quality >= QUALITY_MIN AND score >= threshold
        AND (margin >= MARGIN_MIN when 2+ identities enrolled)
      Low quality NEVER relaxes the threshold — it rejects.
    """

    def __init__(self):
        # Threshold is module-level config (env-tunable, calibration-set).
        self.MATCH_THRESHOLD = RECOGNITION_THRESHOLD
        self.detector = cv2.FaceDetectorYN_create(
            FACE_DET_MODEL, "", (320, 320),
            score_threshold=0.5, nms_threshold=0.3, top_k=5000,
        )
        self.recognizer = cv2.FaceRecognizerSF_create(
            FACE_REC_MODEL, "",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )
        self.known_faces: dict[str, list[np.ndarray]] = self._load_known()

    @staticmethod
    def _load_known() -> dict:
        known = {}
        if not os.path.isdir(FACES_DIR):
            return known
        for f in sorted(os.listdir(FACES_DIR)):
            if f.endswith(".npy"):
                name = os.path.splitext(f)[0]
                arr = np.load(os.path.join(FACES_DIR, f))
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                # Normalize on load — old templates may be unnormalized
                normalized = []
                for i in range(len(arr)):
                    vec = arr[i].astype(np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        vec = vec / norm
                    normalized.append(vec)
                known[name] = normalized
        return known

    def _blur_score(self, frame: np.ndarray, face) -> float:
        """Compute blur score via Laplacian variance of the face region.

        Higher = sharper, Lower = blurrier.
        Research: blurry faces significantly degrade SFace accuracy.
        """
        x, y, w, h = [int(v) for v in face[:4]]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return 0.0
        face_roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _face_quality(self, face, frame) -> float:
        """Assess face quality for adaptive thresholding.

        Factors: face size relative to frame, confidence score, blur.
        Returns quality score 0.0-1.0; lower quality → higher threshold.
        """
        x, y, w, h = [int(v) for v in face[:4]]
        h_frame, w_frame = frame.shape[:2]
        size_ratio = (w * h) / (w_frame * h_frame)
        # YuNet face row layout: [x, y, w, h, x_re, y_re, x_le, y_le,
        # x_nose, y_nose, x_mouth_r, y_mouth_r, x_mouth_l, y_mouth_l,
        # score]. face[4] is an eye COORDINATE, not the detection score.
        confidence = float(face[14]) if len(face) > 14 else 0.5
        size_score = min(1.0, size_ratio * 25)  # 4%+ of frame = full marks
        conf_score = max(0.5, min(1.0, confidence))
        blur = self._blur_score(frame, face)
        # Blur threshold: 100 is typical cutoff (variance of Laplacian)
        blur_score = min(1.0, blur / 100.0) if blur > 0 else 0.0
        return min(1.0, 0.4 * size_score + 0.3 * conf_score + 0.3 * blur_score)

    def detect(self, frame: np.ndarray, min_size: int = MIN_FACE_SIZE) -> np.ndarray:
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        faces = self.detector.detect(frame)
        if faces[1] is None:
            return np.empty((0, 0))

        # Filter by minimum face size
        valid = []
        for face in faces[1]:
            x, y, fw, fh = [int(v) for v in face[:4]]
            if fw >= min_size and fh >= min_size:
                valid.append(face)
        if not valid:
            return np.empty((0, 0))
        return np.array(valid)

    def _extract_aligned(self, frame: np.ndarray, face) -> np.ndarray | None:
        """Align face and extract embedding with L2 normalization.

        Returns None if alignment fails (poor landmark quality).
        """
        face = np.asarray(face, dtype=np.float32).flatten()
        # B10 fix: SFace's alignCrop() reads the eye/nose/mouth landmarks from
        # indices 4:14. A box-only array (len<5) makes it silently crop a
        # head-and-shoulders region; zero landmarks make it "align" a corner
        # patch. Both produce embeddings of NON-faces — the exact mechanism
        # behind enrolled faces scoring ~0.28 against themselves. Bail out
        # instead of embedding garbage; embed() falls back to detect()-fresh
        # landmarks so live enrollment still works.
        # 14 = bbox + 10 landmark floats (tracker rebuild); 15 = raw YuNet row
        # (bbox + landmarks + confidence). Both carry valid landmarks at 4:14.
        if len(face) < 14:
            return None
        lm = face[4:14]
        if not np.all(np.isfinite(lm)) or np.allclose(lm, 0.0, atol=1e-3):
            return None
        aligned = self.recognizer.alignCrop(frame, face)
        if aligned is None or aligned.size == 0:
            return None
        if aligned.shape[0] < 16 or aligned.shape[1] < 16:
            return None
        emb = self.recognizer.feature(aligned).flatten()
        # L2 normalize for consistent cosine similarity
        norm = np.linalg.norm(emb)
        if norm < 1e-6:
            return None
        return (emb / norm).astype(np.float32)

    def embed(self, frame: np.ndarray, face) -> np.ndarray:
        emb = self._extract_aligned(frame, face)
        if emb is None:
            emb = self.recognizer.feature(self.recognizer.alignCrop(frame, face)).flatten()
            emb = emb / (np.linalg.norm(emb) + 1e-6)
        return emb

    def identify(self, frame: np.ndarray, face) -> tuple[str, float, dict]:
        """Identify a face against known templates — full decision record.

        Returns (identity, score, meta):
          identity  best-matching name when authorized; "" when rejected
                    (callers must treat "" like "unknown")
          score     best cosine similarity (-1.0 when no valid embedding)
          meta      {quality, threshold, score, second_best, margin,
                     margin_required, template_count, authorized, reason}

        Authorization requires ALL of: valid embedding, quality >= QUALITY_MIN,
        score >= threshold, margin >= MARGIN_MIN (when 2+ identities are
        enrolled). Poor quality REJECTS — it never relaxes the threshold.
        """
        quality = self._face_quality(face, frame)
        threshold = self.MATCH_THRESHOLD

        def _reject(reason: str, score: float = -1.0,
                    second: float | None = None) -> tuple[str, float, dict]:
            meta = {
                "quality": round(quality, 3),
                "threshold": round(threshold, 3),
                "score": round(score, 4),
                "second_best": None if second is None else round(second, 4),
                "margin": None if second is None else round(score - second, 4),
                "margin_required": MARGIN_MIN if len(self.known_faces) >= 2 else 0.0,
                "template_count": 0,
                "authorized": False,
                "reason": reason,
            }
            return "", score, meta

        emb = self._extract_aligned(frame, face)
        if emb is None:
            return _reject("alignment_failed")
        if not self.known_faces:
            return _reject("no_identities_enrolled")
        if quality < QUALITY_MIN:
            # Security posture: low quality is EVIDENCE AGAINST — reject with
            # the threshold intact (the old code lowered the bar instead).
            return _reject("low_quality")

        # Best template per identity, then best/second-best across identities.
        best_name, best_score, second_best = "", -1.0, -1.0
        for name, templates in self.known_faces.items():
            person_best = -1.0
            for tmpl in templates:
                if tmpl.shape != emb.shape:
                    continue
                # Cosine similarity (dot product on L2-normalized vectors)
                score = float(np.dot(emb, tmpl))
                if score > person_best:
                    person_best = score
            if person_best > best_score:
                second_best = best_score
                best_name, best_score = name, person_best
            elif person_best >= second_best:
                second_best = person_best

        meta = {
            "quality": round(quality, 3),
            "threshold": round(threshold, 3),
            "score": round(best_score, 4),
            "second_best": None if second_best <= -1.0 else round(second_best, 4),
            "margin": None if second_best <= -1.0 else round(best_score - second_best, 4),
            "margin_required": MARGIN_MIN if len(self.known_faces) >= 2 else 0.0,
            "template_count": len(self.known_faces.get(best_name, [])),
        }

        if best_score < threshold:
            meta.update({"authorized": False, "reason": "threshold_rejection"})
            return "", best_score, meta
        if meta["second_best"] is not None and meta["margin"] < meta["margin_required"]:
            # Ambiguous: two identities within MARGIN_MIN — no safe verdict.
            meta.update({"authorized": False, "reason": "ambiguous_match"})
            return "", best_score, meta

        meta.update({"authorized": True, "reason": "match"})
        return best_name, best_score, meta

    def add_template(self, name: str, embedding: np.ndarray):
        """Add a face embedding template for a person (multi-template support)."""
        emb = embedding.astype(np.float32)
        # Normalize if not already
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm

        if name not in self.known_faces:
            self.known_faces[name] = []
        self.known_faces[name].append(emb)

        os.makedirs(FACES_DIR, exist_ok=True)
        arr = np.array(self.known_faces[name])
        np.save(os.path.join(FACES_DIR, f"{name}.npy"), arr)

    def enroll_identity(self, name: str, samples, *,
                        min_samples: int = ENROLL_MIN_SAMPLES,
                        quality_min: float | None = None) -> dict:
        """Explicit enrollment API — the ONLY path that writes faces/<name>.npy.

        Phase 2 security contract:
        - Runtime recognition never writes templates; this method is called
          exclusively by explicit enrollment flows (enroll.py, reenroll.py,
          run.py voice enrollment).
        - samples: iterable of (frame, face_row) pairs captured from
          INDEPENDENT moments (person moved between captures), or plain
          embedding vectors (ndarray) when the caller has already embedded.
          Re-embedding one frame N times can NEVER pass: duplicates are
          dropped by ENROLL_DUPLICATE_SIM and the remainder fails the
          min_samples gate.
        - Per-sample validation: embedding validity (alignment + norm),
          quality >= QUALITY_MIN (poor samples are DISCARDED, never stored).
        - Set validation: duplicate rejection, diversity/coherence check
          (median pairwise similarity must sit between 'all identical' and
          'not one person').
        - On success it REPLACES the identity's templates with the validated
          set (a re-enrollment cannot inherit poisoned templates).

        Returns a decision record:
          {"enrolled": bool, "reason": str, "templates": int,
           "n_samples": int, "accepted": int,
           "rejected_low_quality": int, "rejected_invalid": int,
           "rejected_duplicate": int,
           "median_intra": float | None, "min_intra": float | None}
        """
        quality_min = QUALITY_MIN if quality_min is None else quality_min
        result = {
            "enrolled": False, "reason": "", "templates": 0,
            "n_samples": 0, "accepted": 0,
            "rejected_low_quality": 0, "rejected_invalid": 0,
            "rejected_duplicate": 0,
            "median_intra": None, "min_intra": None,
        }

        accepted: list[np.ndarray] = []
        samples = list(samples)
        result["n_samples"] = len(samples)
        for sample in samples:
            if isinstance(sample, (tuple, list)):
                if len(sample) != 2:
                    result["rejected_invalid"] += 1
                    continue
                frame, face = sample
                emb = self._extract_aligned(frame, face)
                if emb is None or len(emb) == 0:
                    result["rejected_invalid"] += 1
                    continue
                if self._face_quality(face, frame) < quality_min:
                    # Poor quality is evidence AGAINST storage — never relax.
                    result["rejected_low_quality"] += 1
                    continue
            else:
                emb = np.asarray(sample, dtype=np.float32).flatten()
                norm = np.linalg.norm(emb)
                if norm < 1e-6:
                    result["rejected_invalid"] += 1
                    continue
                emb = (emb / norm).astype(np.float32)

            # Duplicate detection: near-identical embedding == same sample.
            if any(float(np.dot(emb, a)) > ENROLL_DUPLICATE_SIM for a in accepted):
                result["rejected_duplicate"] += 1
                continue
            accepted.append(emb)

        result["accepted"] = len(accepted)
        if len(accepted) < min_samples:
            result["reason"] = (
                f"insufficient_valid_samples({len(accepted)}<{min_samples}; "
                f"dup={result['rejected_duplicate']}, "
                f"low_q={result['rejected_low_quality']}, "
                f"invalid={result['rejected_invalid']})"
            )
            return result

        # Diversity / coherence over the accepted set.
        pairwise = [float(np.dot(accepted[i], accepted[j]))
                    for i in range(len(accepted))
                    for j in range(i + 1, len(accepted))]
        if pairwise:
            med = float(np.median(pairwise))
            result["median_intra"] = round(med, 4)
            result["min_intra"] = round(min(pairwise), 4)
            if med > ENROLL_DUPLICATE_SIM:
                result["reason"] = "no_diversity"
                return result
            if med < ENROLL_COHERENCE_MIN:
                # Samples don't cohere to ONE person (multi-face capture,
                # garbage frames, or genuinely different people).
                result["reason"] = "inconsistent_embeddings"
                return result

        # Store ONLY validated templates (replace, not append).
        arr = np.array(accepted, dtype=np.float32)
        self.known_faces[name] = [a.copy() for a in accepted]
        os.makedirs(FACES_DIR, exist_ok=True)
        np.save(os.path.join(FACES_DIR, f"{name}.npy"), arr)
        result["enrolled"] = True
        result["reason"] = "ok"
        result["templates"] = len(accepted)
        return result
