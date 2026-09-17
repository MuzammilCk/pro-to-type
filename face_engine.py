import os
import numpy as np
import cv2

FACE_DET_MODEL = "models/face_detection_yunet_2023mar.onnx"
FACE_REC_MODEL = "models/face_recognition_sface_2021dec.onnx"
FACES_DIR = "faces"

MIN_FACE_SIZE = 40  # Minimum face dimension in pixels for reliable recognition


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
    """

    def __init__(self):
        self.MATCH_THRESHOLD = 0.36
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
        """Identify a face against known templates.

        Returns (name, score, metadata) where score is cosine similarity.
        Matches against ALL stored templates per person and returns the max.
        Includes quality metadata for adaptive thresholding.
        """
        quality = self._face_quality(face, frame)
        emb = self._extract_aligned(frame, face)
        if emb is None:
            return "unknown", 0.0, {"quality": quality, "error": "alignment_failed"}

        # Adapt threshold based on face quality: lower quality → relax threshold slightly
        adaptive_threshold = self.MATCH_THRESHOLD - (1.0 - quality) * 0.08
        adaptive_threshold = min(self.MATCH_THRESHOLD, max(0.28, adaptive_threshold))

        best_name, best_score = "unknown", -1.0
        for name, templates in self.known_faces.items():
            for tmpl in templates:
                if tmpl.shape != emb.shape:
                    continue
                # Cosine similarity (dot product on L2-normalized vectors)
                score = float(np.dot(emb, tmpl))
                if score > best_score:
                    best_name, best_score = name, score

        is_authorized = best_score >= adaptive_threshold
        return (best_name if is_authorized else "unknown", best_score, {
            "quality": round(quality, 3),
            "threshold": round(adaptive_threshold, 3),
            "match": best_score,
            "template_count": len(self.known_faces.get(best_name, [])),
        })

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

    def capture_templates(self, frame: np.ndarray, face, name: str, count: int = 5) -> int:
        """Capture multiple face embeddings for robust enrollment.

        Returns number of templates actually captured.
        """
        saved = 0
        for _ in range(count):
            emb = self._extract_aligned(frame, face)
            if emb is not None and len(emb) > 0:
                self.add_template(name, emb)
                saved += 1
            # Small delay would normally go here; in batch mode, just re-extract
        return saved
