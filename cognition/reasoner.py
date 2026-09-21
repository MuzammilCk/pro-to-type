import numpy as np
from perception.detector import Detection


class Reasoner:
    WATCHED_LABELS = {"person", "cat", "dog"}
    AUTHORIZED = set()

    def __init__(self, face_engine=None):
        self.face_engine = face_engine

    def set_authorized(self, names: set[str]):
        self.AUTHORIZED = names

    def evaluate(self, detections: list[Detection], frame=None) -> dict:
        person_dets = [
            d for d in detections
            if d.label in self.WATCHED_LABELS and d.confidence > 0.7
        ]

        face_results = []
        threats = []

        if self.face_engine is not None and frame is not None:
            faces = self.face_engine.detect(frame)
            for face in faces:
                x, y, w, h = [int(v) for v in face[:4]]
                landmarks = face[4:14].astype(int).tolist()
                name, dist, meta = self.face_engine.identify(frame, face)
                authorized = name in self.AUTHORIZED
                face_results.append({
                    "identity": name,
                    "authorized": authorized,
                    "distance": round(dist, 3),
                    "face_bbox": (x, y, w, h),
                    "landmarks": landmarks,
                    "quality": meta.get("quality", 0),
                })
                if not authorized:
                    threats.append({
                        "identity": name,
                        "distance": round(dist, 3),
                        "face_bbox": (x, y, w, h),
                    })

        should_act = len(threats) > 0
        return {
            "should_act": should_act,
            "threats": threats,
            "face_results": face_results,
            "total": len(detections),
            "reason": "unauthorized_person" if threats else ("authorized_or_no_person"),
        }
