import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple  # (x, y, w, h)


class Detector:
    def __init__(self, model_path: str):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.class_names = self._load_coco_names()

    @staticmethod
    def _load_coco_names():
        return [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie",
            "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog",
            "pizza", "donut", "cake", "chair", "couch", "potted plant",
            "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "keyboard", "cell phone", "microwave", "oven", "toaster",
            "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush",
        ]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1 / 255.0, size=(640, 640),
            mean=(0, 0, 0), swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        preds = self.net.forward()[0]

        boxes, scores, labels = [], [], []
        for pred in preds:
            conf = float(pred[4])
            if conf < 0.25:
                continue
            class_id = int(np.argmax(pred[5:]))
            class_conf = conf * float(pred[5 + class_id])
            if class_conf < 0.25:
                continue
            cx, cy, w, h = (pred[:4] * np.array([width, height, width, height])).flatten()
            x = int(cx - w / 2)
            y = int(cy - h / 2)
            boxes.append((x, y, int(w), int(h)))
            scores.append(class_conf)
            labels.append(self.class_names[class_id])

        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.25, nms_threshold=0.45)
        indices = np.asarray(indices).flatten().astype(int)
        results = []
        for i in indices:
            results.append(Detection(
                label=labels[i],
                confidence=round(scores[i], 3),
                bbox=boxes[i],
            ))
        return results

    @staticmethod
    def draw(frame: np.ndarray, det: Detection) -> None:
        x, y, w, h = det.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{det.label} {det.confidence:.2f}", (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
