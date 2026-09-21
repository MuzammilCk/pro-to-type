import json
import os
import time
from typing import Any

import cv2
try:
    import boto3
    from botocore.exceptions import ClientError
    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False


class Alerter:
    """Alert actuation engine supporting local high-res evidence persistence and cloud SNS escalation."""

    def __init__(self, enabled: bool = False, evidence_dir: str = "./evidence"):
        self.evidence_dir = evidence_dir
        os.makedirs(self.evidence_dir, exist_ok=True)
        self.history: list[dict[str, Any]] = []

        self.enabled = bool(enabled and os.getenv("ALERT_SNS_TOPIC_ARN") and _HAS_BOTO3)
        if self.enabled:
            self.sns = boto3.client("sns", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
            self.topic_arn = os.getenv("ALERT_SNS_TOPIC_ARN")

    def save_evidence(
        self,
        frame: Any,
        bbox: tuple[int, int, int, int] | list[int] | None = None,
        prefix: str = "incident",
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Save a high-resolution crop or full frame to local evidence storage.

        Args:
            frame: OpenCV image matrix (numpy array).
            bbox: Optional bounding box [x, y, w, h] to crop. If None or invalid, whole frame is saved.
            prefix: Filename prefix (e.g. 'incident', 'anomaly_warn', 'anomaly_halt').
            metadata: Optional dictionary saved as an accompanying metadata JSON sidecar.

        Returns:
            The filepath where the evidence image was saved, or None if frame is invalid.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return None

        os.makedirs(self.evidence_dir, exist_ok=True)
        ih, iw = frame.shape[:2]

        save_img = frame
        if bbox is not None and len(bbox) == 4:
            try:
                x, y, w, h = [int(v) for v in bbox]
                x1 = max(0, min(iw, x))
                y1 = max(0, min(ih, y))
                x2 = max(0, min(iw, x + w))
                y2 = max(0, min(ih, y + h))
                if x2 > x1 and y2 > y1:
                    save_img = frame[y1:y2, x1:x2]
            except Exception:
                save_img = frame

        ts = int(time.time() * 1000)
        filename = f"{prefix}_{ts}.jpg"
        filepath = os.path.join(self.evidence_dir, filename)

        success = cv2.imwrite(filepath, save_img)
        if not success:
            print(f"[Alerter] Failed to save evidence image to {filepath}")
            return None

        if metadata:
            meta_path = os.path.join(self.evidence_dir, f"{prefix}_{ts}.json")
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, default=str, indent=2)
            except Exception as e:
                print(f"[Alerter] Warning: failed to save metadata sidecar: {e}")

        return filepath

    def fire(
        self,
        reason: dict[str, Any],
        frame: Any = None,
        bbox: tuple[int, int, int, int] | list[int] | None = None,
    ) -> str:
        """Process an alert event, persist evidence locally, and optionally publish to SNS."""
        ev_frame = frame if frame is not None else reason.get("frame")
        ev_bbox = bbox if bbox is not None else reason.get("bbox")

        saved_path = None
        if ev_frame is not None:
            clean_meta = {k: v for k, v in reason.items() if k != "frame"}
            saved_path = self.save_evidence(
                ev_frame,
                bbox=ev_bbox,
                prefix="incident",
                metadata=clean_meta,
            )
            if saved_path:
                reason["evidence_path"] = saved_path

        clean_reason = {k: v for k, v in reason.items() if k != "frame"}
        msg = json.dumps(clean_reason, default=lambda o: getattr(o, "__dict__", str(o)), indent=2)
        print(f"[ALERT] {msg}")

        self.history.append({
            "timestamp": time.time(),
            "reason": clean_reason,
            "evidence_path": saved_path,
        })

        if self.enabled:
            try:
                self.sns.publish(TopicArn=self.topic_arn, Message=msg, Subject="Vision Alert")
                return "sns_sent"
            except ClientError as e:
                print(f"[ERROR] SNS publish failed: {e}")
                return "sns_failed"

        return "local_saved" if saved_path else "local_only"
