"""Active OpenCV 5 Micro-Inspection Engine (Phase 9).

Provides stateless, CPU-optimized computer vision routines designed to be
invoked directly by the cognitive ReAct loop as tools:
1. crop_roi: Safe sub-region extraction with boundary clamping and optional CLAHE.
2. inspect_color_hsv: Morphological color segmentation for indicators/safety tags.
3. analyze_geometry: Edge detection, contours, aspect ratio, solidity, and Laplacian sharpness.
4. measure_optical_flow: Sub-regional Farneback optical flow to quantify velocity and motion.

All routines are pure static functions guaranteed to complete in <5ms on CPU
without raising unhandled exceptions on invalid or empty inputs.
"""
from typing import Any
import numpy as np
import cv2


class VisionInspectionEngine:
    """Stateless, CPU-optimized active computer vision inspection suite."""

    @staticmethod
    def crop_roi(
        frame: np.ndarray | None,
        bbox: tuple[int, int, int, int] | list[int] | None,
        enhance: bool = False,
    ) -> np.ndarray:
        """Extract a sub-region from a frame with strict boundary clamping.

        Args:
            frame: Source image (BGR or Grayscale).
            bbox: (x, y, w, h) bounding box.
            enhance: If True, applies CLAHE contrast enhancement on the crop.

        Returns:
            Extracted crop as contiguous ndarray, or empty ndarray (0, 0, 3) on invalid input.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return np.empty((0, 0, 3), dtype=np.uint8)

        if not bbox or len(bbox) != 4:
            return np.empty((0, 0, 3), dtype=np.uint8)

        try:
            x, y, w, h = [int(v) for v in bbox]
        except (ValueError, TypeError):
            return np.empty((0, 0, 3), dtype=np.uint8)

        if w <= 0 or h <= 0:
            return np.empty((0, 0, 3), dtype=np.uint8)

        ih, iw = frame.shape[:2]
        x1 = max(0, min(x, iw))
        y1 = max(0, min(y, ih))
        x2 = max(0, min(x + w, iw))
        y2 = max(0, min(y + h, ih))

        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)

        crop = frame[y1:y2, x1:x2]
        if enhance and crop.size > 0:
            crop = VisionInspectionEngine._apply_clahe(crop)

        return np.ascontiguousarray(crop)

    @staticmethod
    def inspect_color_hsv(
        roi: np.ndarray | None,
        lower_hsv: tuple[int, int, int] | list[int],
        upper_hsv: tuple[int, int, int] | list[int],
        min_threshold: float = 0.05,
    ) -> dict[str, Any]:
        """Perform morphological color segmentation in HSV color space.

        Args:
            roi: Region of interest (BGR image).
            lower_hsv: Lower bound (H: 0-179, S: 0-255, V: 0-255).
            upper_hsv: Upper bound (H: 0-179, S: 0-255, V: 0-255).
            min_threshold: Minimum coverage ratio to flag detection (default 0.05).

        Returns:
            Structured dictionary with coverage ratio, matched pixel count, and presence flag.
        """
        empty_res = {
            "coverage_ratio": 0.0,
            "matched_pixels": 0,
            "total_pixels": 0,
            "detected": False,
            "dominant_hsv": [0, 0, 0],
        }
        if roi is None or not isinstance(roi, np.ndarray) or roi.size == 0:
            return empty_res

        total_pixels = int(roi.shape[0] * roi.shape[1])
        if total_pixels == 0:
            return empty_res

        try:
            if len(roi.shape) == 2:
                hsv = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
            elif roi.shape[2] == 4:
                bgr = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            else:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            low = np.array(lower_hsv, dtype=np.uint8)
            high = np.array(upper_hsv, dtype=np.uint8)

            mask = cv2.inRange(hsv, low, high)

            # Morphological noise removal (3x3 open)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            matched_pixels = int(cv2.countNonZero(mask))
            coverage = float(matched_pixels / total_pixels) if total_pixels > 0 else 0.0

            dominant = [0, 0, 0]
            if matched_pixels > 0:
                mean_val = cv2.mean(hsv, mask=mask)
                dominant = [int(round(v)) for v in mean_val[:3]]

            return {
                "coverage_ratio": round(coverage, 4),
                "matched_pixels": matched_pixels,
                "total_pixels": total_pixels,
                "detected": bool(coverage >= min_threshold),
                "dominant_hsv": dominant,
            }
        except Exception as e:
            empty_res["error"] = str(e)
            return empty_res

    @staticmethod
    def analyze_geometry(roi: np.ndarray | None) -> dict[str, Any]:
        """Analyze structural geometry, contours, aspect ratio, and image sharpness.

        Args:
            roi: Region of interest (BGR or Grayscale).

        Returns:
            Structured dictionary with geometric descriptors.
        """
        empty_res = {
            "contour_count": 0,
            "max_area": 0.0,
            "aspect_ratio": 0.0,
            "solidity": 0.0,
            "sharpness": 0.0,
            "hull_vertices": 0,
            "has_structure": False,
        }
        if roi is None or not isinstance(roi, np.ndarray) or roi.size == 0:
            return empty_res

        if roi.shape[0] < 3 or roi.shape[1] < 3:
            return empty_res

        try:
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi

            # 1. Image sharpness via Laplacian variance
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # 2. Edge discovery via Canny
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # 3. Contour analysis
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_count = len(contours)

            max_area = 0.0
            aspect_ratio = 0.0
            solidity = 0.0
            hull_vertices = 0

            if contour_count > 0:
                c = max(contours, key=cv2.contourArea)
                max_area = float(cv2.contourArea(c))
                _, _, cw, ch = cv2.boundingRect(c)
                aspect_ratio = float(cw / ch) if ch > 0 else 0.0

                hull = cv2.convexHull(c)
                hull_area = float(cv2.contourArea(hull))
                solidity = float(max_area / hull_area) if hull_area > 0 else 0.0
                hull_vertices = int(len(hull))

            return {
                "contour_count": contour_count,
                "max_area": round(max_area, 2),
                "aspect_ratio": round(aspect_ratio, 3),
                "solidity": round(solidity, 3),
                "sharpness": round(sharpness, 2),
                "hull_vertices": hull_vertices,
                "has_structure": bool(contour_count > 0 and max_area >= 10.0),
            }
        except Exception as e:
            empty_res["error"] = str(e)
            return empty_res

    @staticmethod
    def measure_optical_flow(
        prev_roi: np.ndarray | None,
        curr_roi: np.ndarray | None,
    ) -> dict[str, Any]:
        """Compute Gunnar Farneback dense optical flow between consecutive crops.

        Args:
            prev_roi: Previous frame ROI.
            curr_roi: Current frame ROI.

        Returns:
            Structured dictionary with velocity magnitude, direction, and movement flag.
        """
        empty_res = {
            "mean_velocity": 0.0,
            "max_velocity": 0.0,
            "motion_direction": 0.0,
            "is_moving": False,
        }
        if (
            prev_roi is None
            or curr_roi is None
            or not isinstance(prev_roi, np.ndarray)
            or not isinstance(curr_roi, np.ndarray)
            or prev_roi.size == 0
            or curr_roi.size == 0
        ):
            return empty_res

        if prev_roi.shape[:2] != curr_roi.shape[:2]:
            return empty_res

        if prev_roi.shape[0] < 5 or prev_roi.shape[1] < 5:
            return empty_res

        try:
            prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY) if len(prev_roi.shape) == 3 else prev_roi
            curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY) if len(curr_roi.shape) == 3 else curr_roi

            h, w = prev_gray.shape[:2]
            scale = 1.0
            if max(h, w) > 96:
                scale = 96.0 / max(h, w)
                nw, nh = int(round(w * scale)), int(round(h * scale))
                p_flow = cv2.resize(prev_gray, (nw, nh), interpolation=cv2.INTER_AREA)
                c_flow = cv2.resize(curr_gray, (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                p_flow, c_flow = prev_gray, curr_gray

            # OpenCV Farneback optical flow (fast CPU preset)
            flow = cv2.calcOpticalFlowFarneback(
                p_flow,
                c_flow,
                None,
                pyr_scale=0.5,
                levels=1,
                winsize=11,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )

            if scale != 1.0:
                flow = flow / scale

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
            mean_vel = float(np.mean(mag))
            max_vel = float(np.max(mag))
            mean_angle = float(np.mean(ang))

            return {
                "mean_velocity": round(mean_vel, 3),
                "max_velocity": round(max_vel, 3),
                "motion_direction": round(mean_angle, 1),
                "is_moving": bool(mean_vel >= 0.5),
            }
        except Exception as e:
            empty_res["error"] = str(e)
            return empty_res

    @staticmethod
    def _apply_clahe(crop: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if len(crop.shape) == 3:
                lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                lab[..., 0] = clahe.apply(lab[..., 0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return clahe.apply(crop)
        except Exception:
            return crop
