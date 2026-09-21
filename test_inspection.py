"""Unit tests for Active OpenCV 5 Micro-Inspection Engine (Phase 9).

Validates:
1. crop_roi: bounds clamping, out-of-bound coordinates, safe empty fallbacks, CLAHE.
2. inspect_color_hsv: accurate HSV segmentation, morphology cleanup, metrics calculation.
3. analyze_geometry: Canny contours, aspect ratio, solidity, Laplacian sharpness.
4. measure_optical_flow: Farneback velocity extraction, displacement detection.
5. CPU performance budget: operations execute in under 5 ms on CPU.
"""
import time
import numpy as np
import cv2
import pytest

from perception.inspection import VisionInspectionEngine


class TestCropRoi:
    def test_crop_valid_roi(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:200, 150:350] = 255  # 100h x 200w region

        crop = VisionInspectionEngine.crop_roi(frame, (150, 100, 200, 100))
        assert crop.shape == (100, 200, 3)
        assert np.all(crop == 255)

    def test_crop_boundary_clamping(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Bbox extends beyond right and bottom edges
        crop = VisionInspectionEngine.crop_roi(frame, (600, 400, 100, 150))
        # Clamped: x: 600->640 (w=40), y: 400->480 (h=80)
        assert crop.shape == (80, 40, 3)

        # Bbox starts with negative coordinates
        crop_neg = VisionInspectionEngine.crop_roi(frame, (-50, -30, 100, 100))
        # Clamped: x: 0->50, y: 0->70
        assert crop_neg.shape == (70, 50, 3)

    def test_crop_out_of_bounds_empty(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Fully outside bounds
        crop = VisionInspectionEngine.crop_roi(frame, (700, 500, 100, 100))
        assert crop.shape == (0, 0, 3)

    def test_crop_invalid_inputs_safe(self):
        valid_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # None inputs
        assert VisionInspectionEngine.crop_roi(None, (10, 10, 20, 20)).shape == (0, 0, 3)
        assert VisionInspectionEngine.crop_roi(valid_frame, None).shape == (0, 0, 3)

        # Zero or negative width/height
        assert VisionInspectionEngine.crop_roi(valid_frame, (10, 10, 0, 20)).shape == (0, 0, 3)
        assert VisionInspectionEngine.crop_roi(valid_frame, (10, 10, 20, -5)).shape == (0, 0, 3)

        # Malformed bbox
        assert VisionInspectionEngine.crop_roi(valid_frame, [10, 20]).shape == (0, 0, 3)

    def test_crop_with_clahe_enhancement(self):
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        crop = VisionInspectionEngine.crop_roi(frame, (10, 10, 50, 50), enhance=True)
        assert crop.shape == (50, 50, 3)
        assert crop.dtype == np.uint8


class TestColorHsv:
    def test_positive_color_detection(self):
        # Create 100x100 BGR image with a 50x50 pure green square (25% coverage)
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        roi[25:75, 25:75] = [0, 255, 0]  # Green in BGR

        # Green HSV bounds
        lower_green = [35, 100, 100]
        upper_green = [85, 255, 255]

        res = VisionInspectionEngine.inspect_color_hsv(roi, lower_green, upper_green)
        assert res["detected"] is True
        assert 0.23 <= res["coverage_ratio"] <= 0.27
        assert res["matched_pixels"] == 2500
        assert res["total_pixels"] == 10000
        # Dominant hue should be approx 60 (standard OpenCV green)
        assert 50 <= res["dominant_hsv"][0] <= 70

    def test_negative_color_detection(self):
        # Pure blue image searching for green
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        roi[:] = [255, 0, 0]  # Blue in BGR

        lower_green = [35, 100, 100]
        upper_green = [85, 255, 255]

        res = VisionInspectionEngine.inspect_color_hsv(roi, lower_green, upper_green)
        assert res["detected"] is False
        assert res["coverage_ratio"] == 0.0
        assert res["matched_pixels"] == 0

    def test_empty_and_invalid_hsv_safe(self):
        res = VisionInspectionEngine.inspect_color_hsv(None, [0, 0, 0], [255, 255, 255])
        assert res["detected"] is False
        assert res["coverage_ratio"] == 0.0

        empty_img = np.empty((0, 0, 3), dtype=np.uint8)
        res_empty = VisionInspectionEngine.inspect_color_hsv(empty_img, [0, 0, 0], [255, 255, 255])
        assert res_empty["detected"] is False


class TestGeometryAnalysis:
    def test_analyze_rectangle_geometry(self):
        # Create black image with a 100x50 white rectangle (aspect ratio = 2.0)
        roi = np.zeros((200, 200, 3), dtype=np.uint8)
        roi[75:125, 50:150] = 255  # h=50, w=100

        res = VisionInspectionEngine.analyze_geometry(roi)
        assert res["has_structure"] is True
        assert res["contour_count"] >= 1
        assert res["max_area"] > 4000  # Theoretical area ~5000
        # Aspect ratio ~ 2.0
        assert 1.8 <= res["aspect_ratio"] <= 2.2
        # Rectangular solidity close to 1.0
        assert 0.85 <= res["solidity"] <= 1.0
        assert res["sharpness"] > 0.0

    def test_empty_geometry_safe(self):
        res_none = VisionInspectionEngine.analyze_geometry(None)
        assert res_none["has_structure"] is False
        assert res_none["contour_count"] == 0

        res_flat = VisionInspectionEngine.analyze_geometry(np.zeros((1, 1, 3), dtype=np.uint8))
        assert res_flat["has_structure"] is False


class TestOpticalFlow:
    def test_optical_flow_motion_detection(self):
        # Frame 1: Circle at (50, 50)
        f1 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(f1, (50, 50), 15, (255, 255, 255), -1)

        # Frame 2: Circle moved to (55, 50) - 5px horizontal shift
        f2 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(f2, (55, 50), 15, (255, 255, 255), -1)

        res = VisionInspectionEngine.measure_optical_flow(f1, f2)
        assert res["is_moving"] is True
        assert res["mean_velocity"] > 0.2

    def test_optical_flow_static_scene(self):
        f1 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(f1, (50, 50), 15, (255, 255, 255), -1)

        res = VisionInspectionEngine.measure_optical_flow(f1, f1)
        assert res["is_moving"] is False
        assert res["mean_velocity"] < 0.1

    def test_optical_flow_invalid_inputs_safe(self):
        res = VisionInspectionEngine.measure_optical_flow(None, None)
        assert res["is_moving"] is False
        assert res["mean_velocity"] == 0.0


class TestPerformanceBudget:
    def test_cpu_latency_under_5ms(self):
        """Verify each micro-tool executes well within the 5ms CPU budget."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        roi = frame[100:250, 100:250].copy()  # 150x150 typical inspection crop
        roi_next = frame[102:252, 102:252].copy()

        # 1. Benchmark crop_roi
        t0 = time.perf_counter()
        for _ in range(50):
            _ = VisionInspectionEngine.crop_roi(frame, (100, 100, 150, 150))
        dt_crop = (time.perf_counter() - t0) / 50 * 1000  # ms
        assert dt_crop < 1.0, f"crop_roi too slow: {dt_crop:.3f}ms (budget 1ms)"

        # 2. Benchmark inspect_color_hsv
        t0 = time.perf_counter()
        for _ in range(50):
            _ = VisionInspectionEngine.inspect_color_hsv(roi, [35, 50, 50], [85, 255, 255])
        dt_hsv = (time.perf_counter() - t0) / 50 * 1000  # ms
        assert dt_hsv < 5.0, f"inspect_color_hsv too slow: {dt_hsv:.3f}ms (budget 5ms)"

        # 3. Benchmark analyze_geometry
        t0 = time.perf_counter()
        for _ in range(50):
            _ = VisionInspectionEngine.analyze_geometry(roi)
        dt_geom = (time.perf_counter() - t0) / 50 * 1000  # ms
        assert dt_geom < 5.0, f"analyze_geometry too slow: {dt_geom:.3f}ms (budget 5ms)"

        # 4. Benchmark measure_optical_flow
        t0 = time.perf_counter()
        for _ in range(20):
            _ = VisionInspectionEngine.measure_optical_flow(roi, roi_next)
        dt_flow = (time.perf_counter() - t0) / 20 * 1000  # ms
        assert dt_flow < 5.0, f"measure_optical_flow too slow: {dt_flow:.3f}ms (budget 5ms)"
