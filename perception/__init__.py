"""Perception package — computer vision, face detection/recognition, tracking, and presence."""
from .detector import Detector, Detection
from .face_engine import FaceEngine
from .face_pattern import FacePatternEngine, FacePatternEngine as FacePattern
from .tracker import FaceTracker
from .presence import PresenceManager
from .input_source import InputSource, WebcamSource, IPCameraSource, FileSource, InputManager
from .inspection import VisionInspectionEngine

__all__ = [
    "Detector",
    "Detection",
    "FaceEngine",
    "FacePattern",
    "FacePatternEngine",
    "FaceTracker",
    "PresenceManager",
    "InputSource",
    "WebcamSource",
    "IPCameraSource",
    "FileSource",
    "InputManager",
    "VisionInspectionEngine",
]

