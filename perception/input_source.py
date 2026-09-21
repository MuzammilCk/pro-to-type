from abc import ABC, abstractmethod
import os
import cv2
import numpy as np


class InputSource(ABC):
    """Abstract base — every input implements read()/close()."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]: ...

    @abstractmethod
    def close(self) -> None: ...


class WebcamSource(InputSource):
    def __init__(self, device_id: int = 0):
        self._cap = cv2.VideoCapture(device_id)
        self._device_id = device_id

    @property
    def name(self) -> str:
        return f"Webcam {self._device_id}"

    def read(self):
        return self._cap.read()

    def close(self):
        self._cap.release()


class IPCameraSource(InputSource):
    """Reads an MJPEG/RTSP stream (e.g., phone via IP Webcam app).

    Connection is lazy — only connects when read() is first called,
    so registering multiple sources at startup won't fail if one
    is unreachable.
    """

    def __init__(self, url: str, name: str | None = None):
        self._url = url
        self._name = name or url
        self._cap = None
        self._connected = False

    @property
    def name(self) -> str:
        return self._name

    def _ensure_connected(self):
        if not self._connected:
            self._cap = cv2.VideoCapture(self._url)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._connected = True

    def read(self):
        if not self._url:
            return False, None
        self._ensure_connected()
        return self._cap.read()

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._connected = False


class FileSource(InputSource):
    """Reads frames from a video or image-sequence file."""

    def __init__(self, path: str):
        self._path = path
        is_image = os.path.splitext(path)[1].lower() in (".jpg", ".jpeg", ".png")
        if is_image:
            self._img = cv2.imread(path)
            self._is_image = True
        else:
            self._cap = cv2.VideoCapture(path)
            self._is_image = False

    @property
    def name(self) -> str:
        return f"File({os.path.basename(self._path)})"

    def read(self):
        if self._is_image:
            if self._img is not None:
                return True, self._img.copy()
            return False, None
        return self._cap.read()

    def close(self):
        if not self._is_image:
            self._cap.release()


class InputManager:
    """Manages hot-swappable input sources with a simple overlay UI."""

    BUTTONS = [
        ("1", "Webcam", "webcam"),
        ("2", "Phone", "phone"),
        ("3", "File", "file"),
        ("4", "IPCAM", "ipcam"),
    ]

    def __init__(self):
        self._sources: dict[str, InputSource] = {}
        self._active_key: str | None = None

    def register(self, key: str, source: InputSource):
        self._sources[key] = source

    def switch_to(self, key: str) -> bool:
        if key not in self._sources:
            return False
        if self._active_key is not None and self._active_key in self._sources:
            self._sources[self._active_key].close()
        self._active_key = key
        print(f"[InputManager] Switched to: {self._sources[key].name}")
        return True

    @property
    def active_source(self) -> InputSource | None:
        return self._sources.get(self._active_key) if self._active_key else None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.active_source is None:
            return False, None
        return self.active_source.read()

    def draw_controls(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        bar_h = 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        btn_w = w // len(self.BUTTONS)
        for i, (hotkey, label, src_key) in enumerate(self.BUTTONS):
            x0, x1 = i * btn_w, (i + 1) * btn_w
            is_active = self._active_key == src_key
            color = (0, 200, 0) if is_active else (80, 80, 80)
            cv2.rectangle(frame, (x0 + 8, 10), (x1 - 8, bar_h - 10), color, 2)
            cv2.putText(frame, f"[{hotkey}] {label}", (x0 + 20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Input: {self._active_key or 'NONE'}", (w - 220, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return frame

    def handle_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and y < 60:
            btn_w = param["frame_w"] // len(self.BUTTONS)
            for hotkey, label, src_key in self.BUTTONS:
                if x < (self.BUTTONS.index((hotkey, label, src_key)) + 1) * btn_w:
                    self.switch_to(src_key)
                    return
