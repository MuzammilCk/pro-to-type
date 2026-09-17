"""Fetch ARIA's model files into models/ (gitignored — not shipped in git).

Run once after a fresh clone:
    python setup_models.py            # skips files already present
    python setup_models.py --force    # re-download everything

Sources (all verified reachable):
  face_detection_yunet_2023mar.onnx    OpenCV Zoo — face detection (~232 KB)
  face_recognition_sface_2021dec.onnx  OpenCV Zoo — recognition (~37 MB)
  yolov5s.onnx                         Ultralytics v7.0 release (~28 MB)
  face_landmarker.task                 Google MediaPipe host (~3.7 MB)
"""
import os
import sys
import urllib.request

MODELS_DIR = "models"

SOURCES = {
    "face_detection_yunet_2023mar.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ],
    "face_recognition_sface_2021dec.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ],
    "yolov5s.onnx": [
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx",
        "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx",
    ],
    "face_landmarker.task": [
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    ],
}


def fetch(name: str, urls: list[str], force: bool = False) -> bool:
    dest = os.path.join(MODELS_DIR, name)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0 and not force:
        print(f"  [skip] {name} already present")
        return True
    for url in urls:
        try:
            print(f"  [get ] {name}")
            tmp = dest + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dest)  # atomic-ish: no half-written .onnx left behind
            print(f"  [ok  ] {name} ({os.path.getsize(dest):,} bytes)")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {url}\n         {e}")
    return False


def main():
    force = "--force" in sys.argv
    os.makedirs(MODELS_DIR, exist_ok=True)
    failed = []
    for name, urls in SOURCES.items():
        if not fetch(name, urls, force):
            failed.append(name)
    if failed:
        print("\nFAILED to fetch: " + ", ".join(failed))
        print("Download them manually into models/ (see README).")
        sys.exit(1)
    print("\nAll models ready.")


if __name__ == "__main__":
    main()
