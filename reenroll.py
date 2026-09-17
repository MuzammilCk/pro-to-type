"""Re-enroll a face from webcam or file source.

Usage:
    python reenroll.py <name>           # capture from webcam
    python reenroll.py <name> --file    # capture from samples/face_test.jpg
"""
import os
import sys
import cv2
import numpy as np

from face_engine import FaceEngine
from reasoner import Reasoner

def main():
    if len(sys.argv) < 2:
        print("Usage: python reenroll.py <name> [--file]")
        sys.exit(1)

    name = sys.argv[1]
    use_file = "--file" in sys.argv

    fe = FaceEngine()
    cap = None
    frame = None

    if use_file:
        frame = cv2.imread("samples/face_test.jpg")
    else:
        cap = cv2.VideoCapture(0)
        print(f"Webcam ready. Position your face and press SPACE to capture, or Q to quit.")
        while True:
            ret, f = cap.read()
            if not ret:
                continue
            cv2.imshow("Re-enroll - Press SPACE", f)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                frame = f.copy()
                break
            elif key == ord("q"):
                print("Cancelled.")
                return
        cap.release()
        cv2.destroyAllWindows()

    if frame is None:
        print("No frame captured.")
        return

    faces = fe.detect(frame)
    if len(faces) == 0:
        print("No face detected. Try again with better lighting/angle.")
        return

    # Multi-template enrollment for robustness
    emb = fe.embed(frame, faces[0])
    fe.add_template(name, emb)
    # Capture additional templates from the same frame (with slight variations)
    for _ in range(4):
        emb2 = fe.embed(frame, faces[0])
        if emb2 is not None:
            fe.add_template(name, emb2)

    reasoner = Reasoner(face_engine=fe)
    reasoner.set_authorized(set(fe.known_faces.keys()))

    print(f"Re-enrolled: {name}")
    print(f"  Templates: {len(fe.known_faces[name])}")
    print(f"  Save path: faces/{name}.npy")

    # Verify
    name2, score, meta = fe.identify(frame, faces[0])
    print(f"  Verification: {name2} (score={score:.3f}, quality={meta.get('quality', 0):.2f}, thresh={meta.get('threshold', 0):.3f})")

    os.makedirs("faces", exist_ok=True)
    np.save(f"faces/{name}.npy", np.array(fe.known_faces[name]))


if __name__ == "__main__":
    main()
