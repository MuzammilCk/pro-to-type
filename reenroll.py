"""Re-enroll a face from webcam or file source.

Phase 2 contract: templates come from INDEPENDENT samples. Re-embedding one
frame N times is duplicate detection's job to refuse — the old
"same frame x5" loop is gone.

Usage:
    python reenroll.py <name>                        # capture from webcam
    python reenroll.py <name> --file                 # samples/face_test.jpg (weak, --force)
    python reenroll.py <name> --file img1.jpg ...    # N independent photos
"""
import os
import sys
import cv2
import numpy as np

from perception.face_engine import FaceEngine
from enroll import _collect_webcam_samples, _report


def main():
    if len(sys.argv) < 2:
        print("Usage: python reenroll.py <name> [--file [photos...]] [--force]")
        sys.exit(1)

    name = sys.argv[1]
    use_file = "--file" in sys.argv
    force = "--force" in sys.argv

    fe = FaceEngine()

    if use_file:
        paths = [a for a in sys.argv[2:] if not a.startswith("--")]
        if not paths:
            paths = ["samples/face_test.jpg"]
        if len(paths) == 1 and not force:
            print("A single photo is not independent evidence (the old code")
            print("copied it 5 times — that is exactly what this rebuild removes).")
            print("Provide >= 3 photos, or add --force for a weak single template.")
            sys.exit(1)
        samples = []
        for p in paths:
            frame = cv2.imread(p)
            if frame is None:
                print(f"Cannot read image: {p}")
                continue
            faces = fe.detect(frame)
            if len(faces) == 0:
                print(f"No face detected in {p}.")
                continue
            samples.append((frame, max(faces, key=lambda f: f[2] * f[3])))
        min_needed = 1 if (force and len(samples) == 1) else 3
        res = fe.enroll_identity(name, samples, min_samples=min_needed)
    else:
        samples = _collect_webcam_samples(fe, name)
        if not samples:
            print("No frame captured.")
            return
        res = fe.enroll_identity(name, samples)

    _report(res, name)

    if res["enrolled"]:
        # Post-enrollment verification round-trip
        frame, face = samples[0]
        name2, score, meta = fe.identify(frame, face)
        print(f"  Verification: {name2 or 'REJECTED'} "
              f"(score={score:.3f}, quality={meta.get('quality', 0):.2f}, "
              f"thresh={meta.get('threshold', 0):.3f}, reason={meta.get('reason')})")


if __name__ == "__main__":
    main()
