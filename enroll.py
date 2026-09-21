import sys
import os
import numpy as np
import cv2

from perception.face_engine import FaceEngine, FACES_DIR, ENROLL_MIN_SAMPLES


def _collect_webcam_samples(fe: FaceEngine, name: str, target: int = 5):
    """Capture (frame, face) pairs from INDEPENDENT moments.

    Each SPACE press embeds the live frame as one sample; validation
    (quality, duplicates, coherence) happens centrally in enroll_identity().
    """
    samples = []
    cap = cv2.VideoCapture(0)
    print(f"=== Enrolling '{name}' ===")
    print(f"Capture {target} photos from slightly different angles. Press SPACE for each.")
    print("Q = quit early.")
    while len(samples) < target:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Capture {len(samples)+1}/{target} — press SPACE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Name: {name}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Enrollment", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            faces = fe.detect(frame)
            if len(faces) == 1:
                samples.append((frame, faces[0]))
                print(f"  captured {len(samples)}/{target}")
            elif len(faces) == 0:
                print("  no face detected — try again")
            else:
                print(f"  {len(faces)} faces — only 1 person in frame please")
    cap.release()
    cv2.destroyAllWindows()
    return samples


def enroll_interactive(name: str):
    fe = FaceEngine()
    samples = _collect_webcam_samples(fe, name)
    if not samples:
        print("Nothing captured.")
        return

    res = fe.enroll_identity(name, samples)
    _report(res, name)


def enroll_from_files(name: str, photo_paths: list[str], force: bool = False):
    """Enroll from photo files — each file is ONE independent sample.

    Multiple distinct photos are validated like any other enrollment.
    A single photo cannot provide independent evidence: it is refused
    unless --force is given (stores a weak 1-template identity).
    """
    fe = FaceEngine()
    samples = []
    for path in photo_paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"Cannot read image: {path}")
            continue
        faces = fe.detect(frame)
        if len(faces) == 0:
            print(f"No face detected in {path}.")
            continue
        if len(faces) > 1:
            print(f"Multiple faces in {path} — using the largest.")
        # Largest face = intended subject
        face = max(faces, key=lambda f: f[2] * f[3])
        samples.append((frame, face))

    if len(samples) == 1 and not force:
        print("A single photo is not independent evidence for enrollment.")
        print("Provide >= 3 distinct photos, or re-run with --force to store")
        print("a weak single-template identity (not recommended).")
        return
    min_needed = 1 if force else ENROLL_MIN_SAMPLES
    res = fe.enroll_identity(name, samples, min_samples=min_needed)
    _report(res, name)


def _report(res: dict, name: str):
    if res["enrolled"]:
        print(f"Enrolled '{name}' with {res['templates']} validated templates "
              f"(median intra-similarity {res['median_intra']}).")
        print(f"Saved: {os.path.join(FACES_DIR, name + '.npy')}")
    else:
        print(f"Enrollment REJECTED: {res['reason']}")
        print(f"  samples={res['n_samples']} accepted={res['accepted']} "
              f"duplicates={res['rejected_duplicate']} "
              f"low_quality={res['rejected_low_quality']} "
              f"invalid={res['rejected_invalid']}")


def list_enrolled():
    if not os.path.isdir(FACES_DIR):
        print("No faces enrolled yet.")
        return
    files = sorted(f for f in os.listdir(FACES_DIR) if f.endswith(".npy"))
    if not files:
        print("No faces enrolled yet.")
        return
    print("Enrolled faces:")
    for f in files:
        name = os.path.splitext(f)[0]
        templates = np.load(os.path.join(FACES_DIR, f))
        print(f"  {name} ({len(templates)} templates)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv
    if len(args) < 1 or args[0] in ("-l", "list"):
        list_enrolled()
    elif len(args) == 1:
        enroll_interactive(args[0])
    elif len(args) >= 2:
        enroll_from_files(args[0], args[1:], force=force)
    else:
        print("Usage:")
        print("  python enroll.py <name>                      # multi-shot webcam enrollment (5 photos)")
        print("  python enroll.py <name> <photo> [photo...]   # multi-photo enrollment")
        print("  python enroll.py <name> <photo> --force      # weak single-photo enrollment")
        print("  python enroll.py list                        # list enrolled faces")
