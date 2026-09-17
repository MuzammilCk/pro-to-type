import sys
import os
import numpy as np
import cv2

os.makedirs("faces", exist_ok=True)

from face_engine import FaceEngine, FACES_DIR


face_engine = FaceEngine()


def enroll_interactive(name: str):
    cap = cv2.VideoCapture(0)
    print(f"=== Enrolling '{name}' ===")
    print("Capture 5 photos from slightly different angles. Press SPACE for each.")
    print("Q = quit early. Minimum 3 recommended.")
    embeddings = []
    while len(embeddings) < 5:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Capture {len(embeddings)+1}/5 — press SPACE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Name: {name}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Enrollment", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            faces = face_engine.detect(frame)
            if len(faces) == 1:
                emb = face_engine.embed(frame, faces[0])
                embeddings.append(emb)
                print(f"  captured {len(embeddings)}/5 (quality OK)")
            elif len(faces) == 0:
                print("  no face detected — try again")
            else:
                print(f"  {len(faces)} faces — only 1 person in frame please")

    cap.release()
    cv2.destroyAllWindows()
    if len(embeddings) < 3:
        print(f"Need at least 3 good captures (got {len(embeddings)}). Try again.")
        return
    face_engine.known_faces[name] = embeddings
    np.save(os.path.join(FACES_DIR, f"{name}.npy"), np.array(embeddings))
    print(f"Enrolled '{name}' with {len(embeddings)} templates. Ready for recognition.")


def enroll_from_file(name: str, photo_path: str):
    frame = cv2.imread(photo_path)
    if frame is None:
        print(f"Cannot read image: {photo_path}")
        return
    faces = face_engine.detect(frame)
    if len(faces) == 0:
        print("No face detected in photo.")
        return
    if len(faces) > 1:
        print(f"Multiple faces ({len(faces)}) — using the first.")
    emb = face_engine.embed(frame, faces[0])
    face_engine.known_faces[name] = [emb]
    np.save(os.path.join(FACES_DIR, f"{name}.npy"), np.array([emb]))
    print(f"Enrolled '{name}' from {photo_path}")


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
    if len(sys.argv) < 2 or sys.argv[1] in ("-l", "list"):
        list_enrolled()
    elif len(sys.argv) == 2:
        enroll_interactive(sys.argv[1])
    elif len(sys.argv) == 3:
        enroll_from_file(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print("  python enroll.py <name>            # multi-shot webcam enrollment (5 photos)")
        print("  python enroll.py <name> <photo>    # single-photo enrollment")
        print("  python enroll.py list              # list enrolled faces")
