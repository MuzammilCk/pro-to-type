"""Diagnostic: why do enrolled faces fail recognition?

Loads the engine exactly like run.py does, then reports:
1. Which templates loaded from faces/
2. Template health (shape, norm, self-similarity)
3. Cross-person similarity matrix
4. identify() on samples/face_test.jpg with full scoring detail
"""
import os
import numpy as np

os.makedirs("faces", exist_ok=True)
from perception.face_engine import FaceEngine

fe = FaceEngine()

print("=" * 70)
print("STEP 1: Loaded templates")
print("=" * 70)
if not fe.known_faces:
    print("  !! NO TEMPLATES LOADED AT ALL — nothing to match against")
for name, templates in fe.known_faces.items():
    arr = np.array(templates)
    norms = np.linalg.norm(arr, axis=1)
    print(f"  {name}: {len(templates)} template(s), dim={arr.shape[1] if arr.ndim == 2 else '?'}, norms={np.round(norms, 4).tolist()}")

print()
print("=" * 70)
print("STEP 2: Self-similarity (template[0] vs all templates of same person)")
print("=" * 70)
for name, templates in fe.known_faces.items():
    if len(templates) < 2:
        print(f"  {name}: only 1 template — cannot measure spread")
        continue
    base = np.array(templates[0])
    sims = [float(np.dot(base, np.array(t))) for t in templates]
    print(f"  {name}: sims={[round(s, 3) for s in sims]}")
    print(f"    -> spread: intra-person similarity should be > 0.5; low values = mixed/corrupt templates")

print()
print("=" * 70)
print("STEP 3: Cross-person similarity (are different people too similar?)")
print("=" * 70)
names = list(fe.known_faces.keys())
for i, n1 in enumerate(names):
    for n2 in names[i + 1:]:
        sims = [float(np.dot(np.array(a), np.array(b)))
                for a in fe.known_faces[n1] for b in fe.known_faces[n2]]
        print(f"  {n1} vs {n2}: max={max(sims):.3f} (inter-person should be < 0.36)")

print()
print("=" * 70)
print("STEP 4: identify() on samples/face_test.jpg")
print("=" * 70)
import cv2

for img_path in ["samples/face_test.jpg", "samples/final_test.jpg"]:
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"  {img_path}: cannot read")
        continue
    faces = fe.detect(frame)
    print(f"  {img_path}: detected {len(faces)} face(s)")
    for face in faces:
        x, y, w, h = [int(v) for v in face[:4]]
        emb = fe._extract_aligned(frame, face)
        print(f"    face at ({x},{y},{w},{h}): embedding={'OK' if emb is not None else 'FAILED'}")
        if emb is not None:
            # Score against EVERY person's templates individually
            for name, templates in fe.known_faces.items():
                scores = [float(np.dot(emb, np.array(t))) for t in templates]
                best = max(scores) if scores else -1
                verdict = "MATCH" if best >= 0.36 else ("CLOSE" if best >= 0.28 else "NO MATCH")
                print(f"      vs {name:<14} best={best:.3f}  per-template={[round(s, 3) for s in scores]}  -> {verdict}")
print()
print("=" * 70)
print("STEP 5: Per-template contamination matrix (stranger-sim vs own-sim)")
print("=" * 70)
# A template that is MORE similar to a stranger than to its own person's
# other templates is mislabeled/corrupt and poisons max-score matching.
for name, templates in fe.known_faces.items():
    others = {n: t for n, t in fe.known_faces.items() if n != name}
    for i, t in enumerate(templates):
        ta = np.array(t)
        own = [float(np.dot(ta, np.array(u)))
               for j, u in enumerate(templates) if j != i]
        own_best = max(own) if own else None
        stranger_best, stranger_who = -1.0, ""
        for oname, otemps in others.items():
            for u in otemps:
                s = float(np.dot(ta, np.array(u)))
                if s > stranger_best:
                    stranger_best, stranger_who = s, oname
        flag = ""
        if own_best is not None and stranger_best > own_best + 0.05:
            flag = "  <-- CONTAMINATED (closer to stranger than to self)"
        print(f"  {name}[{i}]: own_best={own_best if own_best is None else round(own_best, 3)} "
              f"stranger_best={stranger_best:.3f} ({stranger_who}){flag}")
