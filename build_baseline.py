"""Build baseline-recognition.json — the Phase 0 deliverable.

Re-run this after re-enrollment (Phase 2/11) to produce an after-baseline and
prove the delta. Sections:
  per_person    template counts + intra-person similarity stats
  cross_person  max/mean cross-identity similarity (impostor signal)
  score_dist    identify() score distribution on any samples/*.jpg
"""
import json
import os
import glob

import numpy as np

os.makedirs("faces", exist_ok=True)
from face_engine import FaceEngine  # noqa: E402  (engine after dir creation)

OUT = "baseline-recognition.json"
GENUINE_KEEP = 0.40  # sims below this on genuine pairs are flagged as outliers


def intra_stats(similarities: list[float]) -> dict:
    if not similarities:
        return {"n": 0, "min": None, "median": None, "max": None, "outliers": 0}
    sims = sorted(similarities)
    n = len(sims)
    median = sims[n // 2] if n % 2 else (sims[n // 2 - 1] + sims[n // 2]) / 2.0
    return {
        "n": n,
        "min": round(min(sims), 4),
        "median": round(median, 4),
        "max": round(max(sims), 4),
        "outliers": sum(1 for s in sims if s < GENUINE_KEEP),
        "outlier_sims": [round(s, 4) for s in sims if s < GENUINE_KEEP],
    }


def main():
    fe = FaceEngine()
    baseline: dict = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "engine_threshold": fe.MATCH_THRESHOLD,
        "per_person": {},
        "cross_person": {},
        "score_dist": {"samples": []},
    }

    # ---- 1. Per-person template health + intra-person similarity ----
    for name, templates in sorted(fe.known_faces.items()):
        arr = np.array(templates)
        entry = {"template_count": len(templates), "dim": int(arr.shape[1] if arr.ndim == 2 else 0)}
        if len(templates) >= 2:
            # all unique template pairs (i<j)
            sims = []
            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    sims.append(float(np.dot(templates[i], templates[j])))
            entry["intra_similarity"] = intra_stats(sims)
        else:
            entry["intra_similarity"] = {"n": 0, "note": "needs >=2 templates"}
        baseline["per_person"][name] = entry

    # ---- 2. Cross-person (impostor) similarity ----
    names = sorted(fe.known_faces.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sims = [float(np.dot(x, y))
                    for x in fe.known_faces[a] for y in fe.known_faces[b]]
            baseline["cross_person"][f"{a}|{b}"] = {
                "n_pairs": len(sims),
                "max": round(max(sims), 4),
                "mean": round(sum(sims) / len(sims), 4),
            }

    # ---- 3. Score distribution on sample images (genuine/impostor labels) ----
    import cv2
    labels = {"muzammil": "muzammil", "mine": "muzammil"}  # sample file -> genuine identity
    for path in sorted(glob.glob("samples/*.jpg")):
        frame = cv2.imread(path)
        if frame is None:
            continue
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        for face in fe.detect(frame):
            x, y, w, h = [int(v) for v in face[:4]]
            _, score, meta = fe.identify(frame, face)
            baseline["score_dist"]["samples"].append({
                "file": os.path.basename(path),
                "genuine_identity": labels.get(stem, "unknown"),
                "best_score": round(float(score), 4),
                "best_identity": None,  # filled below via identify detail
                "threshold_used": meta.get("threshold"),
                "quality": meta.get("quality"),
            })
    for entry in baseline["score_dist"]["samples"]:
        if entry["best_identity"] is None:
            entry["best_identity"] = "see per-template detail in diag_recognition.py"

    with open(OUT, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Wrote {OUT}")
    for name, entry in baseline["per_person"].items():
        intra = entry.get("intra_similarity", {})
        print(f"  {name}: {entry['template_count']} templates, intra={intra.get('min')}-{intra.get('max')} "
              f"(median {intra.get('median')}), outliers={intra.get('outliers', 0)}")
    for pair, stats in baseline["cross_person"].items():
        print(f"  cross {pair}: max={stats['max']}, mean={stats['mean']}")


if __name__ == "__main__":
    main()
