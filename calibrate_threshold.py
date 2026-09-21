"""Threshold calibration from measured FAR/FRR — no more guessed 0.36.

Builds pair datasets from enrolled templates + sample images:
  genuine  pairs  — same-identity embeddings (per-identity intra-pairs,
                    plus sample-vs-own-identity-template pairs)
  impostor pairs  — different-identity embeddings (all cross pairs,
                    plus sample-vs-other-identity pairs)

Sweeps candidate thresholds, measures False Accept Rate / False Reject Rate
at each point, and reports the operating curve with:
  - the EER point (FAR == FRR)
  - FAR-first recommendation: the LEAST strict threshold whose measured
    FAR is 0 on the impostor set (security-sensitive authorization
    prioritizes low false acceptance over convenience)

Usage:
    python calibrate_threshold.py                 # report only
    python calibrate_threshold.py --json out.json # also dump the curve

The selected value is written to README/config as ARIA_RECOGNITION_THRESHOLD.
"""
import argparse
import itertools
import json
import os

import numpy as np

from perception.face_engine import FaceEngine, FACES_DIR


def load_templates():
    """L2-normalized templates per identity (mirrors _load_known)."""
    data = {}
    for f in sorted(os.listdir(FACES_DIR)):
        if not f.endswith(".npy"):
            continue
        arr = np.load(os.path.join(FACES_DIR, f)).astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        data[os.path.splitext(f)[0]] = arr / norms
    return data


def load_probe_samples(engine: FaceEngine):
    """Embed every sample image; each yields (identity_or_None, embedding).

    Samples whose filename matches an enrolled identity (case-insensitive
    substring) count as genuine probes; others are impostor probes.
    """
    probes = []
    if not os.path.isdir("samples"):
        return probes
    names_lower = {n.lower(): n for n in os.listdir(FACES_DIR)
                   if n.endswith(".npy")}
    for f in sorted(os.listdir("samples")):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        frame = cv2_imread(os.path.join("samples", f))
        if frame is None:
            continue
        faces = engine.detect(frame)
        if len(faces) == 0:
            continue
        face = max(faces, key=lambda fc: fc[2] * fc[3])
        emb = engine._extract_aligned(frame, face)
        if emb is None:
            continue
        who = None
        stem = os.path.splitext(f)[0].lower()
        for low, real in names_lower.items():
            if low in stem:
                who = real
                break
        probes.append((who, emb))
    return probes


def cv2_imread(path):
    import cv2
    return cv2.imread(path)


def build_pairs(templates: dict, probes):
    """Return (genuine_scores, impostor_scores)."""
    genuine, impostor = [], []

    # Intra-template pairs per identity
    for name, arr in templates.items():
        for i, j in itertools.combinations(range(len(arr)), 2):
            genuine.append(float(np.dot(arr[i], arr[j])))

    # Cross-template pairs
    names = sorted(templates.keys())
    for a, b in itertools.combinations(names, 2):
        for u in templates[a]:
            for v in templates[b]:
                impostor.append(float(np.dot(u, v)))

    # Probe-vs-template pairs
    for who, emb in probes:
        for name, arr in templates.items():
            best = max(float(np.dot(emb, t)) for t in arr)
            if who == name:
                genuine.append(best)
            else:
                impostor.append(best)

    return genuine, impostor


def evaluate(genuine, impostor, thresholds):
    """FAR/FRR per threshold + EER + FAR-zero recommendation."""
    g = np.asarray(genuine, dtype=np.float64)
    imp = np.asarray(impostor, dtype=np.float64)
    rows = []
    for t in thresholds:
        frr = float((g < t).mean()) if g.size else 0.0
        far = float((imp >= t).mean()) if imp.size else 0.0
        rows.append({"threshold": round(float(t), 3),
                     "far": round(far, 4), "frr": round(frr, 4)})

    # EER: closest far/frr crossing
    eer, eer_t = None, None
    best = None
    for r in rows:
        d = abs(r["far"] - r["frr"])
        if best is None or d < best:
            best = d
            eer, eer_t = round((r["far"] + r["frr"]) / 2, 4), r["threshold"]

    # Security-first pick: least strict threshold with FAR == 0 (and at
    # least some impostor evidence measured); fall back to EER threshold.
    far_zero = [r for r in rows if imp.size and r["far"] == 0.0]
    rec = max((r["threshold"] for r in far_zero), default=eer_t)
    return rows, eer, eer_t, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="dump the full curve to a JSON file")
    args = ap.parse_args()

    engine = FaceEngine()
    templates = load_templates()
    if not templates:
        print("No enrolled templates in faces/ — nothing to calibrate.")
        return
    probes = load_probe_samples(engine)
    genuine, impostor = build_pairs(templates, probes)
    if not genuine or not impostor:
        print("Not enough pairs to calibrate "
              f"(genuine={len(genuine)}, impostor={len(impostor)}). "
              "Enroll at least 2 identities with 2+ templates each.")
        return

    g, i = np.asarray(genuine), np.asarray(impostor)
    print(f"Pairs: genuine={len(g)}  impostor={len(i)}")
    print(f"Genuine : min={g.min():.3f} median={np.median(g):.3f} max={g.max():.3f}")
    print(f"Impostor: min={i.min():.3f} median={np.median(i):.3f} max={i.max():.3f}")

    # Dataset sanity: genuinely different people never score ~1.0. An
    # impostor pair at ~1.0 means duplicate-template enrollment (the old
    # "same frame x5" artifact) or a mislabeled probe — the curve is not
    # trustworthy until the data is fixed.
    contaminated = i.size and i.max() > 0.90
    if contaminated:
        print("\n!! CONTAMINATED IMPOSTOR SET: an impostor pair scored "
              f"{i.max():.3f}.")
        print("!! Likely causes: duplicate-template enrollment (re-enroll that "
              "identity with independent samples) or probe images that are "
              "copies of enrolled templates. Re-enroll, then rerun.")
        print("!! Recommendation below is NOT trustworthy for this dataset.")

    # Sweep: fine grid around the empirical separation zone, coarse outside
    lo = max(0.0, min(g.min(), i.max()) - 0.10)
    hi = 1.0
    grid = np.unique(np.concatenate([
        np.arange(0.10, 0.80, 0.01),
        np.arange(max(0.0, lo - 0.05), min(0.95, lo + 0.25), 0.005),
    ]))
    rows, eer, eer_t, rec = evaluate(genuine, impostor, grid)

    print("\n threshold |  FAR   |  FRR")
    print(" -" * 22)
    shown = {0.30, 0.35, 0.36, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65}
    for r in rows:
        if r["threshold"] in shown or r["threshold"] in (eer_t, rec):
            mark = ""
            if r["threshold"] == eer_t:
                mark += "  <- EER"
            if r["threshold"] == rec:
                mark += "  <- RECOMMENDED (FAR=0)"
            print(f"    {r['threshold']:.3f}   | {r['far']:.4f} | {r['frr']:.4f}{mark}")

    print(f"\nEER: {eer:.4f} at threshold={eer_t}")
    if contaminated:
        print("RECOMMENDATION WITHHELD: fix the impostor set first "
              "(re-enroll duplicate-template identities with independent samples,")
        print("then rerun this tool). Keep the current default until then.")
    else:
        print(f"RECOMMENDED (security-first, measured FAR=0): {rec}")
    print("\nAdopt with:  set ARIA_RECOGNITION_THRESHOLD=<value>  (or edit the")
    print("default in face_engine.py). Rerun after every enrollment change.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"genuine": [round(x, 4) for x in genuine],
                       "impostor": [round(x, 4) for x in impostor],
                       "curve": rows, "eer": eer, "eer_threshold": eer_t,
                       "recommended": rec}, f, indent=2)
        print(f"Curve written: {args.json}")


if __name__ == "__main__":
    main()
