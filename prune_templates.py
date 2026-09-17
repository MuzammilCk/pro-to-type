"""Prune contaminated face templates (data hygiene for recognition).

A template is contaminated when it is NOT more similar to its own person's
other templates than to some stranger's — i.e. it was captured from a bad
frame (motion blur, partial face, wrong crop during enrollment). With
max-score matching (face_engine.identify), one poisoned template can push
the WRONG person over the 0.36 threshold on marginal frames.

Usage:
    python prune_templates.py           # dry run: report only
    python prune_templates.py --apply   # actually rewrite faces/*.npy
"""
import os
import sys

import numpy as np

FACES_DIR = "faces"
# A template must cohere with its own person (own-sim >= OWN_MIN) AND be
# closer to its own cluster than to any stranger (margin >= STRANGER_MARGIN).
OWN_MIN = 0.45
STRANGER_MARGIN = 0.0


def load_all():
    data = {}
    for f in sorted(os.listdir(FACES_DIR)):
        if f.endswith(".npy"):
            arr = np.load(os.path.join(FACES_DIR, f)).astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            # L2-normalize: raw SFace features have norm ~300, so raw dot
            # products are meaningless. FaceEngine._load_known normalizes on
            # load; mirror that here so scores are true cosine similarities.
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1.0
            data[os.path.splitext(f)[0]] = arr / norms
    return data


def main():
    apply = "--apply" in sys.argv
    data = load_all()
    if not data:
        print("No enrolled faces found.")
        return

    pruned = {}
    for name, arr in data.items():
        keep = []
        others = {n: a for n, a in data.items() if n != name}
        for i in range(len(arr)):
            v = arr[i]
            own = [float(np.dot(v, arr[j]))
                   for j in range(len(arr)) if j != i]
            own_best = max(own) if own else None
            stranger_best = max(
                (float(np.dot(v, o[j]))
                 for o in others.values() for j in range(len(o))),
                default=-1.0,
            )
            bad_own = own_best is not None and own_best < OWN_MIN
            bad_stranger = stranger_best > own_best + STRANGER_MARGIN \
                if own_best is not None else False
            tag = ""
            if bad_own or bad_stranger:
                tag = ("  <-- DROP ("
                       + (f"own={own_best:.3f}<{OWN_MIN} " if bad_own else "")
                       + (f"stranger={stranger_best:.3f}" if bad_stranger else "")
                       + ")")
                pruned.setdefault(name, []).append(i)
            else:
                tag = f"  keep (own={own_best if own_best is None else round(own_best, 3)}, stranger={stranger_best:.3f})"
            print(f"  {name}[{i}]{tag}")

    if not pruned:
        print("\nAll templates clean — nothing to do.")
        return

    print(f"\nWould remove {sum(len(v) for v in pruned.values())} template(s): "
          + ", ".join(f"{n}{v}" for n, v in pruned.items()))
    if not apply:
        print("Dry run only — re-run with --apply to write changes.")
        return

    for name, idxs in pruned.items():
        kept = np.delete(data[name], idxs, axis=0)
        if len(kept) == 0:
            os.remove(os.path.join(FACES_DIR, f"{name}.npy"))
            print(f"  {name}: all templates removed — deleted {name}.npy "
                  "(re-enroll with: python enroll.py <name>)")
        else:
            np.save(os.path.join(FACES_DIR, f"{name}.npy"), kept)
            print(f"  {name}: kept {len(kept)}/{len(data[name])} templates")


if __name__ == "__main__":
    main()
