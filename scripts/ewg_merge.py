"""Merge sample files from scripts/ewg_run.py (e.g. runs on different machines) into one.

Usage: python scripts/ewg_merge.py out.npz in1.npz in2.npz ...
Duplicate seeds are dropped, so overlapping runs are safe to merge.
"""

import sys

import numpy as np

KEYS = ["records", "seeds", "negativity", "log_negativity"]

out, inputs = sys.argv[1], sys.argv[2:]
parts = [np.load(f) for f in inputs]
for f, d in zip(inputs[1:], parts[1:]):
    for k in ["L", "p", "A", "B"]:
        if not np.array_equal(d[k], parts[0][k]):
            raise ValueError(f"{f} has a different {k}")
merged = {k: np.concatenate([d[k] for d in parts]) for k in KEYS}
_, keep = np.unique(merged["seeds"], return_index=True)
merged = {k: v[keep] for k, v in merged.items()}
meta = {k: parts[0][k] for k in ["L", "p", "A", "B"]}
np.savez_compressed(out, **merged, **meta)
print(f"{len(keep)} unique trajectories, {(merged['negativity'] > 1e-10).sum()} with negativity > 0 -> {out}")
