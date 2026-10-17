"""Average entanglement entropy S(l) of l consecutive qubits, l = 1 ... L/2, at fixed L, for several p.

Usage: python scripts/subregion.py [L] [trajectories per p]
Writes results/subregion_L{L}.json (mean and standard error of S for every l and p).
"""

import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one core per worker process

import numpy as np

from mipt_sim import interval_entropy, run_trajectory

PS = [0.0, 0.05, 0.1, 0.17, 0.3, 0.5, 1.0]


def one(args):
    # average the two starting sites: brickwork bonds alternate, so regions starting on
    # even and odd sites see the last layer differently
    L, p, seed = args
    psi = run_trajectory(L, p, np.random.default_rng(seed))
    return [np.mean([interval_entropy(psi, ell, s) for s in (0, 1)]) for ell in range(1, L // 2 + 1)]


if __name__ == "__main__":
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    out = {"L": L, "n": n, "ell": list(range(1, L // 2 + 1)), "p": PS, "mean": [], "sem": []}
    with Pool() as pool:
        for i, p in enumerate(PS):
            t = time.time()
            s = np.array(pool.map(one, [(L, p, 1000 * i + k) for k in range(n)]))
            out["mean"].append(s.mean(0).tolist())
            out["sem"].append((s.std(0, ddof=1) / np.sqrt(n)).tolist())
            print(f"p={p:.2f} S(L/2)={s[:, -1].mean():.3f} ({time.time() - t:.0f}s)", flush=True)
    os.makedirs("results", exist_ok=True)
    with open(f"results/subregion_L{L}.json", "w") as f:
        json.dump(out, f, indent=1)
