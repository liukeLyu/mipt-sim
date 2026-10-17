"""Average final half-chain entropy over trajectories, for a grid of sizes L and rates p.

Usage: python scripts/sweep.py [--quick]
Writes results/sweep.json (mean, standard error, trajectory count per (L, p)).
"""

import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one core per worker process

import numpy as np

from pl_mipt import half_chain_entropy, run_trajectory

PS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
NTRAJ = {8: 800, 10: 600, 12: 400, 14: 300, 16: 200, 18: 100, 20: 40}
if "--quick" in sys.argv:
    NTRAJ = {8: 50, 10: 50, 12: 30}


def one(args):
    # average two neighbouring cut positions: at strong measurement it matters whether
    # the cut bond was just acted on by the last brickwork layer or skipped by it
    L, p, seed = args
    psi = run_trajectory(L, p, np.random.default_rng(seed))
    return 0.5 * (half_chain_entropy(psi, 0) + half_chain_entropy(psi, 1))


if __name__ == "__main__":
    out = []
    seq = np.random.SeedSequence(2026)
    with Pool() as pool:
        for L, n in NTRAJ.items():
            for p in PS:
                t = time.time()
                seeds = [int(s.generate_state(1)[0]) for s in seq.spawn(n)]
                s = np.array(pool.map(one, [(L, p, x) for x in seeds]))
                out.append({"L": L, "p": p, "n": n, "mean": s.mean(), "sem": s.std(ddof=1) / np.sqrt(n)})
                print(f"L={L:2d} p={p:.2f} S={s.mean():.3f}±{out[-1]['sem']:.3f} ({time.time() - t:.0f}s)", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/sweep.json", "w") as f:
        json.dump(out, f, indent=1)
