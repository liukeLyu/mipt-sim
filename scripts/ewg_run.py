"""Sample MIPT trajectories and save each one's measurement record and pair weights.

Default: L = 18, p = 0.17, regions A = qubits (1, 2) and B = (8, 9) in 1-based labels.
Usage: python scripts/ewg_run.py [n_samples] [out.npz]
Saves every chunk, so a partial run is usable.
"""

import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from mipt_sim.ewg import sample

L, P = 18, 0.17
A, B = [0, 1], [7, 8]  # 0-based
KEYS = ["negativity", "log_negativity", "mutual_information"]


def one(seed):
    return sample(L, P, A, B, seed)


if __name__ == "__main__":
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    out = sys.argv[2] if len(sys.argv) > 2 else f"results/ewg_L{L}_p{P}.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    records, weights, seeds = [], {k: [] for k in KEYS}, []
    chunk, t0 = 1000, time.time()
    with Pool() as pool:
        for start in range(0, total, chunk):
            batch = list(range(start, min(start + chunk, total)))
            for (r, w), s in zip(pool.map(one, batch), batch):
                records.append(r)
                seeds.append(s)
                for k in KEYS:
                    weights[k].append(w[k])
            neg = np.array(weights["negativity"])
            np.savez_compressed(
                out,
                records=np.array(records),
                seeds=np.array(seeds),
                L=L,
                p=P,
                A=A,
                B=B,
                **{k: np.array(v) for k, v in weights.items()},
            )
            print(
                f"{len(seeds)} samples, {(neg > 1e-10).sum()} with negativity > 0, {time.time() - t0:.0f}s", flush=True
            )
