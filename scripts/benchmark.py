"""Time one trajectory (depth 2n) and one brickwork layer for each implementation."""

import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from matlab_port import rand_unitary_mi_state

from pl_mipt import pennylane_sim, run_trajectory

p = 0.17
rng = np.random.default_rng(1)
print(
    f"{'n':>3} {'pennylane_sim':>17} {'numpy_sim':>16} {'MATLAB port':>12}   (seconds per trajectory; per layer = /2n)"
)
for n in [int(a) for a in sys.argv[1:]] or [8, 12, 16, 18, 20, 22]:
    reps = max(1, int(2 ** (18 - n)))
    row = []
    for f in (
        lambda n=n: pennylane_sim.run_trajectory(n, p, rng),
        lambda n=n: run_trajectory(n, p, rng),
        lambda n=n: rand_unitary_mi_state(n, p, rng),
    ):
        t = time.perf_counter()
        for _ in range(reps):
            f()
        row.append((time.perf_counter() - t) / reps)
    print(f"{n:>3} " + " ".join(f"{x:>16.4f}" for x in row), flush=True)
