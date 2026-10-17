# pennylane-mipt

The measurement-induced phase transition (MIPT) in Haar-random brickwork circuits, simulated with [PennyLane](https://pennylane.ai).

A chain of L qubits evolves under random two-qubit gates. Each qubit is also measured with probability p after every layer. Gates create entanglement and measurements destroy it. At low p, the half-chain entanglement entropy grows with L (a **volume law**). At high p it saturates (an **area law**, which in 1D means a constant). For Haar gates on qubits, the transition sits near p_c ≈ 0.17 ([Zabalo et al., arXiv:1911.00008](https://arxiv.org/abs/1911.00008)). For background, see [Li, Chen & Fisher, arXiv:1808.06134](https://arxiv.org/abs/1808.06134), [Skinner, Ruhman & Nahum, arXiv:1808.05953](https://arxiv.org/abs/1808.05953), and the review by [Fisher, Khemani, Nahum & Vijay, arXiv:2207.14280](https://arxiv.org/abs/2207.14280).

![half-chain entropy vs L](results/entropy_vs_L.png)

**Result** (`scripts/sweep.py`, L = 8 to 20, 40 to 800 trajectories per point, error bars are one standard error, about the size of the markers). At p = 0.05 and 0.1, S grows linearly in L, with a constant entropy density S/L of about 0.34 and 0.25 bits per qubit (volume law). At p = 0.3 and 0.5, S stays flat at about 1 bit and 0.25 bit (area law). Between them, p = 0.15 and 0.2 grow slowly while S/L drifts down. That is what sizes this small look like near p_c ≈ 0.17, where S grows like log L. Locating p_c precisely needs a finite-size collapse or a better probe, such as the tripartite mutual information used by Zabalo et al.

## Model

The model matches `Repo/QuantumCircuit/randUnitaryMIstate.m`:

- L qubits (L even), periodic boundary conditions, starting from |0…0⟩.
- Brickwork layers alternate between bonds (0,1),(2,3),… and (1,2),…,(L−1,0). Every gate is an independent Haar-random 4×4 unitary.
- After every layer, each qubit is measured in Z with probability p. The outcome is drawn with the Born rule.
- The circuit has depth 2L layers. The half-chain von Neumann entropy (in bits) is averaged over trajectories and over two neighbouring cut positions. At large p, the entropy depends on whether the last layer acted on the cut bond, which gives an even/odd effect in L/2.

## Two simulators

Both run the same algorithm on a dense `(2,)*L` state tensor and share the measurement and entropy code. Given the same random seed, they produce the same trajectory to rounding error (`tests/test_mipt.py`).

| module | gates applied by | needs PennyLane | use it for |
|---|---|---|---|
| `pl_mipt.numpy_sim` (also `pl_mipt.*`) | one `np.tensordot` per gate | no | production runs: the faster of the two, easier to read, and the natural starting point for a GPU port (swap NumPy for CuPy or JAX) |
| `pl_mipt.pennylane_sim` | PennyLane's `apply_operation` kernel | yes | comparing with PennyLane, and `circuit_qnode`, the circuit as a native QNode with `qml.measure` |

The sweep below uses `numpy_sim`.

## How PennyLane is used, and where it falls short

The simulation needs one post-measurement state per trajectory. In exact (no-shots) mode, PennyLane's `qml.measure` does not provide one. Checked on PennyLane 0.45:

| `mcm_method` | what it does | per-trajectory entropy? |
|---|---|---|
| `deferred` | adds one ancilla qubit per measurement | no: entropy of the outcome-averaged state, and memory doubles with every measurement |
| `tree-traversal` | sums over all outcome branches | no: only averaged expectation values; `qml.state` and `qml.vn_entropy` are refused |
| `one-shot` | samples outcomes, needs `shots` | no: state-based measurements are refused with finite shots |

`default.clifford` behaves the same way. So `pennylane_sim` uses PennyLane's state-vector kernel `qml.devices.qubit.apply_operation`, the routine `default.qubit` runs internally, to apply each gate to a `(2,)*L` tensor. A gate costs about 4·2^L operations and no layer matrix is built. The Born-rule measurement is done by hand on the tensor. `pennylane_sim.circuit_qnode` builds the same circuit with `qml.measure`. It is used to check that the sampled trajectories reproduce PennyLane's exact outcome-averaged expectation values (`tests/test_mipt.py`).

### Speed versus the MATLAB algorithm

Seconds per trajectory at p = 0.17, depth 2L, on a 4-core cloud machine (`scripts/benchmark.py`). Divide by 2L for the time per layer.

| L | `pennylane_sim` | `numpy_sim` | NumPy port of `randUnitaryMIstate.m` |
|---|---|---|---|
| 12 | 0.023 | 0.011 | 0.010 |
| 16 | 0.14 | 0.12 | 0.12 |
| 18 | 0.68 | 0.64 | 0.58 |
| 20 | 2.6 | 2.4 | 2.7 |
| 22 | 26 | 24 | 28 |

For L ≥ 16 the three implementations run within about 10% of each other. Below that, PennyLane's per-gate overhead roughly doubles the time. The NumPy port runs the same steps as the MATLAB code (Kronecker-product layer, a state permutation for odd layers, a block projector), but this table does not time MATLAB itself. For a direct comparison, run `tic; randUnitaryMIstate(20, 0.17, 20); toc` in MATLAB.

The bottom line is that PennyLane adds nothing here beyond a correct gate kernel. Larger L or many more trajectories would need a GPU or batched trajectories, not a different CPU library.

## Possible next steps

- **A PennyLane contribution.** Let `one-shot` return state-based measurements (`qml.vn_entropy`, `qml.purity`, `qml.state`) for each shot. The device already holds the sampled post-measurement state, so the computation exists internally. A [forum thread](https://discuss.pennylane.ai/t/mid-circuit-measurement-with-qml-state/7491) asks about it.
- **Hardware-friendly probes** that need only shots and mid-circuit measurements, which is where PennyLane fits naturally: the reference-qubit purification probe ([Gullans & Huse, arXiv:1905.05195](https://arxiv.org/abs/1905.05195)) and the cross-entropy benchmark ([Li, Vijay & Fisher, arXiv:2209.00609](https://arxiv.org/abs/2209.00609)).
- **Clifford circuits** on `default.clifford` for L in the hundreds ([Li, Chen & Fisher, arXiv:1808.06134](https://arxiv.org/abs/1808.06134)), with the same mid-circuit limitation to work around.
- **Multipartite entanglement** of the steady state with `medt`. `region_entropy` already uses `medt.keep_systems`, and the next step is negativity and GMN of small subregions.

## Use

```bash
pip install -e ../multipartite-entanglement-detection-toolkit
pip install -e .               # NumPy simulator only
pip install -e ".[dev]"        # adds PennyLane, pytest, ruff, matplotlib
pytest -q                      # 11 tests, a few seconds
python scripts/sweep.py        # writes results/sweep.json (about 15 min on 4 cores)
python scripts/plot.py         # writes results/entropy_vs_L.png
python scripts/benchmark.py 12 16 20
```

```python
import numpy as np
from pl_mipt import run_trajectory, half_chain_entropy

psi = run_trajectory(14, 0.1, np.random.default_rng(0))   # L = 14 qubits, p = 0.1
print(half_chain_entropy(psi))   # bits

from pl_mipt import pennylane_sim  # same interface, PennyLane applies the gates
psi = pennylane_sim.run_trajectory(14, 0.1, np.random.default_rng(0))
```
