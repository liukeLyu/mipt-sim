"""Entanglement-weighted graphs (EWG): where in spacetime the measurements sit, on average,
in the circuit realizations that end up entangling two regions.

For each trajectory eps we keep the binary measurement record W_eps(x, t) (1 if qubit x
was measured after layer t) and a weight E(eps) computed on the final state, e.g. the
negativity between regions A and B. The graph is the weighted average

    W[E](x, t) = sum_eps E(eps) W_eps(x, t) / sum_eps E(eps),

which equals the plain measurement rate p wherever the weight does not care about
measurements there, falls below p where measurements kill the entanglement, and rises
above p where measurements help (for example by shielding A and B from the rest).
Lyu et al., arXiv:2602.04969, Sec. III B.
"""

from __future__ import annotations

import numpy as np
from medt import keep_systems, log_negativity

from .numpy_sim import random_circuit, run_circuit


def measurement_record(layers, n: int) -> np.ndarray:
    """Binary record W(t, x): 1 if qubit x was measured after layer t."""
    w = np.zeros((len(layers), n), dtype=bool)
    for t, (_, measured) in enumerate(layers):
        w[t, measured] = True
    return w


def _entropy(rho: np.ndarray) -> float:
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-14]
    return float(-np.sum(ev * np.log2(ev)))


def pair_weights(psi: np.ndarray, a, b) -> dict:
    """Negativity, log-negativity and mutual information (bits) between regions a and b of a pure state."""
    n = psi.ndim
    dims = [2] * n
    keep = list(a) + list(b)
    rho_ab = keep_systems(psi.reshape(-1), keep, dims)
    en = max(log_negativity(rho_ab, list(range(len(a))), [2] * len(keep)), 0.0)
    rho_a = keep_systems(psi.reshape(-1), list(a), dims)
    rho_b = keep_systems(psi.reshape(-1), list(b), dims)
    mi = _entropy(rho_a) + _entropy(rho_b) - _entropy(rho_ab)
    return {"negativity": (2**en - 1) / 2, "log_negativity": en, "mutual_information": mi}


def sample(n: int, p: float, a, b, seed: int, depth: int | None = None) -> tuple[np.ndarray, dict]:
    """One trajectory: its measurement record and its weights."""
    rng = np.random.default_rng(seed)
    layers = random_circuit(n, p, rng, depth)
    psi = run_circuit(n, layers, rng)
    return measurement_record(layers, n), pair_weights(psi, a, b)


def weighted_graph(records: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """W[E](t, x) from stacked records (samples, t, x) and weights (samples,)."""
    return np.tensordot(weights, records.astype(float), axes=1) / weights.sum()
