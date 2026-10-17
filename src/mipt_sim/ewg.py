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


def _reduced(psi: np.ndarray, keep) -> np.ndarray:
    """Reduced density matrix of the qubits `keep` (in that order) of a pure state tensor."""
    rest = [k for k in range(psi.ndim) if k not in keep]
    m = np.transpose(psi, list(keep) + rest).reshape(2 ** len(keep), -1)
    return m @ m.conj().T


def pair_weights(psi: np.ndarray, a, b) -> dict:
    """Negativity, log-negativity and mutual information (bits) between regions a and b of a pure state.

    Plain NumPy, so it runs where SciPy (and hence medt) is not installed; tests check it against medt.
    """
    na, nb = len(a), len(b)
    rho_ab = _reduced(psi, list(a) + list(b))
    pt = rho_ab.reshape([2] * (2 * (na + nb)))  # row indices (a, b), then column indices (a, b)
    axes = list(range(2 * (na + nb)))
    for k in range(na):  # transpose subsystem a: swap its row and column indices
        axes[k], axes[na + nb + k] = axes[na + nb + k], axes[k]
    pt = np.transpose(pt, axes).reshape(rho_ab.shape)
    en = max(float(np.log2(np.abs(np.linalg.eigvalsh(pt)).sum())), 0.0)
    rho_a = _reduced(psi, list(a))
    rho_b = _reduced(psi, list(b))
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
