"""Monitored-circuit trajectories on a dense state vector.

The state is a tensor of shape (2,) * n (PennyLane's internal layout, wire 0 first).
Gates go through PennyLane's state-vector kernel
(``qml.devices.qubit.apply_operation``, what ``default.qubit`` runs internally),
or through an equivalent ``np.tensordot`` when ``backend="numpy"``.
Measurements are sampled with the Born rule directly on the tensor, because in
exact mode PennyLane's ``qml.measure`` either adds an ancilla per measurement
(deferred) or averages over outcomes (tree traversal), and never hands back the
post-measurement state of a single trajectory.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from .circuit import brickwork_pairs, haar_unitary


def zero_state(n: int) -> np.ndarray:
    psi = np.zeros((2,) * n, dtype=complex)
    psi[(0,) * n] = 1.0
    return psi


def apply_gate(psi: np.ndarray, u: np.ndarray, wires: tuple[int, int], backend: str = "pennylane") -> np.ndarray:
    """Apply a 4 x 4 unitary to two wires of the state tensor. Cost ~ 4 * 2^n."""
    if backend == "pennylane":
        return qml.devices.qubit.apply_operation(qml.QubitUnitary(u, wires=wires, unitary_check=False), psi)
    i, j = wires
    out = np.tensordot(u.reshape(2, 2, 2, 2), psi, axes=[[2, 3], [i, j]])
    return np.moveaxis(out, [0, 1], [i, j])


def measure_qubit(psi: np.ndarray, q: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Projective Z measurement of qubit q with a Born-rule outcome.

    Returns (post-measurement state, outcome). Works in place when psi is C-contiguous.
    """
    n = psi.ndim
    psi = np.ascontiguousarray(psi)  # gates return strided views; reshape must not copy
    v = psi.reshape(2**q, 2, 2 ** (n - q - 1))
    p0 = np.vdot(v[:, 0], v[:, 0]).real
    outcome = int(rng.random() >= p0)
    v[:, 1 - outcome] = 0.0
    v /= np.sqrt(p0 if outcome == 0 else 1.0 - p0)
    return psi, outcome


def half_chain_entropy(psi: np.ndarray, start: int = 0, base: float = 2.0) -> float:
    """Von Neumann entropy of the n/2 consecutive qubits start, start+1, ... (periodic), via Schmidt values."""
    n = psi.ndim
    if start % n:
        psi = np.transpose(psi, np.roll(np.arange(n), -start))
    s = np.linalg.svd(psi.reshape(2 ** (n // 2), -1), compute_uv=False) ** 2
    s = s[s > 1e-14]
    return float(-np.sum(s * np.log(s)) / np.log(base))


def region_entropy(psi: np.ndarray, region, base: float = 2.0) -> float:
    """Von Neumann entropy of any set of qubits, via medt.keep_systems (reduced density matrix)."""
    from medt import keep_systems

    rho = keep_systems(psi.reshape(-1), list(region), [2] * psi.ndim)
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-14]
    return float(-np.sum(w * np.log(w)) / np.log(base))


def random_circuit(n: int, p: float, rng: np.random.Generator, depth: int | None = None):
    """Draw one Haar brickwork circuit with measurement locations.

    Each layer applies Haar 2-qubit gates on the brickwork pairs (periodic chain),
    then measures every qubit in Z independently with probability p.
    Default depth is 2n layers (n brickwork periods), as in randUnitaryMIstate.m.
    Returns a list of layers, each (list of (wires, 4x4 unitary), measured qubits).
    """
    depth = 2 * n if depth is None else depth
    return [
        (
            [(w, haar_unitary(4, rng)) for w in brickwork_pairs(n, t)],
            np.flatnonzero(rng.random(n) < p).tolist(),
        )
        for t in range(depth)
    ]


def run_circuit(n: int, layers, rng: np.random.Generator, backend: str = "pennylane", record: bool = False):
    """Run one trajectory of a fixed circuit, sampling measurement outcomes with the Born rule.

    Returns the final state, or (state, half-chain entropy after each layer) if record.
    """
    psi = zero_state(n)
    history = []
    for gates, measured in layers:
        for wires, u in gates:
            psi = apply_gate(psi, u, wires, backend)
        if measured:
            psi = np.ascontiguousarray(psi)  # one copy per layer, then measurements work in place
        for q in measured:
            psi, _ = measure_qubit(psi, q, rng)
        if record:
            history.append(half_chain_entropy(psi))
    return (psi, history) if record else psi


def run_trajectory(
    n: int,
    p: float,
    rng: np.random.Generator,
    depth: int | None = None,
    backend: str = "pennylane",
    record: bool = False,
):
    """Draw a random circuit and run one trajectory of it. See random_circuit and run_circuit."""
    return run_circuit(n, random_circuit(n, p, rng, depth), rng, backend, record)


def circuit_qnode(n: int, layers, observables, mcm_method: str = "tree-traversal"):
    """The same monitored circuit as a native PennyLane QNode with qml.measure.

    In exact mode PennyLane averages over measurement outcomes, so this returns
    trajectory-averaged expectation values (useful to check run_circuit's sampling),
    not per-trajectory entropies.
    """
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, mcm_method=mcm_method)
    def circuit():
        for gates, measured in layers:
            for wires, u in gates:
                qml.QubitUnitary(u, wires=wires)
            for q in measured:
                qml.measure(q)
        return [qml.expval(o) for o in observables]

    return circuit
