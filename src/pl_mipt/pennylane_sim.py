"""The same simulator with gates applied by PennyLane, plus a native-QNode version of the circuit.

Gates go through PennyLane's state-vector kernel ``qml.devices.qubit.apply_operation``
(what ``default.qubit`` runs internally). Measurements and entropies are shared with
numpy_sim: in exact mode PennyLane's ``qml.measure`` either adds an ancilla per
measurement (deferred) or averages over outcomes (tree traversal), and never hands
back the post-measurement state of a single trajectory.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from . import numpy_sim


def apply_gate(psi: np.ndarray, u: np.ndarray, wires: tuple[int, int]) -> np.ndarray:
    """Apply a 4 x 4 unitary to two wires with PennyLane's kernel. Cost ~ 4 * 2^n."""
    return qml.devices.qubit.apply_operation(qml.QubitUnitary(u, wires=wires, unitary_check=False), psi)


def run_circuit(n: int, layers, rng: np.random.Generator, record: bool = False):
    """numpy_sim.run_circuit with PennyLane applying the gates."""
    return numpy_sim.run_circuit(n, layers, rng, record, gate=apply_gate)


def run_trajectory(n: int, p: float, rng: np.random.Generator, depth: int | None = None, record: bool = False):
    """Draw a random circuit and run one trajectory of it with PennyLane applying the gates."""
    return run_circuit(n, numpy_sim.random_circuit(n, p, rng, depth), rng, record)


def circuit_qnode(n: int, layers, observables, mcm_method: str = "tree-traversal"):
    """The same monitored circuit as a native PennyLane QNode with qml.measure.

    In exact mode PennyLane averages over measurement outcomes, so this returns
    trajectory-averaged expectation values (useful to check the sampling of
    run_circuit), not per-trajectory entropies.
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
