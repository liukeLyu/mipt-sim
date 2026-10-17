"""The monitored circuit as a native PennyLane QNode, used to check the NumPy simulator.

In exact mode PennyLane's ``qml.measure`` either adds an ancilla per measurement
(deferred) or sums over outcome branches (tree traversal). It never hands back the
post-measurement state of a single trajectory, so it cannot give per-trajectory
entropies. It does give exact outcome-averaged expectation values, which the
Monte Carlo average over numpy_sim trajectories must reproduce.
"""

from __future__ import annotations

import pennylane as qml


def circuit_qnode(n: int, layers, observables, mcm_method: str = "tree-traversal"):
    """QNode returning the trajectory-averaged expectation values of `observables`.

    `layers` is the output of numpy_sim.random_circuit.
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
