"""Measurement-induced phase transition (MIPT) in Haar brickwork circuits.

Two simulators with the same interface:

- ``pl_mipt.numpy_sim`` (also exported here): pure NumPy, no PennyLane import.
- ``pl_mipt.pennylane_sim``: gates applied by PennyLane's state-vector kernel,
  plus ``circuit_qnode``, the circuit as a native PennyLane QNode.
"""

from .circuit import brickwork_pairs, haar_unitary
from .numpy_sim import (
    apply_gate,
    half_chain_entropy,
    measure_qubit,
    random_circuit,
    region_entropy,
    run_circuit,
    run_trajectory,
    zero_state,
)

__all__ = [
    "apply_gate",
    "brickwork_pairs",
    "haar_unitary",
    "half_chain_entropy",
    "measure_qubit",
    "random_circuit",
    "region_entropy",
    "run_circuit",
    "run_trajectory",
    "zero_state",
]
