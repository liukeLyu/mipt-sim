"""Measurement-induced phase transition (MIPT) in Haar brickwork circuits.

- ``mipt_sim.numpy_sim`` (exported here): the simulator, pure NumPy.
- ``mipt_sim.pennylane_check``: the same circuit as a native PennyLane QNode,
  used to check the simulator's sampling (needs PennyLane).
"""

from .circuit import brickwork_pairs, haar_unitary
from .numpy_sim import (
    apply_gate,
    half_chain_entropy,
    interval_entropy,
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
    "interval_entropy",
    "measure_qubit",
    "random_circuit",
    "region_entropy",
    "run_circuit",
    "run_trajectory",
    "zero_state",
]
