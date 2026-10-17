import numpy as np
import pytest

from mipt_sim import (
    apply_gate,
    brickwork_pairs,
    haar_unitary,
    half_chain_entropy,
    measure_qubit,
    random_circuit,
    region_entropy,
    run_circuit,
    run_trajectory,
)


def random_state(n, rng):
    psi = rng.standard_normal((2,) * n) + 1j * rng.standard_normal((2,) * n)
    return psi / np.linalg.norm(psi)


@pytest.mark.parametrize("wires", [(0, 1), (4, 0), (1, 3)])
def test_gate_matches_dense_matrix(wires):
    rng = np.random.default_rng(0)
    psi, u = random_state(5, rng), haar_unitary(4, rng)
    # dense 32 x 32 matrix of u on `wires`, built independently with einsum
    i, j = wires
    letters = "abcde"
    out = letters.replace(letters[i], "x").replace(letters[j], "y")
    full = np.einsum(f"xy{letters[i]}{letters[j]},{letters}->{out}", u.reshape(2, 2, 2, 2), psi)
    assert np.allclose(apply_gate(psi, u, wires), full)


def test_haar_unitary_is_unitary():
    u = haar_unitary(4, np.random.default_rng(1))
    assert np.allclose(u.conj().T @ u, np.eye(4))


def test_brickwork_is_periodic():
    assert brickwork_pairs(6, 0) == [(0, 1), (2, 3), (4, 5)]
    assert brickwork_pairs(6, 1) == [(1, 2), (3, 4), (5, 0)]


def test_measurement_collapses_and_normalizes():
    rng = np.random.default_rng(2)
    psi = np.moveaxis(random_state(4, rng), 0, 3)  # non-contiguous, like a gate output
    psi, m = measure_qubit(psi, 2, rng)
    assert np.isclose(np.linalg.norm(psi), 1)
    assert np.allclose(np.take(psi, 1 - m, axis=2), 0)


def test_half_chain_entropy_matches_medt():
    psi = random_state(8, np.random.default_rng(3))
    assert np.isclose(half_chain_entropy(psi), region_entropy(psi, [0, 1, 2, 3]))
    assert np.isclose(half_chain_entropy(psi, 3), region_entropy(psi, [3, 4, 5, 6]))
    assert np.isclose(half_chain_entropy(psi, 6), region_entropy(psi, [6, 7, 0, 1]))
    # complementary regions of a pure state have equal entropy
    assert np.isclose(region_entropy(psi, [7, 0, 1]), region_entropy(psi, [2, 3, 4, 5, 6]))


def test_no_measurement_reaches_volume_law_and_full_measurement_gives_product():
    rng = np.random.default_rng(4)
    n = 10
    assert half_chain_entropy(run_trajectory(n, 0.0, rng)) > n / 2 - 1  # Page value ~ n/2 - 0.72
    assert half_chain_entropy(run_trajectory(n, 1.0, rng)) < 1e-10


def test_born_sampling_matches_pennylane_exact_average():
    """Monte Carlo average over trajectories of a fixed circuit equals PennyLane's
    exact outcome-averaged expectation values (tree-traversal mid-circuit measurements)."""
    qml = pytest.importorskip("pennylane")
    from mipt_sim.pennylane_check import circuit_qnode

    rng = np.random.default_rng(5)
    n, ntraj = 4, 20000
    layers = random_circuit(n, 0.5, rng, depth=4)
    obs = [qml.Z(0), qml.Z(0) @ qml.Z(2), qml.X(1), qml.Y(3)]
    exact = np.array(circuit_qnode(n, layers, obs)())
    mats = [qml.matrix(o, wire_order=range(n)) for o in obs]
    mc = np.zeros(len(obs))
    for _ in range(ntraj):
        v = run_circuit(n, layers, rng).reshape(-1)
        mc += [np.vdot(v, m @ v).real for m in mats]
    assert np.allclose(mc / ntraj, exact, atol=4 / np.sqrt(ntraj))


def test_simulator_does_not_import_pennylane():
    import subprocess
    import sys

    code = "import sys, mipt_sim; mipt_sim.run_trajectory(6, 0.3, __import__('numpy').random.default_rng(0)); "
    code += "assert 'pennylane' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_pair_weights_on_bell_pairs():
    from mipt_sim.ewg import pair_weights

    # qubits 0-2 and 1-3 in Bell states: A = {0, 1} and B = {2, 3} share two ebits
    bell = np.array([[1, 0], [0, 1]]) / np.sqrt(2)
    psi = np.einsum("ac,bd->abcd", bell, bell)
    w = pair_weights(psi, [0, 1], [2, 3])
    assert np.isclose(w["log_negativity"], 2)
    assert np.isclose(w["negativity"], 1.5)
    assert np.isclose(w["mutual_information"], 4)
    w = pair_weights(psi, [0], [1])  # unentangled pair
    assert np.isclose(w["negativity"], 0) and np.isclose(w["mutual_information"], 0)


def test_weighted_graph_is_weighted_mean_of_records():
    from mipt_sim.ewg import weighted_graph

    records = np.array([[[1, 0]], [[0, 1]], [[1, 1]]], dtype=bool)
    g = weighted_graph(records, np.array([1.0, 0.0, 3.0]))
    assert np.allclose(g, [[1.0, 0.75]])


def test_pair_weights_match_medt():
    from medt import keep_systems, log_negativity

    from mipt_sim.ewg import pair_weights

    psi = random_state(8, np.random.default_rng(7))
    a, b = [0, 1], [5, 6]
    rho = keep_systems(psi.reshape(-1), a + b, [2] * 8)
    w = pair_weights(psi, a, b)
    assert np.isclose(w["log_negativity"], max(log_negativity(rho, [0, 1], [2] * 4), 0))
    assert np.isclose(
        w["mutual_information"], region_entropy(psi, a) + region_entropy(psi, b) - region_entropy(psi, a + b)
    )
