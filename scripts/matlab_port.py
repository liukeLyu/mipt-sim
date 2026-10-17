"""Line-by-line NumPy port of Liuke's MATLAB trajectory (Repo/QuantumCircuit/randUnitaryMIstate.m).

Used only as a speed baseline: same algorithm (layer as a Kronecker product of
4 x 4 gates applied by reshape + matmul, odd layer by permuting the state,
projector P0 by zeroing a block), same layout (qubit 1 = most significant bit).
"""

import numpy as np

from pl_mipt.circuit import haar_unitary


def ckronx(us, psi):
    # apply kron(us[0], us[1], ...) to psi, like ckronx_fast.m / ukronx.m
    c = psi
    for u in us:
        c = c.reshape(4, -1)
        c = (u @ c).T.copy()  # rotate the next factor to the front
    return c.reshape(-1)


def even_layer(psi, n, rng):
    return ckronx([haar_unitary(4, rng) for _ in range(n // 2)], psi)


def odd_layer(psi, n, rng):
    t = psi.reshape((2,) * n)
    t = np.transpose(t, list(range(1, n)) + [0]).reshape(-1)  # statePermute(perm=[2:N 1])
    t = even_layer(t, n, rng).reshape((2,) * n)
    return np.transpose(t, [n - 1] + list(range(n - 1))).reshape(-1)  # invperm


def measurement_layer(psi, n, p, rng):
    for q in np.flatnonzero(rng.random(n) < p):
        stride = 2 ** (n - 1 - q)
        tmp = psi.reshape(-1, 2 * stride)
        p0psi = tmp.copy()
        p0psi[:, stride:] = 0
        p0 = np.vdot(p0psi, p0psi).real
        if rng.random() < p0:
            psi = p0psi.reshape(-1) / np.sqrt(p0)
        else:
            psi = (psi - p0psi.reshape(-1)) / np.sqrt(1 - p0)
    return psi


def rand_unitary_mi_state(n, p, rng):
    psi = np.zeros(2**n, complex)
    psi[0] = 1
    for _ in range(n):
        psi = measurement_layer(even_layer(psi, n, rng), n, p, rng)
        psi = measurement_layer(odd_layer(psi, n, rng), n, p, rng)
    return psi
