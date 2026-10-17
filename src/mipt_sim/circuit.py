"""Circuit geometry and random gates."""

from __future__ import annotations

import numpy as np


def haar_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random d x d unitary (QR of a complex Ginibre matrix with phase fix).

    Mezzadri, arXiv:math-ph/0609050.
    """
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def brickwork_pairs(n: int, layer: int) -> list[tuple[int, int]]:
    """Qubit pairs acted on in brickwork layer `layer` of an n-qubit periodic chain.

    Even layers: (0,1), (2,3), ...  Odd layers: (1,2), (3,4), ..., (n-1, 0).
    Requires even n.
    """
    if n % 2:
        raise ValueError("n must be even")
    s = layer % 2
    return [((2 * k + s) % n, (2 * k + s + 1) % n) for k in range(n // 2)]
