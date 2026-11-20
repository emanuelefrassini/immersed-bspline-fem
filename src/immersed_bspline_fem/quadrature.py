"""Gauss-Legendre quadrature on physical cells, including cut cells."""
import numpy as np
from numpy.polynomial.legendre import leggauss


def cell_quadrature(mesh, order):
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("Quadrature order must be a positive integer")
    q, w = leggauss(order)
    lo, hi = mesh.cells.T
    half = (hi - lo) / 2
    x = (lo + hi)[:, None] / 2 + half[:, None] * q
    weights = half[:, None] * w
    return x.ravel(), weights.ravel()


def values(function, x):
    """Allow vectorized functions returning either arrays or scalar constants."""
    return np.broadcast_to(np.asarray(function(x), dtype=float), x.shape)
