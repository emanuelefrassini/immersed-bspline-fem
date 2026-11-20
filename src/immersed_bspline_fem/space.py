"""Open B-spline spaces with independently configurable degree and continuity."""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import diags

from .mesh import Mesh


@dataclass(frozen=True)
class SplineSpace:
    mesh: Mesh
    degree: int = 2
    continuity: int = 1

    def __post_init__(self):
        p, k = self.degree, self.continuity
        if not isinstance(p, (int, np.integer)) or not isinstance(k, (int, np.integer)):
            raise ValueError("degree and continuity must be integers")
        if not 1 <= p or not 0 <= k < p:
            raise ValueError("Require degree >= 1 and 0 <= continuity < degree")

    @cached_property
    def knots(self):
        p, k = self.degree, self.continuity
        x = self.mesh.nodes
        t = np.r_[np.repeat(x[0], p + 1), np.repeat(x[1:-1], p - k),
                  np.repeat(x[-1], p + 1)]
        t.setflags(write=False)
        return t

    @property
    def ndof(self):
        return len(self.knots) - self.degree - 1

    @cached_property
    def derivative_operator(self):
        p, t, n = self.degree, self.knots, self.ndof
        factor = p / (t[p + 1:-1] - t[1:n])
        return diags([-factor, factor], [0, 1], shape=(n - 1, n), format="csr")

    def basis(self, x, derivative=False):
        """Sparse basis matrix, evaluated in the original background coordinates."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if derivative:
            low = BSpline.design_matrix(x, self.knots[1:-1], self.degree - 1)
            return (low @ self.derivative_operator).tocsr()
        return BSpline.design_matrix(x, self.knots, self.degree)
