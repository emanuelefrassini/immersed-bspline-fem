"""Sparse Galerkin assembly and Dirichlet enforcement for -u'' = f."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from .quadrature import cell_quadrature, values
from .space import SplineSpace


@dataclass
class Solution:
    space: SplineSpace
    coefficients: np.ndarray
    matrix: csr_matrix
    rhs: np.ndarray
    method: str

    def evaluate(self, x, derivative=False):
        return np.asarray(self.space.basis(x, derivative) @ self.coefficients)

    def errors(self, exact, exact_derivative, order=None):
        """Return L2 and full H1 errors integrated on the physical domain."""
        x, w = cell_quadrature(self.space.mesh, order or max(12, self.space.degree + 3))
        e = self.evaluate(x) - values(exact, x)
        de = self.evaluate(x, True) - values(exact_derivative, x)
        l2 = np.dot(w, e**2)
        return float(np.sqrt(l2)), float(np.sqrt(l2 + np.dot(w, de**2)))

    def condition_number(self, scaled=False):
        """Dense 2-norm diagnostic for small research examples only."""
        A = self.matrix.toarray()
        if scaled:
            diag = np.diag(A)
            if np.any(diag <= 0):
                raise ValueError("Diagonal scaling requires positive diagonal")
            s = 1 / np.sqrt(diag)
            A = s[:, None] * A * s[None, :]
        return float(np.linalg.cond(A))


def solve_poisson(space: SplineSpace, source, boundary=(0.0, 0.0), *,
                  method="nitsche", gamma=10.0, quadrature_order=None):
    """Solve the Poisson problem with strong, penalty, or symmetric Nitsche BCs.

    For penalty, gamma is the absolute coefficient of the boundary term.
    For Nitsche, gamma > 1 multiplies p**2 / physical boundary-cell length.
    The returned matrix is the actual free-DOF system for strong enforcement.
    """
    if method not in {"strong", "penalty", "nitsche"}:
        raise ValueError("Unknown boundary enforcement method")
    if method == "strong" and not space.mesh.fitted:
        raise ValueError("Strong endpoint elimination requires a fitted mesh")
    if method != "strong" and (not np.isfinite(gamma) or gamma <= (1 if method == "nitsche" else 0)):
        raise ValueError("Require gamma > 1 for Nitsche, gamma > 0 for penalty")
    g = np.asarray(boundary, dtype=float)
    if g.shape != (2,) or not np.isfinite(g).all():
        raise ValueError("Two finite Dirichlet values are required")
    order = max(8, space.degree + 1) if quadrature_order is None else quadrature_order
    if order < space.degree:
        raise ValueError("Quadrature order must be >= degree for stiffness integration")
    x, w = cell_quadrature(space.mesh, order)
    B = space.basis(x)
    D = space.basis(x, True)
    A = (D.T @ diags(w) @ D).tocsr()
    rhs = np.asarray(B.T @ (w * values(source, x))).ravel()
    if not np.isfinite(rhs).all():
        raise ValueError("Nonfinite load")
    if method == "strong":
        free = np.arange(1, space.ndof - 1)
        ends = np.array([0, space.ndof - 1])
        rhs = rhs[free] - A[free, :][:, ends] @ g
        A = A[free, :][:, free].tocsr()
        c = np.empty(space.ndof)
        c[ends] = g
        c[free] = spsolve(A, rhs)
    else:
        V = space.basis(space.mesh.domain)
        N = diags([-1.0, 1.0]) @ space.basis(space.mesh.domain, True)
        if method == "nitsche":
            lengths = np.diff(space.mesh.cells, axis=1).ravel()[[0, -1]]
            penalty = gamma * space.degree**2 / lengths
            A = A - V.T @ N - N.T @ V
            rhs -= N.T @ g
        else:
            penalty = np.full(2, gamma, dtype=float)
        A = (A + V.T @ diags(penalty) @ V).tocsr()
        rhs += V.T @ (penalty * g)
        c = spsolve(A, rhs)
    if not np.isfinite(c).all():
        raise RuntimeError("Nonfinite solution; inspect matrix conditioning")
    return Solution(space, c, A, rhs, method)
