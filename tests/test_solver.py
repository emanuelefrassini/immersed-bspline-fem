import numpy as np
import pytest
from scipy.interpolate import BSpline
from scipy.sparse import issparse

from immersed_bspline_fem import Mesh, SplineSpace, solve_poisson, uniform_mesh
from immersed_bspline_fem.quadrature import cell_quadrature


@pytest.mark.parametrize('p,k', [(1, 0), (2, 0), (2, 1), (3, 0), (3, 2), (4, 2)])
def test_basis_and_derivative(p, k):
    s = SplineSpace(uniform_mesh(7, .6), p, k)
    x = np.linspace(0, 1, 91)
    basis = s.basis(x).toarray()
    derivative = s.basis(x, True).toarray()
    assert np.min(basis) >= -1e-14
    np.testing.assert_allclose(basis.sum(axis=1), 1, atol=1e-13)
    np.testing.assert_allclose(derivative.sum(axis=1), 0, atol=1e-12)
    ref = BSpline(s.knots, np.eye(s.ndof), p)
    np.testing.assert_allclose(basis, ref(x), atol=1e-13)
    np.testing.assert_allclose(derivative, ref(x, nu=1), atol=1e-12)


@pytest.mark.parametrize('theta', [0, .5, .99])
def test_physical_quadrature(theta):
    mesh = uniform_mesh(6, theta, domain=(-2, 3))
    x, w = cell_quadrature(mesh, 5)
    assert np.all((x > -2) & (x < 3))
    assert np.all(w > 0)
    assert w.sum() == pytest.approx(5)
    assert np.dot(w, x**4) == pytest.approx((3**5 - (-2)**5)/5)
    assert mesh.cut_fractions.min() <= 1 + 1e-14


@pytest.mark.parametrize('theta,method', [(0, 'strong'), (0, 'nitsche'), (.5, 'nitsche'), (.9, 'nitsche')])
def test_quadratic_patch(theta, method):
    s = SplineSpace(uniform_mesh(6, theta))
    sol = solve_poisson(s, lambda x: -4, (7, 10), method=method)
    assert issparse(sol.matrix)
    assert np.linalg.eigvalsh(sol.matrix.toarray()).min() > 0
    np.testing.assert_allclose(sol.matrix.toarray(), sol.matrix.toarray().T, atol=1e-11)
    x = np.linspace(0, 1, 101)
    np.testing.assert_allclose(sol.evaluate(x), 2*x*x + x + 7, atol=1e-9, rtol=0)
    np.testing.assert_allclose(sol.evaluate(x, True), 4*x + 1, atol=1e-8, rtol=0)
    c = sol.coefficients[1:-1] if method == 'strong' else sol.coefficients
    np.testing.assert_allclose(sol.matrix @ c, sol.rhs, atol=1e-9)


def test_penalty_nonzero_boundary_values():
    s = SplineSpace(uniform_mesh(6, .5))
    # Zero source, equal nonzero boundary data: a constant is represented exactly.
    sol = solve_poisson(s, lambda x: 0, (3, 3), method='penalty', gamma=100)
    np.testing.assert_allclose(sol.evaluate(np.linspace(0, 1, 30)), 3, atol=1e-12)
    # With zero source and unequal Dirichlet values, the penalty method solves
    # the Robin-perturbed problem exactly: u(x)=(gamma*x+1)/(gamma+2).
    sol = solve_poisson(s, lambda x: 0, (0, 1), method='penalty', gamma=100)
    x = np.linspace(0, 1, 30)
    np.testing.assert_allclose(sol.evaluate(x), (100*x + 1)/102, atol=1e-12)


@pytest.mark.parametrize('p,k', [(1, 0), (2, 1), (3, 2), (3, 0)])
@pytest.mark.parametrize('theta,method', [(0, 'strong'), (.5, 'nitsche')])
def test_convergence(p, k, theta, method):
    exact = lambda x: np.expm1(x)/np.expm1(1)
    derivative = lambda x: np.exp(x)/np.expm1(1)
    errors, hs = [], []
    for m in (8, 16, 32):
        mesh = uniform_mesh(m, theta)
        sol = solve_poisson(SplineSpace(mesh, p, k), lambda x: -derivative(x), (0, 1), method=method)
        errors.append(sol.errors(exact, derivative))
        hs.append(mesh.h)
    rates = np.log(np.array(errors[-2])/errors[-1])/np.log(hs[-2]/hs[-1])
    assert rates[0] > p + .7
    assert rates[1] > p - .3


def test_cut_cell_conditioning_and_scaling():
    def run(theta):
        return solve_poisson(SplineSpace(uniform_mesh(6, theta)), lambda x: 0, (0, 1))
    mild, tiny = run(.5), run(.99)
    assert tiny.condition_number() > 1e4 * mild.condition_number()
    assert tiny.condition_number(scaled=True) < tiny.condition_number()/1e4
    assert np.linalg.eigvalsh(tiny.matrix.toarray()).min() > 0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        uniform_mesh(6, 1)
    with pytest.raises(ValueError):
        Mesh(np.array([0, 1, 2, 3]), (1.5, 3))
    with pytest.raises(ValueError):
        SplineSpace(uniform_mesh(6), 2, 2)
    s = SplineSpace(uniform_mesh(6, .5))
    with pytest.raises(ValueError):
        solve_poisson(s, lambda x: 0, method='strong')
    with pytest.raises(ValueError):
        solve_poisson(s, lambda x: 0, gamma=1)
    with pytest.raises(ValueError):
        solve_poisson(s, lambda x: 0, quadrature_order=1)
