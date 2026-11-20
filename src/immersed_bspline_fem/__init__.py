"""Fitted and immersed B-spline discretizations of 1D Poisson problems."""
from .mesh import Mesh, uniform_mesh
from .space import SplineSpace
from .solver import Solution, solve_poisson

__all__ = ["Mesh", "uniform_mesh", "SplineSpace", "Solution", "solve_poisson"]
