"""Uniform background meshes and their physical intersections."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mesh:
    """Strictly increasing background nodes; all cells intersect the domain."""

    nodes: np.ndarray
    domain: tuple[float, float]

    def __post_init__(self):
        nodes = np.array(self.nodes, dtype=float, copy=True)
        a, b = map(float, self.domain)
        if (nodes.ndim != 1 or len(nodes) < 3 or not np.all(np.isfinite(nodes))
                or np.any(np.diff(nodes) <= 0)):
            raise ValueError("At least two positive-length cells are required")
        if not (nodes[0] <= a < nodes[1] and nodes[-2] < b <= nodes[-1]):
            raise ValueError("Both boundary cells must have positive physical length")
        nodes.setflags(write=False)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "domain", (a, b))

    @property
    def cells(self):
        """Physical integration intervals (clipped to the domain)."""
        a, b = self.domain
        return np.column_stack((np.maximum(self.nodes[:-1], a),
                                np.minimum(self.nodes[1:], b)))

    @property
    def cut_fractions(self):
        return np.diff(self.cells, axis=1).ravel() / np.diff(self.nodes)

    @property
    def h(self):
        return float(np.max(np.diff(self.nodes)))

    @property
    def fitted(self):
        return self.nodes[0] == self.domain[0] and self.nodes[-1] == self.domain[1]


def uniform_mesh(elements: int, theta: float = 0.0, domain=(0.0, 1.0)) -> Mesh:
    """Assignment geometry: padding = theta * L / (elements - 2).

    theta=0 is fitted; theta approaching 1 shrinks the two physical end cells.
    The background has `elements` cells, with no completely exterior cells.
    """
    if not isinstance(elements, (int, np.integer)) or elements < 3:
        raise ValueError("elements must be an integer >= 3")
    if not 0 <= theta < 1:
        raise ValueError("theta must be in [0, 1)")
    a, b = map(float, domain)
    if not np.isfinite([a, b]).all() or a >= b:
        raise ValueError("Invalid domain")
    padding = theta * (b - a) / (elements - 2)
    return Mesh(np.linspace(a - padding, b + padding, elements + 1), (a, b))
