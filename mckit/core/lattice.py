"""Lattice (unit cell) — a thin wrapper around ``ase.cell.Cell``.

All cell-parameter and reciprocal-lattice computations are delegated to ASE
so we do not reinvent geometry that already exists upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.cell import Cell as ASECell


@dataclass
class Lattice:
    """Crystal lattice defined by three row-vectors in a 3x3 matrix (Å)."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        arr = self.matrix.array if isinstance(self.matrix, ASECell) else self.matrix
        self.matrix = np.asarray(arr, dtype=np.float64).reshape(3, 3)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_parameters(
        cls, a: float, b: float, c: float,
        alpha: float, beta: float, gamma: float,
    ) -> "Lattice":
        return cls(matrix=ASECell.new([a, b, c, alpha, beta, gamma]))

    @classmethod
    def cubic(cls, a: float) -> "Lattice":
        return cls.from_parameters(a, a, a, 90.0, 90.0, 90.0)

    @classmethod
    def hexagonal(cls, a: float, c: float) -> "Lattice":
        return cls.from_parameters(a, a, c, 90.0, 90.0, 120.0)

    @classmethod
    def from_ase_cell(cls, cell) -> "Lattice":
        return cls(matrix=np.asarray(cell, dtype=np.float64))

    # ------------------------------------------------------------------
    # ASE delegation
    # ------------------------------------------------------------------
    def to_ase_cell(self) -> ASECell:
        return ASECell(self.matrix.copy())

    @property
    def _cellpar(self) -> np.ndarray:
        return self.to_ase_cell().cellpar()

    @property
    def a_vec(self) -> np.ndarray:
        return self.matrix[0]

    @property
    def b_vec(self) -> np.ndarray:
        return self.matrix[1]

    @property
    def c_vec(self) -> np.ndarray:
        return self.matrix[2]

    @property
    def a(self) -> float:
        return float(self._cellpar[0])

    @property
    def b(self) -> float:
        return float(self._cellpar[1])

    @property
    def c(self) -> float:
        return float(self._cellpar[2])

    @property
    def alpha(self) -> float:
        return float(self._cellpar[3])

    @property
    def beta(self) -> float:
        return float(self._cellpar[4])

    @property
    def gamma(self) -> float:
        return float(self._cellpar[5])

    @property
    def volume(self) -> float:
        return float(self.to_ase_cell().volume)

    @property
    def reciprocal(self) -> "Lattice":
        """Reciprocal lattice (2π convention)."""
        rec = self.to_ase_cell().reciprocal().array * 2 * np.pi
        return Lattice(matrix=rec)

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------
    def fractional_to_cartesian(self, frac_coords) -> np.ndarray:
        return np.asarray(frac_coords, dtype=np.float64) @ self.matrix

    def cartesian_to_fractional(self, cart_coords) -> np.ndarray:
        return np.asarray(cart_coords, dtype=np.float64) @ np.linalg.inv(self.matrix)

    def __repr__(self) -> str:
        cp = self._cellpar
        return (
            f"Lattice(a={cp[0]:.4f}, b={cp[1]:.4f}, c={cp[2]:.4f}, "
            f"alpha={cp[3]:.2f}, beta={cp[4]:.2f}, gamma={cp[5]:.2f})"
        )
