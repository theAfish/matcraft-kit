"""Atomic structure — a thin wrapper around ``ase.Atoms``.

``Structure`` keeps a single ``ase.Atoms`` instance internally and forwards
properties to it. Use ``structure.atoms`` to reach the full ASE API, or
``structure.to_pymatgen()`` for pymatgen.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from ase import Atom, Atoms

from mckit.core.lattice import Lattice


# 1 amu/Å³ in g/cm³
_AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660


class Structure:
    """A periodic atomic structure backed by ``ase.Atoms``.

    Construct either from explicit ``lattice/species/positions`` (fractional
    coords) or directly from an existing ``ase.Atoms`` via the ``atoms`` kwarg.
    """

    def __init__(
        self,
        lattice: Optional[Union[Lattice, np.ndarray]] = None,
        species: Optional[Sequence[str]] = None,
        positions: Optional[np.ndarray] = None,
        *,
        atoms: Optional[Atoms] = None,
    ) -> None:
        if atoms is not None:
            self._atoms = atoms
            return
        if lattice is None:
            raise ValueError("Provide either `atoms=` or `lattice=`.")
        species = list(species) if species is not None else []
        symbols = [str(s).strip().capitalize() for s in species]
        if positions is None:
            positions = np.empty((0, 3), dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if len(symbols) != positions.shape[0]:
            raise ValueError(
                f"species ({len(symbols)}) and positions ({positions.shape[0]}) "
                "must have the same length."
            )
        cell = lattice.matrix if isinstance(lattice, Lattice) else lattice
        self._atoms = Atoms(
            symbols=symbols, scaled_positions=positions, cell=cell, pbc=True,
        )

    # ------------------------------------------------------------------
    # Underlying ASE handle (escape hatch)
    # ------------------------------------------------------------------
    @property
    def atoms(self) -> Atoms:
        """The underlying ``ase.Atoms`` (mutating it mutates the Structure)."""
        return self._atoms

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------
    @property
    def lattice(self) -> Lattice:
        return Lattice(matrix=np.asarray(self._atoms.cell.array))

    @property
    def species(self) -> List[str]:
        return self.symbols

    @property
    def symbols(self) -> List[str]:
        return list(self._atoms.get_chemical_symbols())

    @property
    def positions(self) -> np.ndarray:
        """Fractional coordinates (no wrapping)."""
        return self._atoms.get_scaled_positions(wrap=False)

    @property
    def cart_positions(self) -> np.ndarray:
        return self._atoms.positions.copy()

    @property
    def num_atoms(self) -> int:
        return len(self._atoms)

    def __len__(self) -> int:
        return len(self._atoms)

    @property
    def volume(self) -> float:
        return float(self._atoms.cell.volume)

    @property
    def total_mass(self) -> float:
        return float(self._atoms.get_masses().sum())

    @property
    def density(self) -> Optional[float]:
        """Mass density (g/cm³).  ``None`` when the cell has zero volume."""
        vol = self.volume
        if vol < 1e-10:
            return None
        return self.total_mass * _AMU_PER_A3_TO_G_PER_CM3 / vol

    @property
    def composition(self) -> Dict[str, int]:
        return dict(Counter(self._atoms.get_chemical_symbols()))

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add_atom(self, element: str, frac_position: Sequence[float]) -> None:
        cart = np.asarray(frac_position, dtype=np.float64) @ self._atoms.cell.array
        self._atoms.append(Atom(str(element).strip().capitalize(), position=cart))

    def remove_atom(self, index: int) -> None:
        del self._atoms[index]

    def wrap_to_cell(self) -> None:
        self._atoms.wrap()

    def copy(self) -> "Structure":
        return Structure(atoms=self._atoms.copy())

    # ------------------------------------------------------------------
    # Geometry / transforms (delegate to ASE)
    # ------------------------------------------------------------------
    def get_distance(self, i: int, j: int, mic: bool = True) -> float:
        return float(self._atoms.get_distance(i, j, mic=mic))

    def supercell(self, na: int, nb: int, nc: int) -> "Structure":
        return Structure(atoms=self._atoms.repeat((na, nb, nc)))

    # ------------------------------------------------------------------
    # Interop
    # ------------------------------------------------------------------
    def to_ase_atoms(self) -> Atoms:
        return self._atoms.copy()

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms) -> "Structure":
        return cls(atoms=atoms.copy())

    def to_pymatgen(self):
        from pymatgen.io.ase import AseAtomsAdaptor
        return AseAtomsAdaptor().get_structure(self._atoms)

    @classmethod
    def from_pymatgen(cls, struct) -> "Structure":
        from pymatgen.io.ase import AseAtomsAdaptor
        return cls(atoms=AseAtomsAdaptor().get_atoms(struct))

    def __repr__(self) -> str:
        comp_str = " ".join(f"{s}{n}" for s, n in self.composition.items())
        vol = self.volume
        vol_str = f"{vol:.2f}" if vol > 1e-10 else "0 (no cell)"
        return f"Structure({comp_str}, natoms={self.num_atoms}, V={vol_str} A^3)"
