"""Conversions between the structure types supported by mckit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from ase import Atoms

if TYPE_CHECKING:
    from pymatgen.core import Structure as PmgStructure


StructureLike = Union[Atoms, "PmgStructure"]


def to_ase_atoms(structure: StructureLike, *, copy: bool = True) -> Atoms:
    """Convert a supported structure to ``ase.Atoms``.

    Copies are returned by default so operations cannot accidentally mutate
    their caller's input.
    """
    if isinstance(structure, Atoms):
        atoms = structure
    else:
        try:
            from pymatgen.core import Structure as PmgStructure
            from pymatgen.io.ase import AseAtomsAdaptor
        except ImportError as exc:
            raise TypeError(
                "pymatgen is required to convert this structure type."
            ) from exc

        if not isinstance(structure, PmgStructure):
            raise TypeError(
                "Expected ase.Atoms or pymatgen Structure; "
                f"got {type(structure).__name__}."
            )
        atoms = AseAtomsAdaptor().get_atoms(structure)

    return atoms.copy() if copy else atoms


def to_pymatgen_structure(structure: StructureLike, *, copy: bool = True):
    """Convert a supported structure to ``pymatgen.core.Structure``."""
    from pymatgen.core import Structure as PmgStructure
    from pymatgen.io.ase import AseAtomsAdaptor

    if isinstance(structure, PmgStructure):
        return structure.copy() if copy else structure
    return AseAtomsAdaptor().get_structure(to_ase_atoms(structure, copy=False))
