"""Conversions between the structure types supported by mckit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from ase import Atoms

from mckit.core.structure import Structure

if TYPE_CHECKING:
    from pymatgen.core import Structure as PmgStructure


StructureLike = Union[Atoms, Structure, "PmgStructure"]


def to_ase_atoms(structure: StructureLike, *, copy: bool = True) -> Atoms:
    """Convert a supported structure to ``ase.Atoms``.

    Copies are returned by default so operations cannot accidentally mutate
    their caller's input.
    """
    if isinstance(structure, Structure):
        atoms = structure.atoms
    elif isinstance(structure, Atoms):
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
                "Expected ase.Atoms, pymatgen Structure, or mckit Structure; "
                f"got {type(structure).__name__}."
            )
        atoms = AseAtomsAdaptor().get_atoms(structure)

    return atoms.copy() if copy else atoms


def to_mckit_structure(structure: StructureLike, *, copy: bool = True) -> Structure:
    """Convert a supported structure to ``mckit.core.Structure``."""
    if isinstance(structure, Structure):
        return structure.copy() if copy else structure
    return Structure(atoms=to_ase_atoms(structure, copy=copy))


def to_pymatgen_structure(structure: StructureLike, *, copy: bool = True):
    """Convert a supported structure to ``pymatgen.core.Structure``."""
    from pymatgen.core import Structure as PmgStructure
    from pymatgen.io.ase import AseAtomsAdaptor

    if isinstance(structure, PmgStructure):
        return structure.copy() if copy else structure
    return AseAtomsAdaptor().get_structure(to_ase_atoms(structure, copy=False))
