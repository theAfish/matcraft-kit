"""Write atomic structures to files via ``ase.io.write``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ase import Atoms
from ase.io import write as ase_write

from mckit.core.structure import Structure


def write_structure(
    path: str,
    structure: Union[Structure, Atoms, "PmgStructure"],
    format: Optional[str] = None,
    **kwargs,
) -> str:
    """Write a ``Structure`` (or raw ``ase.Atoms`` / pymatgen ``Structure``) to a file.

    The output format is auto-detected from the extension when ``format`` is
    omitted; missing extensions default to ``.extxyz``.
    """
    if isinstance(structure, Structure):
        atoms = structure.to_ase_atoms()
    elif isinstance(structure, Atoms):
        atoms = structure
    else:
        # Assume pymatgen Structure (or compatible)
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms = AseAtomsAdaptor().get_atoms(structure)
    p = Path(path)
    if p.suffix == "":
        p = p.with_suffix(".extxyz")
    ase_write(str(p), atoms, format=format, **kwargs)
    return str(p)


# Back-compat alias used by ``operations.surface``.
write_atoms = write_structure
