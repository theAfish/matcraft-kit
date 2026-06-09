"""Write atomic structures to files via ``ase.io.write``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ase.io import write as ase_write

from mckit.core.conversion import StructureLike, to_ase_atoms


def write_structure(
    path: str,
    structure: StructureLike,
    format: Optional[str] = None,
    **kwargs,
) -> str:
    """Write a ``Structure`` (or raw ``ase.Atoms`` / pymatgen ``Structure``) to a file.

    The output format is auto-detected from the extension when ``format`` is
    omitted; missing extensions default to ``.extxyz``.
    """
    atoms = to_ase_atoms(structure, copy=False)
    p = Path(path)
    if p.suffix == "":
        p = p.with_suffix(".extxyz")
    ase_write(str(p), atoms, format=format, **kwargs)
    return str(p)


# Back-compat alias used by ``operations.surface``.
write_atoms = write_structure
