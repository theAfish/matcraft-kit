"""Read structures as ``ase.Atoms`` using ASE (and pymatgen for CIF)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ase import Atoms
from ase.io import read as ase_read


def read_structure(path: str, format: Optional[str] = None, **kwargs) -> Atoms:
    """Read a structure file as ``ase.Atoms``.

    Supports all formats that ASE supports (extxyz, vasp, cif, xyz, poscar, ...)
    plus CIF via pymatgen for better handling of symmetry.

    Parameters
    ----------
    path : str
        File path.
    format : str, optional
        ASE format hint (e.g. ``"vasp"``, ``"cif"``). Auto-detected if omitted.
    **kwargs
        Extra keyword arguments forwarded to ``ase.io.read``.

    Returns
    -------
    ase.Atoms
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Use pymatgen for CIF (better symmetry handling)
    if p.suffix.lower() == ".cif" and format is None:
        from pymatgen.io.cif import CifParser

        parser = CifParser(str(p))
        structs = parser.parse_structures(primitive=False)
        if not structs:
            raise ValueError(f"No structures parsed from {path}")
        from pymatgen.io.ase import AseAtomsAdaptor

        return AseAtomsAdaptor().get_atoms(structs[0])

    return ase_read(str(p), format=format, **kwargs)
