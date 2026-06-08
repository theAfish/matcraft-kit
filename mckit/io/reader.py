"""Read structures as ``ase.Atoms`` using ASE (and pymatgen for CIF)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ase import Atoms
from ase.io import read as ase_read


def _read_xyz_xysz(path: Path) -> Atoms:
    """Read an XYZ file in ``x y z Z`` format (positions first, atomic number last).

    This is a non-standard variant sometimes produced by custom codes.
    """
    from ase.data import chemical_symbols

    with open(path) as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError(f"XYZ file too short: {path}")

    natoms = int(lines[0].strip())
    comment = lines[1].strip()

    numbers = []
    positions = []
    for i, line in enumerate(lines[2:2 + natoms], start=3):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(
                f"Line {i} in {path}: expected at least 4 columns "
                f"(x y z Z), got {len(parts)}"
            )
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        z_num = int(round(float(parts[3])))
        numbers.append(z_num)
        positions.append([x, y, z])

    return Atoms(numbers=numbers, positions=positions)


def read_structure(path: str, format: Optional[str] = None, **kwargs) -> Atoms:
    """Read a structure file as ``ase.Atoms``.

    Supports all formats that ASE supports (extxyz, vasp, cif, xyz, poscar, ...)
    plus CIF via pymatgen for better handling of symmetry.  Also handles a
    non-standard XYZ variant where columns are ``x y z Z`` (positions first,
    atomic number last) instead of the standard ``symbol x y z``.

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

    try:
        return ase_read(str(p), format=format, **kwargs)
    except (KeyError, ValueError) as exc:
        # Fallback: try non-standard XYZ format (x y z Z)
        is_xyz = p.suffix.lower() in (".xyz", ".extxyz") or format in ("xyz", "extxyz")
        if is_xyz and isinstance(exc, (KeyError, ValueError)):
            try:
                return _read_xyz_xysz(p)
            except Exception:
                pass  # Fall through to original error
        raise
