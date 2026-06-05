"""Supercell builder — supports diagonal repeats and full 3×3 transformation matrices."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from ase import Atoms
from ase.build import make_supercell

from mmkit.core.structure import Structure
from mmkit.core.tool import Operation

# Type alias for structure-like inputs
StructureLike = Union[Atoms, Structure, "PmgStructure"]

# ---------------------------------------------------------------------------
# Lazy pymatgen import helper
# ---------------------------------------------------------------------------

def _get_pymatgen_types():
    """Import pymatgen types lazily so the module loads without pymatgen."""
    from pymatgen.core.structure import Structure as PmgStructure
    return PmgStructure


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------

def _to_ase_atoms(obj: StructureLike) -> Atoms:
    """Coerce various structure types to ``ase.Atoms``."""
    # mmkit Structure
    if isinstance(obj, Structure):
        return obj.to_ase_atoms()

    # Already ASE Atoms
    if isinstance(obj, Atoms):
        return obj.copy()

    # pymatgen Structure (check by name to avoid hard import)
    try:
        PmgStructure = _get_pymatgen_types()
        if isinstance(obj, PmgStructure):
            from pymatgen.io.ase import AseAtomsAdaptor
            return AseAtomsAdaptor().get_atoms(obj)
    except ImportError:
        pass

    raise TypeError(
        f"Expected Atoms, pymatgen Structure, or mmkit Structure, "
        f"got {type(obj).__name__}"
    )


# ---------------------------------------------------------------------------
# Repeat normalisation
# ---------------------------------------------------------------------------

def _normalise_repeat(
    repeat: Sequence[int] | Sequence[Sequence[int]],
) -> np.ndarray:
    """Convert *repeat* to a 3×3 transformation matrix.

    Accepts either:
    * A length-3 sequence of ints ``[n1, n2, n3]`` → diagonal matrix.
    * A 3×3 nested sequence (row-major) → arbitrary transformation.

    Returns a ``(3, 3)`` numpy array suitable for :func:`ase.build.make_supercell`.
    """
    arr = np.asarray(repeat)

    # Diagonal form: [n1, n2, n3]
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError(
                f"Diagonal repeat must have 3 elements, got {arr.shape[0]}."
            )
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError(
                f"Diagonal repeat values must be integers, got {arr.dtype}."
            )
        return np.diag(arr.astype(int))

    # Full matrix form: [[...], [...], [...]]
    if arr.ndim == 2:
        if arr.shape != (3, 3):
            raise ValueError(
                f"Supercell matrix must be 3×3, got shape {arr.shape}."
            )
        return arr.astype(float)

    raise ValueError(
        "repeat must be a length-3 list of ints [n1, n2, n3] "
        "or a 3×3 nested list [[...], [...], [...]]."
    )


# ---------------------------------------------------------------------------
# SupercellBuilder
# ---------------------------------------------------------------------------

class SupercellBuilder(Operation):
    """Build supercells from an input structure.

    Supports both diagonal repeats (``[2, 2, 2]``) and full 3×3
    transformation matrices (``[[1,1,0],[1,-1,0],[0,0,1]]``).

    Examples
    --------
    >>> builder = SupercellBuilder()
    >>> sc = builder.apply(structure=my_structure, repeat=[2, 2, 2])
    >>> sc = builder.apply(structure=my_structure,
    ...                    repeat=[[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    """

    def apply(
        self,
        *,
        structure: StructureLike,
        repeat: Sequence[int] | Sequence[Sequence[int]],
    ) -> Structure:
        """Build a supercell.

        Parameters
        ----------
        structure :
            Input structure.  Accepts :class:`ase.Atoms`,
            :class:`pymatgen.core.structure.Structure`, or
            :class:`mmkit.core.structure.Structure`.
        repeat :
            Either a 3-element list ``[n1, n2, n3]`` for a diagonal
            repeat, or a 3×3 nested list for an arbitrary supercell
            transformation matrix.

        Returns
        -------
        mmkit.core.structure.Structure
            The supercell structure.
        """
        atoms = _to_ase_atoms(structure)
        matrix = _normalise_repeat(repeat)
        supercell_atoms = make_supercell(atoms, matrix)
        return Structure(atoms=supercell_atoms)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_build(args):
    """CLI handler: build a supercell from a structure file."""
    from pathlib import Path

    from mmkit.io import read_structure, write_atoms

    atoms = read_structure(args.input)

    if args.matrix is not None:
        repeat: list = [
            [args.matrix[0], args.matrix[1], args.matrix[2]],
            [args.matrix[3], args.matrix[4], args.matrix[5]],
            [args.matrix[6], args.matrix[7], args.matrix[8]],
        ]
    else:
        repeat = list(args.repeat)

    builder = SupercellBuilder()
    result = builder.apply(structure=atoms, repeat=repeat)

    stem = Path(args.input).stem
    output = args.output or f"supercell_{stem}.extxyz"
    path = write_atoms(output, result.atoms)
    print(
        f"Supercell: {len(atoms)} -> {len(result.atoms)} atoms  "
        f"({result.lattice.volume:.2f} A^3)  -> {path}"
    )


def register_cli(subparsers) -> None:
    """Register supercell subcommands with the mmkit CLI."""
    sc = subparsers.add_parser(
        "supercell",
        help="Build supercells (diagonal or matrix form)",
    )
    sc.add_argument(
        "input",
        help="Input structure file (CIF, POSCAR, extxyz, ...)",
    )

    mode = sc.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--repeat", "-r",
        type=int, nargs=3, metavar=("N1", "N2", "N3"),
        help="Diagonal repeat counts, e.g. 2 2 2",
    )
    mode.add_argument(
        "--matrix", "-m",
        type=float, nargs=9, metavar="M",
        help="3x3 supercell matrix (row-major), e.g. 2 0 0  0 2 0  0 0 1",
    )

    sc.add_argument(
        "--output", "-o",
        help="Output file (default: supercell_<input>.extxyz)",
    )
    sc.set_defaults(handler=_cmd_build)
