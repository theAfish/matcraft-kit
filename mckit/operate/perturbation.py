"""Perturbation builder — apply random displacements to atomic positions and lattice vectors."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from ase import Atoms

from mckit.core.structure import Structure
from mckit.core.tool import Operation

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
# Perturbation modes
# ---------------------------------------------------------------------------

_PERTURBATION_MODES = ("random", "gaussian", "scaled_random")
_CELL_PERTURBATION_MODES = ("tri", "aniso", "iso")


def _perturb_cell(
    cell: np.ndarray,
    magnitude: float,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a perturbed 3×3 cell matrix.

    Parameters
    ----------
    cell :
        Original 3×3 cell matrix (rows are lattice vectors).
    magnitude :
        Perturbation scale.  For ``"tri"`` this is in angstroms
        (absolute displacement of each lattice-vector component).
        For ``"aniso"`` and ``"iso"`` this is a dimensionless
        fractional change (e.g. 0.01 ≈ 1% length change).
    mode :
        Cell perturbation type — one of:

        * ``"tri"`` (default): add random Cartesian displacement to each
          lattice vector.  An orthogonal box becomes triclinic — all
          angles and lengths change.
        * ``"aniso"``: rescale each lattice vector's length
          independently, preserving its direction.  Angles stay fixed,
          lengths change independently (a ≠ b ≠ c in general).
        * ``"iso"``: uniform scaling of all lattice vectors by the same
          random factor.  Only the volume changes; shape is preserved.
    rng :
        NumPy random generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        A perturbed 3×3 cell matrix.
    """
    if mode == "tri":
        # Add random Cartesian displacement to each lattice vector (Å)
        perturbation = rng.uniform(-magnitude, magnitude, (3, 3))
        return cell + perturbation

    elif mode == "aniso":
        # Rescale each lattice vector's length, preserve direction
        # scale_i in [1 - magnitude, 1 + magnitude]
        scales = 1.0 + rng.uniform(-magnitude, magnitude, 3)
        return cell * scales[:, np.newaxis]

    elif mode == "iso":
        # Uniform scaling of all lattice vectors
        scale = 1.0 + rng.uniform(-magnitude, magnitude)
        return cell * scale

    else:
        raise ValueError(
            f"Unknown cell perturbation mode {mode!r}. "
            f"Choose from {_CELL_PERTURBATION_MODES}"
        )


def _generate_displacements(
    atoms: Atoms,
    indices: Sequence[int],
    magnitude: float,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a ``(len(atoms), 3)`` displacement array for the selected atoms.

    Parameters
    ----------
    atoms :
        The full atomic structure (used for covalent radii in ``scaled_random``).
    indices :
        Atom indices to perturb.
    magnitude :
        Displacement scale parameter (interpretation depends on *mode*).
    mode :
        One of ``"random"``, ``"gaussian"``, or ``"scaled_random"``.
    rng :
        NumPy random generator for reproducibility.
    """
    n_atoms = len(atoms)
    displacements = np.zeros((n_atoms, 3))

    if mode == "random":
        displacements[indices] = rng.uniform(
            -magnitude, magnitude, (len(indices), 3),
        )

    elif mode == "gaussian":
        displacements[indices] = rng.normal(
            0.0, magnitude, (len(indices), 3),
        )

    elif mode == "scaled_random":
        from ase.data import covalent_radii
        for idx in indices:
            radius = covalent_radii[atoms[idx].number]
            scale = radius if radius > 0.0 else 1.0
            displacements[idx] = rng.uniform(
                -magnitude * scale, magnitude * scale, 3,
            )

    else:
        raise ValueError(
            f"Unknown perturbation mode {mode!r}. "
            f"Choose from {_PERTURBATION_MODES}"
        )

    return displacements


# ---------------------------------------------------------------------------
# PerturbationBuilder
# ---------------------------------------------------------------------------

class PerturbationBuilder(Operation):
    """Perturb atomic positions and lattice vectors with random displacements.

    Supports uniform-random, Gaussian, and covalent-radii-scaled perturbation
    modes for atomic positions.  The unit cell can also be perturbed via
    three modes — ``"tri"`` (full Cartesian displacement, breaks all
    symmetry), ``"aniso"`` (rescale axis lengths independently, preserve
    angles), and ``"iso"`` (uniform volume scaling, preserve shape).

    When the cell is perturbed, atomic positions are rescaled with it so
    that fractional coordinates are preserved (atoms move with the
    lattice).

    Examples
    --------
    >>> builder = PerturbationBuilder()
    >>> perturbed = builder.apply(structure=my_atoms, magnitude=0.1, mode="gaussian")
    >>> # Perturb both positions and cell:
    >>> perturbed = builder.apply(
    ...     structure=my_atoms, magnitude=0.1, cell_magnitude=0.01,
    ... )
    """

    def apply(
        self,
        *,
        structure: StructureLike,
        magnitude: float = 0.1,
        mode: str = "random",
        atom_indices: Optional[Sequence[int]] = None,
        cell_magnitude: Optional[float] = None,
        cell_mode: str = "tri",
        seed: Optional[int] = None,
    ) -> Structure:
        """Apply random displacements to atomic positions and/or cell vectors.

        Parameters
        ----------
        structure :
            Input structure.  Accepts :class:`ase.Atoms`,
            :class:`pymatgen.core.structure.Structure`, or
            :class:`mmkit.core.structure.Structure`.
        magnitude :
            Maximum displacement in angstroms (default ``0.1``).
            For ``"gaussian"`` mode this is the standard deviation.
        mode :
            Perturbation type — one of ``"random"`` (uniform in
            ``[-magnitude, +magnitude]``), ``"gaussian"`` (normal
            distribution with *std* = magnitude), or ``"scaled_random"``
            (uniform random scaled by each atom's covalent radius).
        atom_indices :
            Indices of atoms to perturb.  ``None`` (default) perturbs
            **all** atoms.
        cell_magnitude :
            Perturbation scale for cell vectors.  For ``"tri"`` mode
            this is in angstroms (same units as *magnitude*).  For
            ``"aniso"`` and ``"iso"`` this is a dimensionless
            fractional change (e.g. 0.01 ≈ 1% length change).
            ``None`` (default) leaves the cell unchanged.
        cell_mode :
            Cell perturbation type — one of:

            * ``"tri"`` (default): add random Cartesian displacement to
              each lattice vector.  An orthogonal box becomes triclinic —
              all angles and lengths change.
            * ``"aniso"``: rescale each lattice vector's length
              independently, preserving its direction.  Angles stay
              fixed, lengths change independently.
            * ``"iso"``: uniform scaling of all lattice vectors by the
              same random factor.  Only volume changes; shape preserved.

            When the cell is perturbed, atomic positions are rescaled
            (fractional coordinates preserved).
        seed :
            Random seed for reproducibility (optional).

        Returns
        -------
        mmkit.core.structure.Structure
            A new structure with displaced atomic positions and/or cell vectors.
        """
        atoms = _to_ase_atoms(structure)

        # Validate mode early
        if mode not in _PERTURBATION_MODES:
            raise ValueError(
                f"Unknown perturbation mode {mode!r}. "
                f"Choose from {_PERTURBATION_MODES}"
            )

        # Validate and resolve atom indices
        n_atoms = len(atoms)
        if atom_indices is None:
            indices = list(range(n_atoms))
        else:
            indices = list(atom_indices)
            for idx in indices:
                if idx < 0 or idx >= n_atoms:
                    raise IndexError(
                        f"Atom index {idx} out of range for structure "
                        f"with {n_atoms} atoms"
                    )

        rng = np.random.default_rng(seed)

        # Perturb atomic positions
        displacements = _generate_displacements(atoms, indices, magnitude, mode, rng)
        atoms.positions += displacements

        # Perturb cell if requested (rescale atom positions with the cell)
        if cell_magnitude is not None:
            if cell_mode not in _CELL_PERTURBATION_MODES:
                raise ValueError(
                    f"Unknown cell perturbation mode {cell_mode!r}. "
                    f"Choose from {_CELL_PERTURBATION_MODES}"
                )
            perturbed_cell = _perturb_cell(atoms.cell, cell_magnitude, cell_mode, rng)
            atoms.set_cell(perturbed_cell, scale_atoms=True)

        return Structure(atoms=atoms)


# ---------------------------------------------------------------------------
# BatchPerturbationBuilder
# ---------------------------------------------------------------------------

class BatchPerturbationBuilder(Operation):
    """Generate multiple independently perturbed structures.

    Useful for creating training datasets or sampling diverse starting
    configurations from a single input structure.  Both atomic positions
    and lattice vectors can be perturbed.

    When the cell is perturbed, atomic positions are rescaled with it so
    that fractional coordinates are preserved (atoms move with the
    lattice).

    Examples
    --------
    >>> builder = BatchPerturbationBuilder()
    >>> results = builder.apply(
    ...     structure=my_atoms, num_structures=10, magnitude=0.1, seed=42,
    ... )
    >>> len(results)
    10
    >>> # Perturb both positions and cell:
    >>> results = builder.apply(
    ...     structure=my_atoms, num_structures=10,
    ...     magnitude=0.1, cell_magnitude=0.01, seed=42,
    ... )
    """

    def apply(
        self,
        *,
        structure: StructureLike,
        num_structures: int = 10,
        magnitude: float = 0.1,
        mode: str = "random",
        atom_indices: Optional[Sequence[int]] = None,
        cell_magnitude: Optional[float] = None,
        cell_mode: str = "tri",
        seed: Optional[int] = None,
    ) -> list[Structure]:
        """Generate a batch of perturbed structures.

        Parameters
        ----------
        structure :
            Input structure.  Accepts :class:`ase.Atoms`,
            :class:`pymatgen.core.structure.Structure`, or
            :class:`mmkit.core.structure.Structure`.
        num_structures :
            Number of perturbed copies to generate (default ``10``).
        magnitude :
            Maximum displacement in angstroms (default ``0.1``).
        mode :
            Perturbation type — see :class:`PerturbationBuilder`.
        atom_indices :
            Indices of atoms to perturb.  ``None`` perturbs all atoms.
        cell_magnitude :
            Perturbation scale for cell vectors.  For ``"tri"`` mode
            this is in angstroms.  For ``"aniso"`` and ``"iso"`` this is
            a dimensionless fractional change.  ``None`` (default)
            leaves the cell unchanged.
        cell_mode :
            Cell perturbation type — see :class:`PerturbationBuilder`.
        seed :
            Base random seed.  Each structure uses ``seed + i`` so that
            results are reproducible yet independent (optional).

        Returns
        -------
        list[mmkit.core.structure.Structure]
            A list of perturbed structures, one per requested copy.
        """
        atoms = _to_ase_atoms(structure)

        if mode not in _PERTURBATION_MODES:
            raise ValueError(
                f"Unknown perturbation mode {mode!r}. "
                f"Choose from {_PERTURBATION_MODES}"
            )

        if cell_magnitude is not None and cell_mode not in _CELL_PERTURBATION_MODES:
            raise ValueError(
                f"Unknown cell perturbation mode {cell_mode!r}. "
                f"Choose from {_CELL_PERTURBATION_MODES}"
            )

        n_atoms = len(atoms)
        if atom_indices is None:
            indices = list(range(n_atoms))
        else:
            indices = list(atom_indices)
            for idx in indices:
                if idx < 0 or idx >= n_atoms:
                    raise IndexError(
                        f"Atom index {idx} out of range for structure "
                        f"with {n_atoms} atoms"
                    )

        results: list[Structure] = []
        for i in range(num_structures):
            child_seed = None if seed is None else seed + i
            rng = np.random.default_rng(child_seed)

            perturbed = atoms.copy()

            # Perturb atomic positions
            displacements = _generate_displacements(
                perturbed, indices, magnitude, mode, rng,
            )
            perturbed.positions += displacements

            # Perturb cell if requested (rescale atom positions with the cell)
            if cell_magnitude is not None:
                perturbed_cell = _perturb_cell(perturbed.cell, cell_magnitude, cell_mode, rng)
                perturbed.set_cell(perturbed_cell, scale_atoms=True)

            results.append(Structure(atoms=perturbed))

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_perturb(args):
    """CLI handler: perturb a structure and write the result."""
    from pathlib import Path

    from mckit.io import read_structure, write_structure

    atoms = read_structure(args.input)

    builder = PerturbationBuilder()
    result = builder.apply(
        structure=atoms,
        magnitude=args.magnitude,
        mode=args.mode,
        atom_indices=args.indices,
        cell_magnitude=args.cell_magnitude,
        cell_mode=args.cell_mode,
        seed=args.seed,
    )

    stem = Path(args.input).stem
    output = args.output or f"perturbed_{stem}.extxyz"
    path = write_structure(output, result.atoms)

    cell_info = f", cell={args.cell_magnitude}" if args.cell_magnitude is not None else ""
    print(
        f"Perturbed {len(atoms)} atoms  "
        f"(mode={args.mode}, magnitude={args.magnitude} A{cell_info})  -> {path}"
    )


def _cmd_batch(args):
    """CLI handler: generate a batch of perturbed structures."""
    from pathlib import Path

    from mckit.io import read_structure, write_structure

    atoms = read_structure(args.input)

    builder = BatchPerturbationBuilder()
    results = builder.apply(
        structure=atoms,
        num_structures=args.num,
        magnitude=args.magnitude,
        mode=args.mode,
        atom_indices=args.indices,
        cell_magnitude=args.cell_magnitude,
        cell_mode=args.cell_mode,
        seed=args.seed,
    )

    stem = Path(args.input).stem
    prefix = args.output or f"perturbed_{stem}"
    paths: list[str] = []
    for i, struct in enumerate(results):
        path = write_structure(f"{prefix}_{i:04d}.extxyz", struct.atoms)
        paths.append(path)

    cell_info = f", cell={args.cell_magnitude}" if args.cell_magnitude is not None else ""
    print(
        f"Generated {len(results)} perturbed structures from "
        f"{len(atoms)} atoms  (mode={args.mode}{cell_info})  -> {prefix}_*.extxyz"
    )


def register_cli(subparsers) -> None:
    """Register perturbation subcommands with the mmkit CLI."""
    # -- single perturbation -------------------------------------------------
    p = subparsers.add_parser(
        "perturb",
        help="Perturb atomic positions and/or lattice vectors with random displacements",
    )
    p.add_argument(
        "input",
        help="Input structure file (CIF, POSCAR, extxyz, ...)",
    )
    p.add_argument(
        "--magnitude", "-m",
        type=float, default=0.1,
        help="Maximum displacement in angstroms (default: 0.1)",
    )
    p.add_argument(
        "--mode",
        choices=_PERTURBATION_MODES, default="random",
        help="Perturbation mode (default: random)",
    )
    p.add_argument(
        "--indices",
        type=int, nargs="*", default=None,
        help="Atom indices to perturb (default: all atoms)",
    )
    p.add_argument(
        "--cell-magnitude",
        type=float, default=None,
        help="Perturbation scale for cell vectors (default: disabled). "
             "In angstroms for --cell-mode=tri, fractional for aniso/iso",
    )
    p.add_argument(
        "--cell-mode",
        choices=_CELL_PERTURBATION_MODES, default="tri",
        help="Cell perturbation mode (default: tri — breaks all symmetry)",
    )
    p.add_argument(
        "--seed", "-s",
        type=int, default=None,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--output", "-o",
        help="Output file (default: perturbed_<input>.extxyz)",
    )
    p.set_defaults(handler=_cmd_perturb)

    # -- batch perturbation --------------------------------------------------
    b = subparsers.add_parser(
        "batch-perturb",
        help="Generate multiple perturbed structures from one input (positions and/or cell)",
    )
    b.add_argument(
        "input",
        help="Input structure file (CIF, POSCAR, extxyz, ...)",
    )
    b.add_argument(
        "--num", "-n",
        type=int, default=10,
        help="Number of structures to generate (default: 10)",
    )
    b.add_argument(
        "--magnitude", "-m",
        type=float, default=0.1,
        help="Maximum displacement in angstroms (default: 0.1)",
    )
    b.add_argument(
        "--mode",
        choices=_PERTURBATION_MODES, default="random",
        help="Perturbation mode (default: random)",
    )
    b.add_argument(
        "--indices",
        type=int, nargs="*", default=None,
        help="Atom indices to perturb (default: all atoms)",
    )
    b.add_argument(
        "--cell-magnitude",
        type=float, default=None,
        help="Perturbation scale for cell vectors (default: disabled). "
             "In angstroms for --cell-mode=tri, fractional for aniso/iso",
    )
    b.add_argument(
        "--cell-mode",
        choices=_CELL_PERTURBATION_MODES, default="tri",
        help="Cell perturbation mode (default: tri — breaks all symmetry)",
    )
    b.add_argument(
        "--seed", "-s",
        type=int, default=None,
        help="Base random seed (each structure uses seed + i)",
    )
    b.add_argument(
        "--output", "-o",
        help="Output prefix (default: perturbed_<input>_0000.extxyz, ...)",
    )
    b.set_defaults(handler=_cmd_batch)
