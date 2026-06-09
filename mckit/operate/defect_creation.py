"""Defect creation — point defects in bulk crystals.

Create symmetry-unique point defects using pymatgen's defect generators:

- :class:`VacancyCreator` — remove one atom at a time from each
  symmetry-unique site.
- :class:`SubstitutionCreator` — replace a host species with a dopant.
- :class:`AntiSiteCreator` — swap two species (A on B site / B on A site).
- :class:`InterstitialCreator` — insert an atom at a Voronoi-derived
  interstitial site.

Each creator enumerates all symmetry-inequivalent defects for the input
structure.  Use :meth:`list_defects` to see what is available, then pass
``index`` to :meth:`apply` to select which one to create.

Requires the ``pymatgen-analysis-defects`` package::

    pip install pymatgen-analysis-defects

CLI examples::

    mmkit operate defect vacancy list bulk.cif
    mmkit operate defect vacancy create bulk.cif --index 0 -o vac.extxyz
    mmkit operate defect substitution create bulk.cif --sub "Ga=Zn" --index 0
    mmkit operate defect antisite list bulk.cif
    mmkit operate defect interstitial create bulk.cif --species Li --index 0
"""

from __future__ import annotations

from abc import abstractmethod
import random
from typing import Union
import numpy as np

from ase import Atoms

from mckit.core.conversion import StructureLike, to_pymatgen_structure
from mckit.core.tool import Operation


# Type alias for flexible structure input


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _get_pymatgen_types():
    """Import pymatgen types lazily so the module loads without pymatgen."""
    from pymatgen.core.structure import Structure as PmgStructure
    return PmgStructure


def _get_defect_generators():
    """Import defect generators (requires *pymatgen-analysis-defects*)."""
    try:
        from pymatgen.analysis.defects.generators import (
            AntiSiteGenerator,
            SubstitutionGenerator,
            VacancyGenerator,
            VoronoiInterstitialGenerator,
        )
    except ImportError as exc:
        raise ImportError(
            "The 'pymatgen-analysis-defects' package is required for defect "
            "creation.  Install it with:  pip install pymatgen-analysis-defects"
        ) from exc
    return (
        VacancyGenerator,
        SubstitutionGenerator,
        AntiSiteGenerator,
        VoronoiInterstitialGenerator,
    )


def _defect_to_atoms(defect) -> Atoms:
    """Convert a pymatgen defect's ``defect_structure`` to ``ase.Atoms``."""
    from pymatgen.io.ase import AseAtomsAdaptor

    dstruct = defect.defect_structure
    try:
        dstruct = dstruct.clone()
    except AttributeError:
        dstruct = dstruct.copy()
    return AseAtomsAdaptor().get_atoms(dstruct)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class _DefectCreator(Operation):
    """Base class for point-defect creators.

    Subclasses provide a :meth:`_generate_defects` method that returns a
    list of pymatgen defect objects.  :meth:`apply` selects one by *index*
    and returns the modified structure as :class:`ase.Atoms`.
    """

    # -- public API ---------------------------------------------------------

    def list_defects(self, structure: StructureLike, **kwargs) -> list[str]:
        """Return defect names for every symmetry-unique site.

        Use this before calling :meth:`apply` to decide which ``index``
        to pass.
        """
        pmg = to_pymatgen_structure(structure)
        defects = self._generate_defects(pmg, **kwargs)
        return [d.name for d in defects]

    def apply(self, *, structure: StructureLike, index: int = 0,
              **kwargs) -> Atoms:
        """Create a single point defect and return the result.

        Parameters
        ----------
        structure :
            Input bulk structure.
        index :
            Index into the list returned by :meth:`list_defects`.
            Defaults to ``0`` (first symmetry-unique defect).
        **kwargs :
            Subclass-specific parameters (e.g. *substitution*, *species*).
        """
        pmg = to_pymatgen_structure(structure)
        defects = self._generate_defects(pmg, **kwargs)

        if not defects:
            raise ValueError(
                f"{type(self).__name__}: no defects generated.  "
                "Check the input structure and parameters."
            )
        if index < 0 or index >= len(defects):
            raise IndexError(
                f"{type(self).__name__}: index {index} out of range "
                f"[0, {len(defects) - 1}].  "
                f"Use list_defects() to see available defects."
            )

        selected = defects[index]
        return _defect_to_atoms(selected)

    @abstractmethod
    def _generate_defects(self, pmg_structure, **kwargs):
        """Return a list of pymatgen defect objects."""


# ---------------------------------------------------------------------------
# VacancyCreator
# ---------------------------------------------------------------------------

class VacancyCreator(_DefectCreator):
    """Create vacancy defects by removing atoms from symmetry-unique sites.

    Uses :class:`pymatgen.analysis.defects.generators.VacancyGenerator`.

    Parameters
    ----------
    symprec : float
        Symmetry precision for space-group analysis (default 0.01).
    angle_tolerance : float
        Angle tolerance in degrees (default 5).

    Examples
    --------
    >>> creator = VacancyCreator()
    >>> creator.list_defects(bulk)          # ['v_Ga', 'v_As']
    >>> vac = creator.apply(structure=bulk, index=0)
    """

    def __init__(self, symprec: float = 0.01,
                 angle_tolerance: float = 5) -> None:
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

    def _generate_defects(self, pmg_structure):
        VacancyGenerator = _get_defect_generators()[0]
        gen = VacancyGenerator(
            symprec=self.symprec,
            angle_tolerance=self.angle_tolerance,
        )
        return gen.get_defects(pmg_structure)

    def __repr__(self) -> str:
        return (
            f"VacancyCreator(symprec={self.symprec}, "
            f"angle_tolerance={self.angle_tolerance})"
        )


# ---------------------------------------------------------------------------
# SubstitutionCreator
# ---------------------------------------------------------------------------

class SubstitutionCreator(_DefectCreator):
    """Create substitution (dopant) defects.

    Uses :class:`pymatgen.analysis.defects.generators.SubstitutionGenerator`.

    Parameters
    ----------
    symprec : float
        Symmetry precision (default 0.01).
    angle_tolerance : float
        Angle tolerance in degrees (default 5).

    Examples
    --------
    >>> creator = SubstitutionCreator()
    >>> creator.list_defects(bulk, substitution={"Ga": "Zn"})
    ['Zn_Ga']
    >>> sub = creator.apply(structure=bulk, substitution={"Ga": "Zn"}, index=0)
    """

    def __init__(self, symprec: float = 0.01,
                 angle_tolerance: float = 5) -> None:
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

    def _generate_defects(
        self, pmg_structure, *,
        substitution: dict[str, Union[str, list[str]]],
    ):
        SubstitutionGenerator = _get_defect_generators()[1]
        gen = SubstitutionGenerator(
            symprec=self.symprec,
            angle_tolerance=self.angle_tolerance,
        )
        return gen.get_defects(pmg_structure, substitution)

    def apply(self, *, structure: StructureLike,
              substitution: dict[str, Union[str, list[str]]],
              index: int = 0) -> Atoms:
        """Create a substitution defect.

        Parameters
        ----------
        structure :
            Input bulk structure.
        substitution :
            Mapping of host element to dopant element(s),
            e.g. ``{"Ga": "Zn"}`` or ``{"Ga": ["Zn", "Mg"]}``.
        index :
            Index into the symmetry-unique defect list.
        """
        return super().apply(
            structure=structure, index=index, substitution=substitution,
        )

    def list_defects(self, structure: StructureLike, *,
                     substitution: dict[str, Union[str, list[str]]],
                     ) -> list[str]:
        """List available substitution defects.

        Parameters
        ----------
        structure :
            Input bulk structure.
        substitution :
            Mapping of host element to dopant element(s).
        """
        pmg = to_pymatgen_structure(structure)
        defects = self._generate_defects(pmg, substitution=substitution)
        return [d.name for d in defects]

    def __repr__(self) -> str:
        return (
            f"SubstitutionCreator(symprec={self.symprec}, "
            f"angle_tolerance={self.angle_tolerance})"
        )


# ---------------------------------------------------------------------------
# AntiSiteCreator
# ---------------------------------------------------------------------------

class AntiSiteCreator(_DefectCreator):
    """Create anti-site defects (species swaps).

    Automatically generates all anti-site pairs for every pair of elements
    in the structure.  Uses
    :class:`pymatgen.analysis.defects.generators.AntiSiteGenerator`.

    .. note::
       Anti-site defects are represented as ``Substitution`` objects
       internally (e.g. ``Ga_As`` means Ga on an As site).

    Parameters
    ----------
    symprec : float
        Symmetry precision (default 0.01).
    angle_tolerance : float
        Angle tolerance in degrees (default 5).

    Examples
    --------
    >>> creator = AntiSiteCreator()
    >>> creator.list_defects(gaas_bulk)   # ['Ga_As', 'As_Ga']
    >>> anti = creator.apply(structure=gaas_bulk, index=0)
    """

    def __init__(self, symprec: float = 0.01,
                 angle_tolerance: float = 5) -> None:
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

    def _generate_defects(self, pmg_structure):
        AntiSiteGenerator = _get_defect_generators()[2]
        gen = AntiSiteGenerator(
            symprec=self.symprec,
            angle_tolerance=self.angle_tolerance,
        )
        return gen.get_defects(pmg_structure)

    def __repr__(self) -> str:
        return (
            f"AntiSiteCreator(symprec={self.symprec}, "
            f"angle_tolerance={self.angle_tolerance})"
        )


# ---------------------------------------------------------------------------
# InterstitialCreator
# ---------------------------------------------------------------------------

class InterstitialCreator(_DefectCreator):
    """Create interstitial defects at Voronoi-derived sites.

    Uses Voronoi decomposition of the structure to find candidate
    interstitial positions, then inserts the requested species there.
    Uses :class:`pymatgen.analysis.defects.generators.VoronoiInterstitialGenerator`.

    Parameters
    ----------
    clustering_tol : float
        Tolerance for clustering nearby Voronoi nodes (default 0.5 A).
    min_dist : float
        Minimum distance between the interstitial and existing atoms
        (default 0.9 A).
    ltol : float
        Length tolerance for symmetry matching (default 0.2).
    stol : float
        Site tolerance for symmetry matching (default 0.3).
    angle_tol : float
        Angle tolerance in degrees (default 5).

    Examples
    --------
    >>> creator = InterstitialCreator()
    >>> creator.list_defects(bulk, species="Li")   # ['Li_i', ...]
    >>> inter = creator.apply(structure=bulk, species="Li", index=0)
    """

    def __init__(
        self,
        clustering_tol: float = 0.5,
        min_dist: float = 0.9,
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5,
    ) -> None:
        self.clustering_tol = clustering_tol
        self.min_dist = min_dist
        self.ltol = ltol
        self.stol = stol
        self.angle_tol = angle_tol

    def _generate_defects(
        self, pmg_structure, *,
        species: Union[str, list[str]],
    ):
        VoronoiInterstitialGenerator = _get_defect_generators()[3]
        gen = VoronoiInterstitialGenerator(
            clustering_tol=self.clustering_tol,
            min_dist=self.min_dist,
            ltol=self.ltol,
            stol=self.stol,
            angle_tol=self.angle_tol,
        )
        insert = self._parse_species(species)
        return gen.get_defects(pmg_structure, insert)

    def apply(self, *, structure: StructureLike,
              species: Union[str, list[str]],
              index: int = 0) -> Atoms:
        """Create an interstitial defect.

        Parameters
        ----------
        structure :
            Input bulk structure.
        species :
            Element symbol(s) to insert, e.g. ``"Li"`` or ``["Li", "Na"]``.
        index :
            Index into the symmetry-unique interstitial list.
        """
        return super().apply(
            structure=structure, index=index, species=species,
        )

    def list_defects(self, structure: StructureLike, *,
                     species: Union[str, list[str]]) -> list[str]:
        """List available interstitial sites.

        Parameters
        ----------
        structure :
            Input bulk structure.
        species :
            Element symbol(s) to consider inserting.
        """
        pmg = to_pymatgen_structure(structure)
        defects = self._generate_defects(pmg, species=species)
        return [d.name for d in defects]

    @staticmethod
    def _parse_species(species) -> list[str]:
        if isinstance(species, str):
            return [species]
        if isinstance(species, (list, tuple, set)):
            return list(species)
        raise TypeError(
            f"species must be a string or list of strings, got {type(species)}"
        )

    def __repr__(self) -> str:
        return (
            f"InterstitialCreator(min_dist={self.min_dist}, "
            f"clustering_tol={self.clustering_tol})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_input(path_or_atoms) -> Atoms:
    """Resolve an input to ``ase.Atoms`` (from file path or Atoms object)."""
    if isinstance(path_or_atoms, str):
        from mckit.io import read_structure
        return read_structure(path_or_atoms)
    if isinstance(path_or_atoms, Atoms):
        return path_or_atoms
    raise TypeError(f"Cannot resolve input of type {type(path_or_atoms)}")


def _parse_substitution_arg(spec: str) -> dict[str, list[str]]:
    """Parse substitution spec like "Ga=Zn" or "Ga=Zn/Mg"."""
    parts = spec.split("=")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            'Substitution must be "Host=Dopant", e.g. "Ga=Zn" '
            'or "Ga=Zn/Mg" for multiple dopants.'
        )
    return {parts[0]: parts[1].split("/")}


def _resolve_target_count(args, n_atoms: int) -> int:
    """Resolve number of defects from explicit count or site-fraction density."""
    count = getattr(args, "defect_count", None)
    density = getattr(args, "defect_density", None)

    if count is not None and density is not None:
        raise ValueError("Use only one of --defect-count or --defect-density.")

    if count is not None:
        if count < 1:
            raise ValueError("--defect-count must be >= 1.")
        return count

    if density is not None:
        if density <= 0:
            raise ValueError("--defect-density must be > 0.")
        return max(1, int(round(n_atoms * density)))

    return 1


def _add_batch_defect_args(parser) -> None:
    """Add density/count arguments for batch defect generation."""
    parser.add_argument(
        "--defect-count", type=int,
        help="Create this many defects sequentially on the same structure",
    )
    parser.add_argument(
        "--defect-density", type=float,
        help=(
            "Target defect density as site fraction (e.g. 0.01 means about "
            "1 defect per 100 atoms)"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed used for random defect selection (default: 0)",
    )
    parser.add_argument(
        "--random-index", action="store_true",
        help="Randomly sample from symmetry-unique defects instead of using --index",
    )


def _cmd_list_defects(creator, args):
    """Shared handler for ``list`` subcommands."""
    atoms = _resolve_input(args.input)
    kwargs = {}
    if hasattr(args, "substitution") and args.substitution:
        kwargs["substitution"] = _parse_substitution_arg(args.substitution)
    if hasattr(args, "species_list") and args.species_list:
        kwargs["species"] = args.species_list

    names = creator.list_defects(atoms, **kwargs)
    print(f"Found {len(names)} symmetry-unique defect(s):")
    for i, name in enumerate(names):
        print(f"  [{i}]  {name}")
    if len(names) > 20:
        print(f"  ... and {len(names) - 20} more")


def _cmd_create_defect(creator, args, extra_kwargs=None):
    """Shared handler for ``create`` subcommands."""
    from pathlib import Path
    from mckit.io import read_structure, write_atoms

    atoms = read_structure(args.input)
    kwargs = extra_kwargs or {}

    # Build kwargs for list_defects too (to get the name)
    list_kwargs = dict(kwargs)
    if hasattr(args, "substitution") and args.substitution and "substitution" not in list_kwargs:
        list_kwargs["substitution"] = _parse_substitution_arg(args.substitution)
    if hasattr(args, "species_list") and args.species_list and "species" not in list_kwargs:
        list_kwargs["species"] = args.species_list

    target_count = _resolve_target_count(args, len(atoms))
    use_random = bool(getattr(args, "random_index", False)) or target_count > 1
    rng = random.Random(getattr(args, "seed", 0))

    current_atoms = atoms
    selected_names: list[str] = []

    for _ in range(target_count):
        names = creator.list_defects(current_atoms, **list_kwargs)
        if not names:
            break

        if use_random:
            index = rng.randrange(len(names))
        else:
            index = args.index
            if index < 0 or index >= len(names):
                raise IndexError(
                    f"index {index} out of range [0, {len(names) - 1}] for current structure"
                )

        selected_names.append(names[index])
        current_atoms = creator.apply(
            structure=current_atoms, index=index, **kwargs,
        )

    if not selected_names:
        raise ValueError("No defects could be generated for the provided input.")

    stem = Path(args.input).stem
    if len(selected_names) == 1:
        safe_name = selected_names[0].replace("/", "_")
        output = args.output or f"{stem}_{safe_name}.extxyz"
        path = write_atoms(output, current_atoms)
        print(f"Created defect [{args.index}] {selected_names[0]}")
    else:
        output = args.output or f"{stem}_multi_{len(selected_names)}defects.extxyz"
        path = write_atoms(output, current_atoms)
        print(
            f"Created {len(selected_names)} defect(s) "
            f"(requested {target_count}; random={use_random})."
        )
        preview = ", ".join(selected_names[:8])
        if preview:
            tail = " ..." if len(selected_names) > 8 else ""
            print(f"  sequence: {preview}{tail}")
    print(f"  {len(atoms)} -> {len(current_atoms)} atoms  -> {path}")


def _apply_mixed_fast(
    *,
    atoms: Atoms,
    types: list[str],
    type_weights: dict[str, float],
    target_count: int,
    substitution: dict[str, list[str]] | None,
    species_list: list[str] | None,
    min_dist: float,
    seed: int,
    max_trials: int,
):
    """Fast mixed-defect creation for MD-scale supercells (approximate)."""
    work = atoms.copy()
    rng = random.Random(seed)

    sequence: list[str] = []
    type_counts = {t: 0 for t in types}

    for _ in range(target_count):
        candidates = []
        for defect_type in types:
            if _fast_defect_available(
                work, defect_type,
                substitution=substitution,
                species_list=species_list,
            ):
                candidates.append(defect_type)

        if not candidates:
            break

        defect_type = rng.choices(
            candidates,
            weights=[type_weights[t] for t in candidates],
            k=1,
        )[0]
        name = _apply_one_fast_defect(
            work,
            defect_type,
            substitution=substitution,
            species_list=species_list,
            min_dist=min_dist,
            rng=rng,
            max_trials=max_trials,
        )
        if name is None:
            continue

        type_counts[defect_type] += 1
        sequence.append(f"{defect_type}:{name}")

    return work, sequence, type_counts


def _fast_defect_available(
    atoms: Atoms,
    defect_type: str,
    *,
    substitution: dict[str, list[str]] | None,
    species_list: list[str] | None,
) -> bool:
    symbols = atoms.get_chemical_symbols()
    unique = set(symbols)

    if defect_type == "vacancy":
        return len(symbols) > 0
    if defect_type == "substitution":
        return bool(substitution) and any(host in unique for host in substitution)
    if defect_type == "antisite":
        return len(unique) >= 2
    if defect_type == "interstitial":
        return bool(species_list)
    return False


def _apply_one_fast_defect(
    atoms: Atoms,
    defect_type: str,
    *,
    substitution: dict[str, list[str]] | None,
    species_list: list[str] | None,
    min_dist: float,
    rng: random.Random,
    max_trials: int,
) -> str | None:
    symbols = atoms.get_chemical_symbols()

    if defect_type == "vacancy":
        if not symbols:
            return None
        idx = rng.randrange(len(symbols))
        removed = symbols[idx]
        del atoms[idx]
        return f"v_{removed}"

    if defect_type == "substitution":
        if not substitution:
            return None
        host_candidates = [h for h in substitution if h in symbols]
        if not host_candidates:
            return None
        host = rng.choice(host_candidates)
        host_indices = [i for i, s in enumerate(symbols) if s == host]
        if not host_indices:
            return None
        idx = rng.choice(host_indices)
        dopant = rng.choice(substitution[host])
        atoms[idx].symbol = dopant
        return f"{dopant}_{host}"

    if defect_type == "antisite":
        unique = list(set(symbols))
        if len(unique) < 2:
            return None
        to_species = rng.choice(unique)
        from_species_choices = [s for s in unique if s != to_species]
        from_species = rng.choice(from_species_choices)
        target_indices = [i for i, s in enumerate(symbols) if s == to_species]
        if not target_indices:
            return None
        idx = rng.choice(target_indices)
        atoms[idx].symbol = from_species
        return f"{from_species}_{to_species}"

    if defect_type == "interstitial":
        if not species_list:
            return None
        inter = _insert_random_interstitial(
            atoms, species=rng.choice(species_list), min_dist=min_dist,
            rng=rng, max_trials=max_trials,
        )
        return inter

    return None


def _insert_random_interstitial(
    atoms: Atoms,
    *,
    species: str,
    min_dist: float,
    rng: random.Random,
    max_trials: int,
) -> str | None:
    """Insert an interstitial via random sampling with minimum-image distance."""
    cell = np.array(atoms.cell)
    if np.linalg.det(cell) == 0:
        return None

    positions = np.asarray(atoms.get_positions())
    inv_cell = np.linalg.inv(cell)
    pbc = np.asarray(atoms.pbc, dtype=bool)

    for _ in range(max_trials):
        frac = np.array([rng.random(), rng.random(), rng.random()])
        cart = frac @ cell

        if len(positions) > 0:
            delta = positions - cart
            frac_delta = delta @ inv_cell
            for axis in range(3):
                if pbc[axis]:
                    frac_delta[:, axis] -= np.round(frac_delta[:, axis])
            cart_delta = frac_delta @ cell
            min_d2 = np.min(np.sum(cart_delta * cart_delta, axis=1))
            if min_d2 < (min_dist * min_dist):
                continue

        atoms += Atoms(symbols=[species], positions=[cart], cell=atoms.cell, pbc=atoms.pbc)
        return f"{species}_i"

    return None


# -- vacancy ---------------------------------------------------------------

def _cmd_vacancy(args):
    creator = VacancyCreator(symprec=args.symprec)
    if args.action == "list":
        _cmd_list_defects(creator, args)
    else:
        _cmd_create_defect(creator, args)


# -- substitution ----------------------------------------------------------

def _cmd_substitution(args):
    creator = SubstitutionCreator(symprec=args.symprec)
    sub_dict = _parse_substitution_arg(args.substitution)
    if args.action == "list":
        _cmd_list_defects(creator, args)
    else:
        _cmd_create_defect(creator, args, extra_kwargs={"substitution": sub_dict})


# -- antisite --------------------------------------------------------------

def _cmd_antisite(args):
    creator = AntiSiteCreator(symprec=args.symprec)
    if args.action == "list":
        _cmd_list_defects(creator, args)
    else:
        _cmd_create_defect(creator, args)


# -- interstitial ----------------------------------------------------------

def _cmd_interstitial(args):
    creator = InterstitialCreator(min_dist=args.min_dist)
    if args.action == "list":
        _cmd_list_defects(creator, args)
    else:
        _cmd_create_defect(
            creator, args, extra_kwargs={"species": args.species_list},
        )


def _build_creator_for_defect_type(args):
    """Build a creator and kwargs from a user-selected defect type."""
    defect_type = args.defect_type
    if defect_type == "vacancy":
        return VacancyCreator(symprec=args.symprec), {}
    if defect_type == "substitution":
        if not args.substitution:
            raise ValueError("substitution requires --substitution, e.g. Ga=Zn")
        return (
            SubstitutionCreator(symprec=args.symprec),
            {"substitution": _parse_substitution_arg(args.substitution)},
        )
    if defect_type == "antisite":
        return AntiSiteCreator(symprec=args.symprec), {}
    if defect_type == "interstitial":
        if not args.species_list:
            raise ValueError("interstitial requires --species")
        return InterstitialCreator(min_dist=args.min_dist), {"species": args.species_list}
    raise ValueError(f"Unknown defect type: {defect_type}")


def _cmd_enumerate(args):
    """Symmetry-unique enumeration and optional single-defect creation."""
    from pathlib import Path
    from mckit.io import read_structure, write_atoms

    creator, kwargs = _build_creator_for_defect_type(args)
    atoms = read_structure(args.input)
    names = creator.list_defects(atoms, **kwargs)

    if args.index is None:
        print(f"Found {len(names)} symmetry-unique defect(s):")
        for i, name in enumerate(names):
            print(f"  [{i}]  {name}")
        if len(names) > 20:
            print(f"  ... and {len(names) - 20} more")
        return

    result = creator.apply(structure=atoms, index=args.index, **kwargs)
    stem = Path(args.input).stem
    defect_name = names[args.index]
    safe_name = defect_name.replace("/", "_")
    output = args.output or f"{stem}_{safe_name}.extxyz"
    path = write_atoms(output, result)
    print(f"Created defect [{args.index}] {defect_name}")
    print(f"  {len(atoms)} -> {len(result)} atoms  -> {path}")


def _cmd_populate(args):
    """Populate a structure with single-type or mixed defect populations."""
    from pathlib import Path
    from mckit.io import read_structure, write_atoms

    atoms = read_structure(args.input)
    target_count = _resolve_target_count(args, len(atoms))

    raw_weights = {
        "vacancy": args.vacancy_weight,
        "substitution": args.substitution_weight,
        "antisite": args.antisite_weight,
        "interstitial": args.interstitial_weight,
    }

    for defect_type, weight in raw_weights.items():
        if weight is not None and weight < 0:
            raise ValueError(f"--{defect_type} must be >= 0.")

    types = [t for t, w in raw_weights.items() if w is not None and w > 0]
    if not types:
        raise ValueError(
            "At least one positive defect weight is required. "
            "Example: --vacancy 1.0 or --vacancy 0.7 --interstitial 0.3"
        )

    type_weights = {t: raw_weights[t] for t in types}

    substitution = None
    if "substitution" in types:
        if not args.substitution_map:
            raise ValueError(
                "populate requires --sub-map when substitution is enabled. "
                "Example: --sub-map Ga=Zn"
            )
        substitution = _parse_substitution_arg(args.substitution_map)

    if "interstitial" in types and not args.species_list:
        raise ValueError("populate requires --species when interstitial is enabled.")

    result_atoms, sequence, type_counts = _apply_mixed_fast(
        atoms=atoms,
        types=types,
        type_weights=type_weights,
        target_count=target_count,
        substitution=substitution,
        species_list=args.species_list,
        min_dist=args.min_dist,
        seed=args.seed,
        max_trials=args.max_trials,
    )

    if not sequence:
        raise ValueError("No defects could be generated for this populate setup.")

    stem = Path(args.input).stem
    output = args.output or f"{stem}_populate_{len(sequence)}_fast.extxyz"
    path = write_atoms(output, result_atoms)

    print(
        f"Created defect population: {len(sequence)} applied "
        f"(requested {target_count})."
    )
    for defect_type in types:
        print(f"  {defect_type:12s}: {type_counts[defect_type]}")
    preview = ", ".join(sequence[:8])
    if preview:
        tail = " ..." if len(sequence) > 8 else ""
        print(f"  sequence: {preview}{tail}")
    print(f"  {len(atoms)} -> {len(result_atoms)} atoms  -> {path}")


def _register_defect_commands(defect_sub):
    """Register user-facing defect workflows."""
    enum = defect_sub.add_parser(
        "enumerate",
        help="List symmetry-unique defects and optionally create one",
    )
    enum.add_argument("input", help="Input structure file")
    enum.add_argument(
        "--type", dest="defect_type", required=True,
        choices=["vacancy", "substitution", "antisite", "interstitial"],
        help="Defect type to enumerate",
    )
    enum.add_argument(
        "--index", type=int,
        help="Create this index from the enumerated list (omit to just list)",
    )
    enum.add_argument("--output", "-o", help="Output file (used with --index)")
    enum.add_argument(
        "--substitution", "--sub",
        help='For --type substitution: "Host=Dopant" (e.g. "Ga=Zn")',
    )
    enum.add_argument(
        "--species", nargs="+", dest="species_list",
        help="For --type interstitial: species list (e.g. Li Na)",
    )
    enum.add_argument(
        "--symprec", type=float, default=0.01,
        help="Symmetry precision for enumeration (default: 0.01)",
    )
    enum.add_argument(
        "--min-dist", type=float, default=0.9,
        help="Minimum distance for interstitial generation (default: 0.9 A)",
    )
    enum.set_defaults(handler=_cmd_enumerate)

    pop = defect_sub.add_parser(
        "populate",
        help="Create scalable defect populations (single or mixed types)",
    )
    pop.add_argument("input", help="Input structure file")
    pop.add_argument(
        "--vacancy", dest="vacancy_weight", type=float,
        help="Weight for vacancy defects (e.g. 1.0 or 0.3)",
    )
    pop.add_argument(
        "--substitution", dest="substitution_weight", type=float,
        help="Weight for substitution defects",
    )
    pop.add_argument(
        "--antisite", dest="antisite_weight", type=float,
        help="Weight for antisite defects",
    )
    pop.add_argument(
        "--interstitial", dest="interstitial_weight", type=float,
        help="Weight for interstitial defects",
    )
    pop.add_argument(
        "--sub-map", "--substitution-map", "--sub", dest="substitution_map",
        help='Required when --substitution is set: "Host=Dopant"',
    )
    pop.add_argument(
        "--species", nargs="+", dest="species_list",
        help="Required when --interstitial is set",
    )
    pop.add_argument(
        "--min-dist", type=float, default=0.9,
        help="Minimum distance to existing atoms for interstitials (default: 0.9 A)",
    )
    pop.add_argument(
        "--max-trials", type=int, default=500,
        help="Max random trials to place each fast interstitial (default: 500)",
    )
    pop.add_argument("--output", "-o", help="Output file")
    _add_batch_defect_args(pop)
    pop.set_defaults(handler=_cmd_populate)


# -- registration ----------------------------------------------------------

def register_cli(subparsers) -> None:
    """Register defect subcommands with the mmkit CLI."""
    defect = subparsers.add_parser(
        "defect",
        help="Defect tools: enumerate (symmetry) and populate (scalable)",
        description=(
            "Defect workflows:\n"
            "  - enumerate: symmetry-unique defect listing / single creation\n"
            "  - populate: scalable defect populations for supercells/MD"
        ),
    )
    defect_sub = defect.add_subparsers(dest="workflow", required=True)
    _register_defect_commands(defect_sub)


def register_cli_root(subparsers) -> None:
    """Register top-level shortcut commands for defect workflows."""
    _register_defect_commands(subparsers)
