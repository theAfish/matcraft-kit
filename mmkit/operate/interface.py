"""Build coherent ZSL-matched interfaces between two bulk crystals.

Uses pymatgen's ``SubstrateAnalyzer`` to find the best lattice match
(lowest von Mises strain), then ``CoherentInterfaceBuilder`` to construct
the interface with the specified gap, vacuum, and slab thicknesses.

When ``preserve_molecules=True`` (the default), molecules are detected in
each bulk structure, replaced by single pseudo-atoms for ZSL matching, and
then recovered with correct orientations after interface construction.
This prevents molecules from being cut by the slab boundaries.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from ase import Atoms
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.core import Lattice as PmgLattice
from pymatgen.core import Structure as PmgStructure
from pymatgen.io.ase import AseAtomsAdaptor

from mmkit.core.structure import Structure
from mmkit.core.tool import Operation


# ---------------------------------------------------------------------------
# Pseudo-atom support
# ---------------------------------------------------------------------------
# We use pymatgen's DummySpecies with symbol "X" (not a real element) as the
# pseudo-atom for molecule replacement.  DummySpecies lacks ``atomic_mass``,
# which pymatgen's ``center_slab`` (called by CoherentInterfaceBuilder)
# requires.  We monkey-patch it to return an arbitrary mass so the full
# interface-building pipeline works.

_PSEUDO_SYMBOL = "X"
_dummy_patched = False


def _ensure_dummy_patched() -> None:
    """Ensure DummySpecies provides atomic_mass for CoherentInterfaceBuilder."""
    global _dummy_patched
    if _dummy_patched:
        return
    from pymatgen.core.periodic_table import DummySpecies

    def _patched_getattr(self, attr):
        if attr in ("atomic_mass", "mass"):
            return 100.0
        raise AttributeError

    DummySpecies.__getattr__ = _patched_getattr
    _dummy_patched = True


# ---------------------------------------------------------------------------
# Molecule helpers
# ---------------------------------------------------------------------------

def _pbc_center(structure: PmgStructure, indices: List[int]) -> np.ndarray:
    """PBC-aware geometric center (fractional coords)."""
    coords = structure.frac_coords[indices].copy()
    ref = coords[0]
    for j in range(3):
        diff = coords[:, j] - ref[j]
        coords[:, j] = ref[j] + (diff + 0.5) % 1.0 - 0.5
    return coords.mean(axis=0) % 1.0


def _build_mol_templates(
    bulk: PmgStructure, molecules: List[List[int]],
) -> List[Dict]:
    """Extract Cartesian offset templates for each detected molecule."""
    templates = []
    for mol in molecules:
        center_cart = bulk.lattice.get_cartesian_coords(_pbc_center(bulk, mol))
        offsets = []
        species = []
        for idx in mol:
            cart = bulk.lattice.get_cartesian_coords(bulk.frac_coords[idx])
            d = cart - center_cart
            for j in range(3):
                ll = bulk.lattice.abc[j]
                if d[j] > ll / 2:
                    d[j] -= ll
                elif d[j] < -ll / 2:
                    d[j] += ll
            offsets.append(d)
            species.append(str(bulk[idx].specie))
        templates.append({
            "offsets": np.array(offsets),
            "species": species,
            "center": _pbc_center(bulk, mol),
        })
    return templates


def _create_pseudo_structure(
    bulk: PmgStructure,
    molecules: List[List[int]],
    templates: List[Dict],
) -> PmgStructure:
    """Replace each molecule with a single X pseudo-atom at its center."""
    _ensure_dummy_patched()
    from pymatgen.core.periodic_table import DummySpecies

    mol_set = set()
    for mol in molecules:
        mol_set.update(mol)

    species = []
    frac_coords = []
    for i, site in enumerate(bulk):
        if i not in mol_set:
            species.append(site.specie)
            frac_coords.append(site.frac_coords)

    for template in templates:
        species.append(DummySpecies(_PSEUDO_SYMBOL))
        frac_coords.append(template["center"])

    return PmgStructure(bulk.lattice, species, frac_coords)


# ---------------------------------------------------------------------------
# Transformation extraction
# ---------------------------------------------------------------------------

def _compute_rotation(M_bulk: np.ndarray, M_interface: np.ndarray) -> np.ndarray:
    """Extract the rotation component from the deformation gradient.

    F = M_interface^T @ M_bulk^{-T} captures the full lattice transformation.
    Polar decomposition F = R @ U gives R (pure rotation) and U (stretch).
    """
    from scipy.linalg import polar
    F = M_interface.T @ np.linalg.inv(M_bulk.T)
    R, _ = polar(F)
    return R


# ---------------------------------------------------------------------------
# Molecule recovery
# ---------------------------------------------------------------------------

def _wrap_molecules(
    cart_coords: List[np.ndarray],
    mol_ids: List[int],
    lattice: PmgLattice,
) -> List[np.ndarray]:
    """Wrap atoms to [0, 1) while keeping molecules intact.

    For atoms with mol_id >= 0 (molecules), wrap the molecule center and
    shift all atoms in the molecule by the same amount.
    For atoms with mol_id < 0 (inorganic), wrap individually.

    Parameters
    ----------
    cart_coords
        List of Cartesian coordinates.
    mol_ids
        Molecule ID for each atom (-1 for inorganic).
    lattice
        Interface lattice.

    Returns
    -------
    List of wrapped Cartesian coordinates.
    """
    M = np.array(lattice.matrix)
    M_inv = np.linalg.inv(M)

    # Convert to fractional
    frac_coords = [M_inv @ c for c in cart_coords]

    # Group by molecule ID
    mol_groups = {}
    for i, mid in enumerate(mol_ids):
        if mid >= 0:
            if mid not in mol_groups:
                mol_groups[mid] = []
            mol_groups[mid].append(i)

    # Wrap each molecule by its center
    for mid, indices in mol_groups.items():
        # Compute PBC-aware center of this molecule
        mfracs = np.array([frac_coords[i] for i in indices])
        ref = mfracs[0]
        for j in range(3):
            diff = mfracs[:, j] - ref[j]
            mfracs[:, j] = ref[j] + (diff + 0.5) % 1.0 - 0.5
        center = mfracs.mean(axis=0)

        # Wrap center to [0, 1)
        wrapped_center = center % 1.0
        shift = wrapped_center - center

        # Apply shift to all atoms in molecule
        for i in indices:
            frac_coords[i] = frac_coords[i] + shift

    # Wrap inorganic atoms individually
    for i, mid in enumerate(mol_ids):
        if mid < 0:
            frac_coords[i] = frac_coords[i] % 1.0

    # Convert back to Cartesian
    return [M @ f for f in frac_coords]


def _recover_molecules(
    interface: PmgStructure,
    region_indices: List[int],
    templates: List[Dict],
    R: np.ndarray,
    M_bulk: np.ndarray,
    is_film: bool,
) -> Tuple[List[str], List[np.ndarray], List[int]]:
    """Place molecules at pseudo-atom sites using position-based mapping.

    Uses the deformation gradient to map each pseudo atom back to its original
    bulk position, then finds the corresponding molecule template by proximity.

    Parameters
    ----------
    interface
        The interface structure from CoherentInterfaceBuilder.
    region_indices
        Indices of pseudo-atom sites in this region (film or substrate).
    templates
        Molecule templates from ``_build_mol_templates``.
    R
        Rotation matrix from ``_compute_rotation``.
    M_bulk
        Original bulk lattice matrix (3x3).
    is_film
        If True, flip z-component of offsets (film is inverted by pymatgen).

    Returns
    -------
    (species_list, cart_coords_list, molecule_ids)
        All molecule atoms in interface Cartesian coordinates, with molecule IDs.
    """
    if not region_indices:
        return [], [], []

    # Compute supercell mapping: bulk_frac -> interface_frac
    # S_invT = M_bulk.T @ inv(R @ M_bulk.T)
    S_invT = M_bulk.T @ np.linalg.inv(R @ M_bulk.T)

    all_species: List[str] = []
    all_cart: List[np.ndarray] = []
    all_mol_ids: List[int] = []

    for si, site_idx in enumerate(region_indices):
        # Map pseudo atom from interface to bulk fractional coords
        f_iface = interface[site_idx].frac_coords
        f_bulk = (S_invT @ f_iface) % 1.0

        # Find closest template by comparing bulk center positions
        best_template = None
        best_dist = float('inf')

        for template in templates:
            center_bulk = template["center"]
            # Periodic distance in bulk fractional coords
            delta = f_bulk - center_bulk
            delta = delta - np.round(delta)
            dist = np.linalg.norm(M_bulk.T @ delta)

            if dist < best_dist:
                best_dist = dist
                best_template = template

        # Recover molecule at this pseudo atom position
        center_cart = interface.lattice.get_cartesian_coords(f_iface)

        for d, sp in zip(best_template["offsets"], best_template["species"]):
            rd = R @ d
            if is_film:
                rd = rd.copy()
                rd[2] *= -1.0  # film z-flip
            atom_cart = center_cart + rd
            all_species.append(sp)
            all_cart.append(atom_cart)
            all_mol_ids.append(si)

    return all_species, all_cart, all_mol_ids


def _recalculate_gap(
    cart_coords: List[np.ndarray],
    labels: List[str],
    lattice: PmgLattice,
    requested_gap: float,
) -> Tuple[List[np.ndarray], PmgLattice]:
    """Adjust film positions to ensure minimum gap between film and substrate.

    Parameters
    ----------
    cart_coords
        List of Cartesian coordinates for all atoms.
    labels
        List of "film" or "substrate" for each atom.
    lattice
        Current interface lattice.
    requested_gap
        Minimum required gap in Angstroms.

    Returns
    -------
    (cart_coords, lattice)
        Updated Cartesian coordinates and lattice.
    """
    if not cart_coords:
        return cart_coords, lattice

    cart_z = np.array([c[2] for c in cart_coords])

    sub_mask = np.array([l == "substrate" for l in labels])
    film_mask = np.array([l == "film" for l in labels])

    if not np.any(sub_mask) or not np.any(film_mask):
        return cart_coords, lattice

    z_max_sub = np.max(cart_z[sub_mask])
    z_min_film = np.min(cart_z[film_mask])
    actual_gap = z_min_film - z_max_sub

    if actual_gap >= requested_gap:
        return cart_coords, lattice

    # Shift film atoms up by the deficit
    shift_z = requested_gap - actual_gap

    new_cart = []
    for c, l in zip(cart_coords, labels):
        c = c.copy()
        if l == "film":
            c[2] += shift_z
        new_cart.append(c)

    # Expand lattice c-vector to accommodate the shift
    M = np.array(lattice.matrix)
    new_c_length = lattice.c + shift_z
    new_c = M[2].copy()
    new_c = new_c / np.linalg.norm(new_c) * new_c_length
    new_lattice = PmgLattice([M[0], M[1], new_c])

    return new_cart, new_lattice


# ---------------------------------------------------------------------------
# InterfaceBuilder
# ---------------------------------------------------------------------------

class InterfaceBuilder(Operation):
    """Build a coherent ZSL-matched interface between two bulk crystals.

    When ``preserve_molecules`` is enabled (default), molecules in each bulk
    are detected, replaced by pseudo-atoms for lattice matching, and then
    recovered with correct orientations in the final interface.

    Example
    -------
    >>> from mmkit.io.reader import read_structure
    >>> builder = InterfaceBuilder()
    >>> interface = builder.apply(
    ...     film=read_structure("film.cif"),
    ...     substrate=read_structure("substrate.cif"),
    ...     miller_film=(1, 0, 0),
    ...     miller_substrate=(1, 1, 1),
    ... )
    """

    def apply(
        self,
        *,
        film: Union[Atoms, PmgStructure, Structure],
        substrate: Union[Atoms, PmgStructure, Structure],
        miller_film: Tuple[int, int, int] = (1, 0, 0),
        miller_substrate: Tuple[int, int, int] = (1, 1, 1),
        termination=0,
        max_area: Optional[float] = 400.0,
        max_length_tol: float = 0.03,
        max_angle_tol: float = 0.01,
        gap: float = 2.5,
        vacuum_between: Optional[float] = 0.0,
        thickness_film: int = 2,
        thickness_substrate: int = 2,
        in_layers: bool = True,
        preserve_molecules: bool = True,
        mol_tol: float = 0.45,
        mol_min_size: int = 2,
    ) -> PmgStructure:
        """Build a coherent interface from two bulk structures.

        Parameters
        ----------
        film, substrate
            The film and substrate bulk structures.  Accepts ``ase.Atoms``,
            ``pymatgen.Structure``, or ``mmkit.Structure``.
        miller_film, miller_substrate
            Miller indices of the surfaces to expose.
        termination
            Which termination to use.  An integer selects by index, a string
            matches by label, ``"all"`` returns all terminations (first one
            is used for now).
        max_area
            Maximum supercell area for ZSL search (A^2).
        max_length_tol, max_angle_tol
            ZSL matching tolerances.
        gap
            Distance between film and substrate (A).
        vacuum_between
            Vacuum above the film (A).  ``0`` means same as ``gap``.
        thickness_film, thickness_substrate
            Slab thickness in layers (or A if ``in_layers=False``).
        in_layers
            If ``True``, thickness is counted in atomic layers.
        preserve_molecules
            If ``True``, detect molecules and use pseudo-atom approach.
        mol_tol
            Bond-detection tolerance for molecule detection (A).
        mol_min_size
            Minimum atoms to count as a molecule.
        """
        film_pmg = self._to_pymatgen(film)
        substrate_pmg = self._to_pymatgen(substrate)

        # ---- Molecule detection ------------------------------------------
        film_molecules: List[List[int]] = []
        sub_molecules: List[List[int]] = []
        film_templates: List[Dict] = []
        sub_templates: List[Dict] = []

        if preserve_molecules:
            from mmkit.operate.surface import MoleculeDetector

            detector = MoleculeDetector(tol=mol_tol, min_size=mol_min_size)
            film_molecules = detector.detect(film_pmg)
            sub_molecules = detector.detect(substrate_pmg)

            if film_molecules:
                film_templates = _build_mol_templates(film_pmg, film_molecules)
            if sub_molecules:
                sub_templates = _build_mol_templates(substrate_pmg, sub_molecules)

        # ---- Choose structures for interface builder ---------------------
        if preserve_molecules and (film_molecules or sub_molecules):
            build_film = (
                _create_pseudo_structure(film_pmg, film_molecules, film_templates)
                if film_molecules else film_pmg
            )
            build_sub = (
                _create_pseudo_structure(substrate_pmg, sub_molecules, sub_templates)
                if sub_molecules else substrate_pmg
            )
        else:
            build_film = film_pmg
            build_sub = substrate_pmg

        # ---- ZSL matching ------------------------------------------------
        analyzer = SubstrateAnalyzer(
            max_area_ratio_tol=0.09,
            max_area=max_area,
            max_length_tol=max_length_tol,
            max_angle_tol=max_angle_tol,
        )
        matches = list(analyzer.calculate(
            film=build_film,
            substrate=build_sub,
            film_millers=[miller_film],
            substrate_millers=[miller_substrate],
        ))
        if not matches:
            raise ValueError(
                "No lattice matches found. Try adjusting tolerances or Miller indices."
            )
        match = sorted(matches, key=lambda m: m.von_mises_strain)[0]

        # ---- Interface construction --------------------------------------
        builder = CoherentInterfaceBuilder(
            film_structure=build_film,
            substrate_structure=build_sub,
            film_miller=match.film_miller,
            substrate_miller=match.substrate_miller,
            zslgen=analyzer,
        )
        all_terms = builder.terminations
        if not all_terms:
            raise ValueError("No terminations available for the selected slabs.")

        selected = self._select_termination(all_terms, termination)

        effective_vacuum = vacuum_between if vacuum_between != 0 else gap
        interfaces = list(builder.get_interfaces(
            termination=selected,
            gap=gap,
            vacuum_over_film=effective_vacuum,
            film_thickness=thickness_film,
            substrate_thickness=thickness_substrate,
            in_layers=in_layers,
        ))
        if not interfaces:
            raise ValueError("No interfaces generated. Check parameters.")

        interface = interfaces[0]

        # ---- Molecule recovery -------------------------------------------
        if preserve_molecules and (film_molecules or sub_molecules):
            M_iface = np.array(interface.lattice.matrix)

            all_species: List[str] = []
            all_cart: List[np.ndarray] = []
            all_labels: List[str] = []
            all_mol_ids: List[int] = []  # -1 for inorganic, >=0 for molecule ID

            # Substrate: recover molecules + keep inorganic atoms
            sub_indices = interface.substrate_indices
            if sub_molecules:
                sub_pseudo = [
                    i for i in sub_indices
                    if interface[i].specie.symbol == _PSEUDO_SYMBOL
                ]
                R_sub = _compute_rotation(
                    np.array(substrate_pmg.lattice.matrix), M_iface
                )
                sp, cc, mids = _recover_molecules(
                    interface, sub_pseudo, sub_templates, R_sub,
                    np.array(substrate_pmg.lattice.matrix), is_film=False,
                )
                all_species.extend(sp)
                all_cart.extend(cc)
                all_labels.extend(["substrate"] * len(sp))
                all_mol_ids.extend(mids)

                for i in sub_indices:
                    if interface[i].specie.symbol != _PSEUDO_SYMBOL:
                        all_species.append(str(interface[i].specie))
                        all_cart.append(interface.lattice.get_cartesian_coords(
                            interface[i].frac_coords
                        ))
                        all_labels.append("substrate")
                        all_mol_ids.append(-1)  # inorganic
            else:
                # No molecules in substrate — keep all substrate atoms
                for i in sub_indices:
                    all_species.append(str(interface[i].specie))
                    all_cart.append(interface.lattice.get_cartesian_coords(
                        interface[i].frac_coords
                    ))
                    all_labels.append("substrate")
                    all_mol_ids.append(-1)

            # Film: recover molecules + keep inorganic atoms
            film_indices = interface.film_indices
            if film_molecules:
                film_pseudo = [
                    i for i in film_indices
                    if interface[i].specie.symbol == _PSEUDO_SYMBOL
                ]
                R_film = _compute_rotation(
                    np.array(film_pmg.lattice.matrix), M_iface
                )
                sp, cc, mids = _recover_molecules(
                    interface, film_pseudo, film_templates, R_film,
                    np.array(film_pmg.lattice.matrix), is_film=True,
                )
                all_species.extend(sp)
                all_cart.extend(cc)
                all_labels.extend(["film"] * len(sp))
                # Offset film molecule IDs to avoid collision with substrate IDs
                max_sub_id = max(all_mol_ids) if any(m >= 0 for m in all_mol_ids) else -1
                all_mol_ids.extend([m + max_sub_id + 1 if m >= 0 else -1 for m in mids])

                for i in film_indices:
                    if interface[i].specie.symbol != _PSEUDO_SYMBOL:
                        all_species.append(str(interface[i].specie))
                        all_cart.append(interface.lattice.get_cartesian_coords(
                            interface[i].frac_coords
                        ))
                        all_labels.append("film")
                        all_mol_ids.append(-1)
            else:
                # No molecules in film — keep all film atoms
                for i in film_indices:
                    all_species.append(str(interface[i].specie))
                    all_cart.append(interface.lattice.get_cartesian_coords(
                        interface[i].frac_coords
                    ))
                    all_labels.append("film")
                    all_mol_ids.append(-1)

            # Wrap molecules while keeping them intact
            all_cart = _wrap_molecules(all_cart, all_mol_ids, interface.lattice)

            # Gap recalculation (in Cartesian space)
            all_cart, new_lattice = _recalculate_gap(
                all_cart, all_labels,
                interface.lattice, gap,
            )

            # Convert Cartesian to fractional for final structure
            M_inv = np.linalg.inv(new_lattice.matrix)
            all_frac = [M_inv @ c for c in all_cart]

            interface = PmgStructure(
                new_lattice, all_species, all_frac,
                coords_are_cartesian=False,
            )
        else:
            # No molecule preservation — just wrap atoms
            species = [site.specie for site in interface]
            frac_coords = [site.frac_coords % 1.0 for site in interface]
            interface = PmgStructure(interface.lattice, species, frac_coords)

        return interface

    @staticmethod
    def _select_termination(terminations, selection):
        """Select a termination by index, label, or 'all'."""
        if isinstance(selection, str) and selection.lower() == "all":
            return terminations[0]  # use first for now

        if isinstance(selection, int):
            if 0 <= selection < len(terminations):
                return terminations[selection]
            raise ValueError(
                f"Termination index {selection} out of range "
                f"[0, {len(terminations) - 1}]"
            )

        # String label match
        sel_str = str(selection)
        for t in terminations:
            if sel_str in str(t):
                return t
        raise ValueError(
            f"No termination matching '{sel_str}'. "
            f"Available: {terminations}"
        )

    @staticmethod
    def _to_pymatgen(obj: Union[Atoms, PmgStructure, Structure]) -> PmgStructure:
        """Coerce various structure types to pymatgen Structure."""
        if isinstance(obj, PmgStructure):
            return obj
        if isinstance(obj, Structure):
            return obj.to_pymatgen()
        if isinstance(obj, Atoms):
            return AseAtomsAdaptor().get_structure(obj)
        raise TypeError(
            f"Expected Atoms, pymatgen Structure, or mmkit Structure, "
            f"got {type(obj).__name__}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_build(args):
    """CLI handler: build a coherent interface."""
    from pathlib import Path

    from mmkit.io.reader import read_structure
    from mmkit.io.writer import write_structure

    film_atoms = read_structure(args.film)
    substrate_atoms = read_structure(args.substrate)

    builder = InterfaceBuilder()
    interface = builder.apply(
        film=film_atoms,
        substrate=substrate_atoms,
        miller_film=tuple(args.miller_film),
        miller_substrate=tuple(args.miller_substrate),
        termination=args.termination,
        max_area=args.max_area,
        max_length_tol=args.max_length_tol,
        max_angle_tol=args.max_angle_tol,
        gap=args.gap,
        vacuum_between=args.vacuum,
        thickness_film=args.thickness_film,
        thickness_substrate=args.thickness_substrate,
        in_layers=not args.angstrom_thickness,
        preserve_molecules=not args.no_preserve_molecules,
        mol_tol=args.mol_tol,
        mol_min_size=args.mol_min_size,
    )

    if args.output:
        output = args.output
    else:
        film_stem = Path(args.film).stem
        sub_stem = Path(args.substrate).stem
        output = f"{film_stem}-{sub_stem}_interface.extxyz"

    path = write_structure(output, interface)
    print(f"Built interface -> {path}  ({len(interface)} atoms)")


def register_cli(subparsers) -> None:
    """Register interface subcommands with the mmkit CLI."""
    interface = subparsers.add_parser("interface", help="Build coherent interfaces")
    iface_sub = interface.add_subparsers(dest="action", required=True)

    p = iface_sub.add_parser("build", help="Build a coherent ZSL-matched interface")
    # Positional arguments for required inputs
    p.add_argument("film", help="Film bulk structure file")
    p.add_argument("substrate", help="Substrate bulk structure file")
    p.add_argument(
        "--miller-film", type=int, nargs=3, default=[1, 0, 0],
        help="Film surface Miller indices (h k l)",
    )
    p.add_argument(
        "--miller-substrate", type=int, nargs=3, default=[1, 1, 1],
        help="Substrate surface Miller indices (h k l)",
    )
    p.add_argument(
        "--termination", default=0,
        help="Termination index, label, or 'all' (default: 0)",
    )
    p.add_argument(
        "--max-area", type=float, default=400.0,
        help="Max supercell area for ZSL search (A^2)",
    )
    p.add_argument(
        "--max-length-tol", type=float, default=0.03,
        help="Max length tolerance for ZSL matching",
    )
    p.add_argument(
        "--max-angle-tol", type=float, default=0.01,
        help="Max angle tolerance for ZSL matching",
    )
    p.add_argument(
        "--gap", type=float, default=2.5,
        help="Gap between film and substrate (A)",
    )
    p.add_argument(
        "--vacuum", type=float, default=0.0,
        help="Vacuum above film (A, 0 = same as gap)",
    )
    p.add_argument(
        "--thickness-film", type=int, default=2,
        help="Film slab thickness (layers or A)",
    )
    p.add_argument(
        "--thickness-substrate", type=int, default=2,
        help="Substrate slab thickness (layers or A)",
    )
    p.add_argument(
        "--angstrom-thickness", action="store_true",
        help="Interpret thickness as Angstroms instead of layers",
    )
    # Molecule preservation options
    p.add_argument(
        "--no-preserve-molecules", action="store_true",
        help="Disable molecule preservation (use raw structures)",
    )
    p.add_argument(
        "--mol-tol", type=float, default=0.45,
        help="Bond detection tolerance for molecules (A)",
    )
    p.add_argument(
        "--mol-min-size", type=int, default=2,
        help="Min atoms to count as a molecule",
    )
    p.add_argument(
        "--output", "-o",
        help="Output file (default: <film>-<substrate>_interface.extxyz)",
    )
    p.set_defaults(handler=_cmd_build)
