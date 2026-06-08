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

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from ase import Atoms

from mmkit.core.structure import Structure
from mmkit.core.tool import Operation


def _get_pymatgen_types():
    """Import pymatgen types lazily so the module loads without pymatgen."""
    from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
    from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
    from pymatgen.core import Lattice as PmgLattice
    from pymatgen.core import Structure as PmgStructure
    from pymatgen.io.ase import AseAtomsAdaptor
    return {
        "CoherentInterfaceBuilder": CoherentInterfaceBuilder,
        "SubstrateAnalyzer": SubstrateAnalyzer,
        "PmgLattice": PmgLattice,
        "PmgStructure": PmgStructure,
        "AseAtomsAdaptor": AseAtomsAdaptor,
    }


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class InterfaceTermination:
    """A single interface termination pair (film + substrate)."""

    index: int
    film_label: str
    substrate_label: str
    film_shift: float
    substrate_shift: float


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

    PmgStructure = _get_pymatgen_types()["PmgStructure"]

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
    """Wrap atoms in-plane while keeping molecules intact.

    For atoms with mol_id >= 0 (molecules), wrap the molecule center (x/y) and
    shift all atoms in the molecule by the same amount.
    For atoms with mol_id < 0 (inorganic), wrap individually in x/y.

    Notes
    -----
    Interfaces are slab-like with vacuum along z. Wrapping z per-atom can
    split one slab across the periodic boundary and place a few film atoms
    near the substrate side. We therefore wrap only the in-plane axes.

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
    # mmkit uses row-vector convention: cart = frac @ lattice.
    frac_coords = [c @ M_inv for c in cart_coords]

    # Group by molecule ID
    mol_groups = {}
    for i, mid in enumerate(mol_ids):
        if mid >= 0:
            if mid not in mol_groups:
                mol_groups[mid] = []
            mol_groups[mid].append(i)

    # Wrap each molecule by its center (x/y only)
    for mid, indices in mol_groups.items():
        # Compute PBC-aware center of this molecule
        mfracs = np.array([frac_coords[i] for i in indices])
        ref = mfracs[0]
        for j in (0, 1):
            diff = mfracs[:, j] - ref[j]
            mfracs[:, j] = ref[j] + (diff + 0.5) % 1.0 - 0.5
        center = mfracs.mean(axis=0)

        # Wrap center to [0, 1) in-plane only
        wrapped_center = center % 1.0
        wrapped_center[2] = center[2]
        shift = wrapped_center - center

        # Apply shift to all atoms in molecule
        for i in indices:
            frac_coords[i] = frac_coords[i] + shift

    # Wrap inorganic atoms individually (x/y only)
    for i, mid in enumerate(mol_ids):
        if mid < 0:
            frac_coords[i][0] = frac_coords[i][0] % 1.0
            frac_coords[i][1] = frac_coords[i][1] % 1.0

    # Convert back to Cartesian
    return [f @ M for f in frac_coords]


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
            # Film z-flip must happen in the bulk frame BEFORE rotation.
            # Pymatgen inverts film z-coords (1-z) in the bulk frame; the
            # lattice transformation (deformation gradient F = R @ U) then
            # maps the flipped offsets into the interface frame.  Flipping
            # after R is wrong whenever R has off-diagonal elements that
            # mix z with x/y (i.e. most non-trivial Miller-index combos).
            d_work = d.copy()
            if is_film:
                d_work[2] *= -1.0
            rd = R @ d_work
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

    # Expand lattice c-vector to accommodate the shift.
    # Only increase the z-component of c so the in-plane lattice is
    # undisturbed (the c-vector from CoherentInterfaceBuilder is along z,
    # but rescaling the full direction would also alter cx/cy if non-zero).
    M = np.array(lattice.matrix)
    new_c = M[2].copy()
    new_c[2] += shift_z
    PmgLattice = _get_pymatgen_types()["PmgLattice"]
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_bulk(
        bulk_pmg: PmgStructure,
        preserve_molecules: bool,
        mol_tol: float,
        mol_min_size: int,
    ) -> Tuple[PmgStructure, List[List[int]], List[Dict]]:
        """Detect molecules and build pseudo-structure if needed.

        Returns
        -------
        (build_pmg, molecules, templates)
            ``build_pmg`` is the pseudo-atom structure (or original if no
            molecules found).  ``molecules`` and ``templates`` are empty
            lists when no molecules are detected.
        """
        molecules: List[List[int]] = []
        templates: List[Dict] = []

        if preserve_molecules:
            from mmkit.operate.surface import MoleculeDetector

            detector = MoleculeDetector(tol=mol_tol, min_size=mol_min_size)
            molecules = detector.detect(bulk_pmg)
            if molecules:
                templates = _build_mol_templates(bulk_pmg, molecules)

        if molecules:
            build_pmg = _create_pseudo_structure(bulk_pmg, molecules, templates)
        else:
            build_pmg = bulk_pmg

        return build_pmg, molecules, templates

    @staticmethod
    def _find_zsl_match(
        build_film: PmgStructure,
        build_sub: PmgStructure,
        miller_film: Tuple[int, int, int],
        miller_substrate: Tuple[int, int, int],
        max_area: Optional[float],
        max_length_tol: float,
        max_angle_tol: float,
    ):
        """Run SubstrateAnalyzer and return the best match + analyzer."""
        SubstrateAnalyzer = _get_pymatgen_types()["SubstrateAnalyzer"]
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
        return match, analyzer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_terminations(
        self,
        *,
        film: Union[Atoms, PmgStructure, Structure],
        substrate: Union[Atoms, PmgStructure, Structure],
        miller_film: Tuple[int, int, int] = (1, 0, 0),
        miller_substrate: Tuple[int, int, int] = (1, 1, 1),
        max_area: Optional[float] = 400.0,
        max_length_tol: float = 0.03,
        max_angle_tol: float = 0.01,
        termination_ftol: Union[float, Tuple[float, float]] = 0.25,
        label_index: bool = False,
        filter_out_sym_slabs: bool = True,
        preserve_molecules: bool = True,
        mol_tol: float = 0.45,
        mol_min_size: int = 2,
    ) -> dict:
        """Discover all interface terminations for a film/substrate pair.

        Parameters
        ----------
        film, substrate
            The film and substrate bulk structures.
        miller_film, miller_substrate
            Miller indices of the surfaces to expose.
        max_area
            Maximum supercell area for ZSL search (A^2).
        max_length_tol, max_angle_tol
            ZSL matching tolerances.
        termination_ftol
            Tolerance for distinguishing termination planes (A).
            A tuple sets (film_ftol, substrate_ftol) independently.
        label_index
            If ``True``, prefix termination labels with an index to
            disambiguate labels that would otherwise be identical.
        filter_out_sym_slabs
            If ``True``, filter out symmetrically equivalent slabs.
        preserve_molecules
            If ``True``, detect molecules and use pseudo-atom approach.
        mol_tol
            Bond-detection tolerance for molecule detection (A).
        mol_min_size
            Minimum atoms to count as a molecule.

        Returns
        -------
        dict
            Keys: ``film_formula``, ``substrate_formula``, ``film_miller``,
            ``substrate_miller``, ``von_mises_strain``, ``match_area``,
            ``terminations`` (list of :class:`InterfaceTermination`).
        """
        film_pmg = self._to_pymatgen(film)
        substrate_pmg = self._to_pymatgen(substrate)

        build_film, _, _ = self._prepare_bulk(
            film_pmg, preserve_molecules, mol_tol, mol_min_size,
        )
        build_sub, _, _ = self._prepare_bulk(
            substrate_pmg, preserve_molecules, mol_tol, mol_min_size,
        )

        match, analyzer = self._find_zsl_match(
            build_film, build_sub,
            miller_film, miller_substrate,
            max_area, max_length_tol, max_angle_tol,
        )

        CoherentInterfaceBuilder = _get_pymatgen_types()["CoherentInterfaceBuilder"]
        cib = CoherentInterfaceBuilder(
            film_structure=build_film,
            substrate_structure=build_sub,
            film_miller=match.film_miller,
            substrate_miller=match.substrate_miller,
            zslgen=analyzer,
            termination_ftol=termination_ftol,
            label_index=label_index,
            filter_out_sym_slabs=filter_out_sym_slabs,
        )

        terminations: List[InterfaceTermination] = []
        for i, (pair, shifts) in enumerate(cib._terminations.items()):
            film_label, sub_label = pair
            film_shift, sub_shift = shifts
            terminations.append(InterfaceTermination(
                index=i,
                film_label=film_label,
                substrate_label=sub_label,
                film_shift=film_shift,
                substrate_shift=sub_shift,
            ))

        return {
            "film_formula": film_pmg.composition.formula,
            "substrate_formula": substrate_pmg.composition.formula,
            "film_miller": match.film_miller,
            "substrate_miller": match.substrate_miller,
            "von_mises_strain": match.von_mises_strain,
            "match_area": match.match_area,
            "terminations": terminations,
        }

    def apply(
        self,
        *,
        film: Union[Atoms, PmgStructure, Structure],
        substrate: Union[Atoms, PmgStructure, Structure],
        miller_film: Tuple[int, int, int] = (1, 0, 0),
        miller_substrate: Tuple[int, int, int] = (1, 1, 1),
        termination=0,
        termination_film: Optional[Union[int, str]] = None,
        termination_substrate: Optional[Union[int, str]] = None,
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
            Which termination to use (backward-compatible).  An integer
            selects by index from the pair list, a string matches by label
            substring, ``"all"`` uses the first termination.  Ignored when
            ``termination_film`` or ``termination_substrate`` is given.
        termination_film
            Select the film termination independently.  An integer indexes
            into unique film labels, a string matches by substring.
            ``None`` means "match any".
        termination_substrate
            Select the substrate termination independently.  Same semantics
            as ``termination_film``.
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

        # ---- Molecule detection + pseudo-atom replacement ---------------
        build_film, film_molecules, film_templates = self._prepare_bulk(
            film_pmg, preserve_molecules, mol_tol, mol_min_size,
        )
        build_sub, sub_molecules, sub_templates = self._prepare_bulk(
            substrate_pmg, preserve_molecules, mol_tol, mol_min_size,
        )

        # ---- ZSL matching ------------------------------------------------
        match, analyzer = self._find_zsl_match(
            build_film, build_sub,
            miller_film, miller_substrate,
            max_area, max_length_tol, max_angle_tol,
        )

        # ---- Interface construction --------------------------------------
        CoherentInterfaceBuilder = _get_pymatgen_types()["CoherentInterfaceBuilder"]
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

        # Select termination — independent flags take priority
        if termination_film is not None or termination_substrate is not None:
            selected = self._resolve_termination_pair(
                all_terms, termination_film, termination_substrate,
            )
        else:
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

            # Re-wrap with the updated lattice so fractional coords stay
            # inside [0, 1) — gap recalculation may shift molecules past
            # the original cell boundary.
            all_cart = _wrap_molecules(all_cart, all_mol_ids, new_lattice)

            # Convert Cartesian to fractional for final structure
            M_inv = np.linalg.inv(new_lattice.matrix)
            all_frac = [c @ M_inv for c in all_cart]

            PmgStructure = _get_pymatgen_types()["PmgStructure"]
            interface = PmgStructure(
                new_lattice, all_species, all_frac,
                coords_are_cartesian=False,
            )
        else:
            # No molecule preservation — just wrap atoms
            species = [site.specie for site in interface]
            frac_coords = [site.frac_coords % 1.0 for site in interface]
            PmgStructure = _get_pymatgen_types()["PmgStructure"]
            interface = PmgStructure(interface.lattice, species, frac_coords)

        return interface

    # ------------------------------------------------------------------
    # Termination selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_termination(terminations, selection):
        """Select a termination pair by index, label substring, or 'all'."""
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
    def _resolve_termination_pair(
        terminations: List[Tuple[str, str]],
        film_selection: Optional[Union[int, str]] = None,
        substrate_selection: Optional[Union[int, str]] = None,
    ) -> Tuple[str, str]:
        """Resolve independent film/substrate selections to a termination pair.

        Parameters
        ----------
        terminations
            Full list of ``(film_label, sub_label)`` pairs.
        film_selection
            Index into unique film labels, or label substring.
            ``None`` matches any film termination.
        substrate_selection
            Index into unique substrate labels, or label substring.
            ``None`` matches any substrate termination.

        Returns
        -------
        The matching ``(film_label, sub_label)`` pair.
        """
        # Ordered unique labels (preserves first-seen order)
        film_labels = list(dict.fromkeys(t[0] for t in terminations))
        sub_labels = list(dict.fromkeys(t[1] for t in terminations))

        def _resolve_one(selection, labels, kind):
            if selection is None:
                return None  # wildcard
            if isinstance(selection, int):
                if 0 <= selection < len(labels):
                    return labels[selection]
                raise ValueError(
                    f"{kind} index {selection} out of range "
                    f"[0, {len(labels) - 1}]. "
                    f"Available {kind.lower()} terminations: {labels}"
                )
            sel_str = str(selection)
            matched = [l for l in labels if sel_str in l]
            if matched:
                return matched[0]
            raise ValueError(
                f"No {kind.lower()} termination matching '{sel_str}'. "
                f"Available: {labels}"
            )

        film_target = _resolve_one(film_selection, film_labels, "Film")
        sub_target = _resolve_one(substrate_selection, sub_labels, "Substrate")

        for pair in terminations:
            if (film_target is None or pair[0] == film_target) and \
               (sub_target is None or pair[1] == sub_target):
                return pair

        raise ValueError(
            f"No termination pair found for film={film_target!r}, "
            f"substrate={sub_target!r}. "
            f"Available pairs: {terminations}"
        )

    @staticmethod
    def _to_pymatgen(obj: Union[Atoms, PmgStructure, Structure]) -> PmgStructure:
        """Coerce various structure types to pymatgen Structure."""
        pmg_types = _get_pymatgen_types()
        PmgStructure = pmg_types["PmgStructure"]
        AseAtomsAdaptor = pmg_types["AseAtomsAdaptor"]

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

def _add_interface_common_args(p) -> None:
    """Shared argument definitions for interface CLI commands."""
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


def _parse_termination_value(value):
    """Try to parse a termination CLI value as int, falling back to string."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _cmd_list(args) -> None:
    """CLI handler: list all interface terminations."""
    import json

    from mmkit.io.reader import read_structure

    film = read_structure(args.film)
    substrate = read_structure(args.substrate)

    builder = InterfaceBuilder()
    result = builder.list_terminations(
        film=film,
        substrate=substrate,
        miller_film=tuple(args.miller_film),
        miller_substrate=tuple(args.miller_substrate),
        max_area=args.max_area,
        max_length_tol=args.max_length_tol,
        max_angle_tol=args.max_angle_tol,
        termination_ftol=args.termination_ftol,
        preserve_molecules=not args.no_preserve_molecules,
        mol_tol=args.mol_tol,
        mol_min_size=args.mol_min_size,
    )

    # Print header
    print(f"Film: {args.film}")
    print(f"  Formula: {result['film_formula']}")
    print(f"  Miller: ({' '.join(str(x) for x in result['film_miller'])})")
    print()
    print(f"Substrate: {args.substrate}")
    print(f"  Formula: {result['substrate_formula']}")
    print(f"  Miller: ({' '.join(str(x) for x in result['substrate_miller'])})")
    print()
    print("ZSL Match:")
    print(f"  von Mises strain: {result['von_mises_strain']:.6f}")
    print(f"  Match area: {result['match_area']:.2f} A^2")
    print()

    terms = result["terminations"]
    if not terms:
        print("No terminations found.")
        return

    film_labels = list(dict.fromkeys(t.film_label for t in terms))
    sub_labels = list(dict.fromkeys(t.substrate_label for t in terms))

    print(f"Found {len(terms)} termination(s):\n")

    # Table
    hdr = (
        f"  {'#':<4} {'Film Label':<24} {'Substrate Label':<24} "
        f"{'Film Shift':>11} {'Sub Shift':>10}"
    )
    print(hdr)
    print(f"  {'-' * 4} {'-' * 24} {'-' * 24} {'-' * 11} {'-' * 10}")

    for t in terms:
        print(
            f"  {t.index:<4} {t.film_label:<24} {t.substrate_label:<24} "
            f"{t.film_shift:>11.4f} {t.substrate_shift:>10.4f}"
        )

    # Unique labels summary
    print()
    print(f"  Unique film terminations ({len(film_labels)}): "
          f"{', '.join(film_labels)}")
    print(f"  Unique substrate terminations ({len(sub_labels)}): "
          f"{', '.join(sub_labels)}")

    # Optional JSON output
    if args.json:
        out = [
            {
                "index": t.index,
                "film_label": t.film_label,
                "substrate_label": t.substrate_label,
                "film_shift": t.film_shift,
                "substrate_shift": t.substrate_shift,
            }
            for t in terms
        ]
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nJSON saved to {args.json}")


def _cmd_build(args):
    """CLI handler: build a coherent interface."""
    from pathlib import Path

    from mmkit.io.reader import read_structure
    from mmkit.io.writer import write_structure

    film_atoms = read_structure(args.film)
    substrate_atoms = read_structure(args.substrate)

    term_film = _parse_termination_value(
        getattr(args, "termination_film", None)
    )
    term_sub = _parse_termination_value(
        getattr(args, "termination_substrate", None)
    )

    builder = InterfaceBuilder()
    interface = builder.apply(
        film=film_atoms,
        substrate=substrate_atoms,
        miller_film=tuple(args.miller_film),
        miller_substrate=tuple(args.miller_substrate),
        termination=args.termination,
        termination_film=term_film,
        termination_substrate=term_sub,
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
    interface = subparsers.add_parser(
        "interface", 
        help="Build coherent interfaces",
        description=(
            "Build coherent interfaces between two bulk crystals using the "
            "Zur algorithm.  Supports molecule preservation and flexible "
            "termination selection."
        ),
        epilog="Hint: it is very important in many cases to choose and check the termination of each slab for modelling.\n" \
        "Unless explicitly requested, avoid creating interfaces by cutting through strong chemical bonds.\n" \
        "Prefer cleavage planes that minimize bond breaking and preserve stable coordination environments on both sides of the interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,  # preserve newlines
    )
    iface_sub = interface.add_subparsers(dest="action", required=True)

    # --- list subcommand ---------------------------------------------------
    p_list = iface_sub.add_parser(
        "list", help="List available interface terminations",
    )
    _add_interface_common_args(p_list)
    p_list.add_argument(
        "--termination-ftol", type=float, default=0.25,
        help="Tolerance for distinguishing termination planes (A)",
    )
    p_list.add_argument("--json", help="Save results as JSON")
    p_list.set_defaults(handler=_cmd_list)

    # --- build subcommand --------------------------------------------------
    p = iface_sub.add_parser(
        "build", help="Build a coherent ZSL-matched interface",
    )
    _add_interface_common_args(p)
    p.add_argument(
        "--termination", default=0,
        help="Termination index, label, or 'all' (default: 0)",
    )
    p.add_argument(
        "--termination-film", default=None,
        help="Select film termination independently (index or label substring)",
    )
    p.add_argument(
        "--termination-substrate", default=None,
        help="Select substrate termination independently (index or label substring)",
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
    p.add_argument(
        "--output", "-o",
        help="Output file (default: <film>-<substrate>_interface.extxyz)",
    )
    p.set_defaults(handler=_cmd_build)
