#!/usr/bin/env python3
"""Surface Modeler — build slabs with termination control.

Object-oriented refactoring of the surface builder pipeline:
  - ``TerminationAnalyzer``: discover all distinct terminations for a
    given bulk structure and Miller index.
  - ``MoleculeDetector``: detect molecular fragments in the bulk.
  - ``MoleculeRepair``: reconstruct cut molecules at slab surfaces.
  - ``SurfaceBuilder``: high-level Operation that ties everything together.

CLI subcommands:
  list_terminations  Discover all terminations for bulk + (hkl)
  build_slab         Build a slab with a specific termination
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from ase import Atoms
from ase.build import surface as ase_surface

from mckit.core.conversion import StructureLike, to_pymatgen_structure
from mckit.core.tool import Operation
from mckit.operate.molecule_utils import MoleculeDetector, build_molecule_templates, pbc_center
from mckit.io.writer import write_atoms


def _get_pymatgen_types():
    """Import pymatgen types lazily so the module loads without pymatgen."""
    from pymatgen.core import Lattice as PmgLattice
    from pymatgen.core import Structure as PmgStructure
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    return {
        "PmgLattice": PmgLattice,
        "PmgStructure": PmgStructure,
        "AseAtomsAdaptor": AseAtomsAdaptor,
        "SpacegroupAnalyzer": SpacegroupAnalyzer,
    }


def _has_overlap(
    new_pos: np.ndarray,
    placed: List[np.ndarray],
    tolerance: float = 0.7,
) -> bool:
    """Check if a new atom overlaps with any already-placed atom.

    Uses a minimum distance threshold to detect clashing positions.

    Parameters
    ----------
    new_pos : np.ndarray
        Cartesian position of the candidate atom.
    placed : list of np.ndarray
        Positions of atoms already placed in the slab.
    tolerance : float
        Minimum allowed distance (Angstroms).

    Returns
    -------
    bool
        True if overlap detected (atom would be too close).
    """
    for existing_pos in placed:
        d = np.linalg.norm(new_pos - existing_pos)
        if d < tolerance:  # Absolute minimum distance
            return True
    return False


# ===================================================================
# Termination dataclass
# ===================================================================

@dataclass
class Termination:
    """A single surface termination discovered by the analyzer.

    Attributes
    ----------
    label : str
        Human-readable label (e.g. ``"SrO"``, ``"TiO2"``).
    top_label : str
        Composition label of the top surface.
    bot_label : str
        Composition label of the bottom surface.
    top_comp : dict
        Element counts at the top surface margin.
    bot_comp : dict
        Element counts at the bottom surface margin.
    slab : Atoms
        ASE Atoms object representing the slab.
    shift : int
        Plane-shift index used to extract this termination.
    n_atoms : int
        Number of atoms in the slab.
    symmetric : bool
        Whether top and bottom surfaces have the same composition.
    slab_thickness : float
        Slab thickness in Angstroms (z-max - z-min).
    """

    label: str
    top_label: str
    bot_label: str
    top_comp: Dict[str, int]
    bot_comp: Dict[str, int]
    slab: Atoms
    shift: int
    n_atoms: int
    symmetric: bool
    slab_thickness: float
    molecules_cut: List = field(default_factory=list)


# ===================================================================
# Termination Analyzer
# ===================================================================

class TerminationAnalyzer:
    """Discover all distinct surface terminations for a given (hkl) plane.

    The algorithm builds a thick reference slab via ASE's surface builder,
    identifies atomic planes by z-levels of a reference element, then slides
    an *n*-plane window across all possible starting positions to enumerate
    distinct terminations.

    Parameters
    ----------
    bulk : PmgStructure
        Bulk crystal structure (pymatgen).
    miller : tuple of int
        Miller indices (h, k, l).
    n_layers : int
        Number of atomic planes per slab.
    vacuum : float
        Vacuum thickness in Angstroms.
    margin : float
        Surface layer margin for composition labeling.
    """

    def __init__(
        self,
        bulk: PmgStructure,
        miller: Tuple[int, int, int],
        n_layers: int = 4,
        vacuum: float = 15.0,
        margin: float = 1.5,
    ):
        self.bulk = bulk
        self.miller = tuple(miller)
        self.n_layers = n_layers
        self.vacuum = vacuum
        self.margin = margin

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def find_terminations(self) -> List[Termination]:
        """Discover and return all distinct terminations."""
        conv = self._to_conventional()
        atoms_conv = conv.to_ase_atoms()

        # Build thick reference slab
        thick_layers = max(self.n_layers + 12, 20)
        thick_slab = ase_surface(atoms_conv, list(self.miller), thick_layers, vacuum=0.0)

        # Identify atomic planes
        ref_el = self._find_reference_element(thick_slab)
        plane_zs, d = self._identify_planes(thick_slab, ref_el)

        if len(plane_zs) < self.n_layers:
            thick_layers *= 2
            thick_slab = ase_surface(atoms_conv, list(self.miller), thick_layers, vacuum=0.0)
            plane_zs, d = self._identify_planes(thick_slab, ref_el)

        if len(plane_zs) < self.n_layers:
            print(
                f"WARNING: Only {len(plane_zs)} planes found, need {self.n_layers}.",
                file=sys.stderr,
            )
            return []

        # Warn if slab would be very thin
        est_thickness = (self.n_layers - 1) * d
        if est_thickness < 8.0:
            recommended = int(np.ceil(8.0 / d)) + 1
            print(
                f"NOTE: {ref_el} interlayer spacing = {d:.2f} A, "
                f"{self.n_layers} layers -> ~{est_thickness:.1f} A (very thin)",
                file=sys.stderr,
            )
            print(
                f"      Recommend --layers {recommended} for >= 8 A thickness",
                file=sys.stderr,
            )

        # Scan all possible n-plane windows
        return self._scan_windows(thick_slab, plane_zs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _to_conventional(self) -> PmgStructure:
        """Convert bulk to conventional standard cell."""
        SpacegroupAnalyzer = _get_pymatgen_types()["SpacegroupAnalyzer"]
        analyzer = SpacegroupAnalyzer(self.bulk, symprec=0.1)
        return analyzer.get_conventional_standard_structure()

    @staticmethod
    def _find_reference_element(atoms: Atoms) -> str:
        """Find a heavy element suitable for plane detection."""
        counts = Counter(atoms.symbols)
        priority = [
            "Pb", "Sn", "Ge", "Ti", "Zr", "Hf", "Fe", "Co", "Ni", "Cu",
            "Zn", "Cd", "Hg", "Bi", "Sb", "In", "Ga", "Al", "Si",
            "Ba", "Sr", "Ca", "Mg", "La", "Y",
            "W", "Mo", "V", "Nb", "Ta", "Cr", "Mn",
            "I", "Br", "Cl", "S", "Se", "Te", "O", "N",
        ]
        for el in priority:
            if el in counts:
                return el
        # Fallback: heaviest element present
        from ase.data import atomic_numbers
        return max(counts.keys(), key=lambda e: atomic_numbers.get(e, 0))

    @staticmethod
    def _identify_planes(
        thick_slab: Atoms, ref_el: str,
    ) -> Tuple[List[float], float]:
        """Identify atomic planes via reference-element z-levels.

        Returns (sorted plane z-centers, mean interlayer spacing).
        """
        ref_z = sorted(
            thick_slab.positions[i, 2]
            for i in range(len(thick_slab))
            if thick_slab[i].symbol == ref_el
        )
        if not ref_z:
            return [], 0.0

        unique_z: List[float] = []
        for z in ref_z:
            if not unique_z or abs(z - unique_z[-1]) > 0.5:
                unique_z.append(z)

        if len(unique_z) < 2:
            return unique_z, 0.0

        d = float(np.mean(np.diff(unique_z)))
        return unique_z, d

    def _extract_n_plane_slab(
        self,
        thick_slab: Atoms,
        plane_zs: List[float],
        start_idx: int,
    ) -> Atoms:
        """Extract an n-plane window from a thick slab."""
        n = self.n_layers
        d = float(np.mean(np.diff(plane_zs)))
        z_lo = plane_zs[start_idx] - d / 2
        z_hi = plane_zs[start_idx + n - 1] + d / 2

        mask = [
            i
            for i in range(len(thick_slab))
            if z_lo - 0.01 <= thick_slab.positions[i, 2] <= z_hi + 0.01
        ]

        positions = thick_slab.positions[mask].copy()
        symbols = [thick_slab[i].symbol for i in mask]

        slab_thickness = positions[:, 2].max() - positions[:, 2].min()
        new_c = slab_thickness + self.vacuum

        new_cell = np.array([
            thick_slab.cell[0],
            thick_slab.cell[1],
            [0.0, 0.0, new_c],
        ])

        slab = Atoms(
            symbols=symbols,
            positions=positions,
            cell=new_cell,
            pbc=[True, True, True],
        )
        slab.translate([0, 0, -slab.positions[:, 2].min()])
        return slab

    @staticmethod
    def _label_termination(
        atoms: Atoms, margin: float = 1.5, top: bool = True,
    ) -> Tuple[str, Dict[str, int]]:
        """Label a surface termination by its composition."""
        z = atoms.positions[:, 2]
        z_min, z_max = z.min(), z.max()
        mask = z > z_max - margin if top else z < z_min + margin

        elems = [atoms[i].symbol for i in range(len(atoms)) if mask[i]]
        comp = Counter(elems)
        parts = []
        for el in sorted(comp.keys()):
            n = comp[el]
            parts.append(f"{el}{n}" if n > 1 else el)
        return "".join(parts), dict(comp)

    def _scan_windows(
        self, thick_slab: Atoms, plane_zs: List[float],
    ) -> List[Termination]:
        """Scan all n-plane windows and deduplicate terminations."""
        max_shift = len(plane_zs) - self.n_layers
        results: Dict[Tuple[str, str], Termination] = {}

        for shift in range(max_shift + 1):
            slab = self._extract_n_plane_slab(thick_slab, plane_zs, shift)
            top_label, top_comp = self._label_termination(
                slab, self.margin, top=True,
            )
            bot_label, bot_comp = self._label_termination(
                slab, self.margin, top=False,
            )

            dedup_key = (top_label, bot_label)
            if dedup_key not in results:
                z = slab.positions[:, 2]
                slab_thickness = z.max() - z.min()
                sym = top_comp == bot_comp
                label = (
                    top_label if top_label != bot_label else f"{top_label}_sym"
                )
                results[dedup_key] = Termination(
                    label=label,
                    top_label=top_label,
                    bot_label=bot_label,
                    top_comp=top_comp,
                    bot_comp=bot_comp,
                    slab=slab,
                    shift=shift,
                    n_atoms=len(slab),
                    symmetric=sym,
                    slab_thickness=slab_thickness,
                )

        # Disambiguate labels when top labels collide
        sorted_results = sorted(results.values(), key=lambda r: r.shift)
        top_counts = Counter(r.top_label for r in sorted_results)
        for r in sorted_results:
            if top_counts[r.top_label] > 1:
                r.label = f"{r.top_label}_bot{r.bot_label}"
            elif r.top_label == r.bot_label:
                r.label = f"{r.top_label}_sym"
            else:
                r.label = r.top_label

        return sorted(sorted_results, key=lambda r: r.label)


# ===================================================================
# Molecule Repair
# ===================================================================

class MoleculeRepair:
    """Repair molecular integrity at slab surfaces using center-based
    reconstruction.

    After a slab is cut, some molecules at the surface may be truncated.
    This class detects which surface molecular fragments are "well inside"
    the inorganic framework and reconstructs truncated ones from templates
    extracted from the bulk.

    Parameters
    ----------
    bulk : PmgStructure
        Bulk structure (source of molecular templates).
    molecules : list of list of int
        Molecular fragments (atom-index lists from ``MoleculeDetector``).
    mol_extent : float
        Approximate molecular radius for edge decisions (Angstroms).
    margin : float
        Margin from the inorganic framework boundary.
    strip_inorganic : bool
        Strip inorganic atoms beyond the molecular extent.
    organic_elements : set of str, optional
        Override the set of organic element symbols.
    """

    def __init__(
        self,
        bulk: PmgStructure,
        molecules: List[List[int]],
        mol_extent: float = 2.5,
        margin: float = 1.5,
        strip_inorganic: bool = True,
        organic_elements: Optional[Set[str]] = None,
    ):
        self.bulk = bulk
        self.molecules = molecules
        self.mol_extent = mol_extent
        self.margin = margin
        self.strip_inorganic = strip_inorganic

        if organic_elements is not None:
            self.organic_elements = organic_elements
        else:
            self.organic_elements = set()
            for mol in molecules:
                for idx in mol:
                    self.organic_elements.add(str(bulk[idx].specie))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def repair(self, slab: PmgStructure) -> Tuple[PmgStructure, Dict[str, int]]:
        """Repair molecules in a slab.

        Returns
        -------
        repaired_slab : PmgStructure
        report : dict
            Keys: kept, reconstructed, removed, n_atoms.
        """
        # Store templates for per-fragment matching
        self._templates = self._build_mol_templates()
        # Compute rigid-body rotation: bulk → slab frame
        R_bulk2slab, t_bulk2slab = self._compute_bulk_to_slab_transform(slab)

        slab_org = [
            i for i, s in enumerate(slab)
            if str(s.specie) in self.organic_elements
        ]
        slab_mols = self._cluster_atoms_by_distance(slab, slab_org)

        fw_min, fw_max = self._inorganic_z_bounds(slab)

        # Build a list of all periodic images of bulk molecules
        # Each entry: (bulk_mol_idx, image_center_cart, template_idx)
        bulk_mol_images = []
        image_ranges = [
            range(
                -int(np.ceil(slab.lattice.abc[i] / self.bulk.lattice.abc[i])) - 3,
                int(np.ceil(slab.lattice.abc[i] / self.bulk.lattice.abc[i])) + 4,
            )
            for i in range(3)
        ]
        for mol_idx, mol in enumerate(self.molecules):
            mol_center_frac = self._pbc_center(self.bulk, mol)
            mol_center_cart = self.bulk.lattice.get_cartesian_coords(mol_center_frac)
            # Generate periodic images
            for da in image_ranges[0]:
                for db in image_ranges[1]:
                    for dc in image_ranges[2]:
                        shift = (
                            da * self.bulk.lattice.matrix[0]
                            + db * self.bulk.lattice.matrix[1]
                            + dc * self.bulk.lattice.matrix[2]
                        )
                        image_center = R_bulk2slab @ (mol_center_cart + shift) + t_bulk2slab
                        bulk_mol_images.append((mol_idx, image_center, mol_idx))

        # Match each slab fragment to the closest bulk molecule image
        fragment_to_bulk_image = []
        for frag in slab_mols:
            frag_center = self._estimate_mol_center_cart(slab, frag)
            best_img_idx = -1
            best_dist = float("inf")
            for img_idx, (_mol_idx, img_center, _tmpl_idx) in enumerate(bulk_mol_images):  # noqa: F841 - _mol_idx, _tmpl_idx unused
                d = np.linalg.norm(frag_center - img_center)
                if d < best_dist:
                    best_dist = d
                    best_img_idx = img_idx
            fragment_to_bulk_image.append((best_img_idx, best_dist))

        # Group slab fragments by bulk molecule image
        # Key: (bulk_mol_idx, image_center_tuple) to uniquely identify each image
        image_to_fragments: Dict[Tuple, List[int]] = {}
        image_centers: Dict[Tuple, np.ndarray] = {}
        for frag_idx, (img_idx, dist) in enumerate(fragment_to_bulk_image):
            if img_idx >= 0 and dist < self.mol_extent * 2:
                mol_idx, img_center, tmpl_idx = bulk_mol_images[img_idx]  # noqa: F841 - tmpl_idx unused
                key = (mol_idx, tuple(np.round(img_center, 3)))
                if key not in image_to_fragments:
                    image_to_fragments[key] = []
                    image_centers[key] = img_center
                image_to_fragments[key].append(frag_idx)

        # Start with inorganic atoms
        new_sp: List[str] = []
        new_cc: List[np.ndarray] = []
        for i, site in enumerate(slab):
            if str(site.specie) not in self.organic_elements:
                new_sp.append(str(site.specie))
                new_cc.append(
                    slab.lattice.get_cartesian_coords(slab.frac_coords[i])
                )

        kept = reconstructed = removed = 0
        # Collect positions already placed (for overlap guard)
        placed_positions: List[np.ndarray] = list(new_cc)
        # Track which slab fragments have been processed
        processed_fragments: Set[int] = set()

        # Process each bulk molecule image
        for image_key, frag_indices in image_to_fragments.items():
            mol_idx = image_key[0]
            img_center = image_centers[image_key]

            # Collect all atoms from all fragments for this image
            all_slab_atoms = []
            for frag_idx in frag_indices:
                all_slab_atoms.extend(slab_mols[frag_idx])
                processed_fragments.add(frag_idx)

            # Estimate center of the combined fragments
            center_cart = self._estimate_mol_center_cart(slab, all_slab_atoms)
            center_z = center_cart[2]

            # Check if well inside the framework
            well_inside = (
                center_z >= fw_min + self.margin
                and center_z <= fw_max - self.margin
            )

            # Check if we have all atoms (or close to it)
            expected_size = len(self.molecules[mol_idx])
            actual_size = len(all_slab_atoms)

            if well_inside and actual_size >= expected_size * 0.8:
                # Keep all fragment atoms (molecule is mostly intact)
                for idx in all_slab_atoms:
                    new_sp.append(str(slab[idx].specie))
                    new_cc.append(
                        slab.lattice.get_cartesian_coords(slab.frac_coords[idx])
                    )
                    placed_positions.append(new_cc[-1])
                kept += 1
            else:
                # Reconstruct from template.  This includes matched surface
                # fragments: preserving molecules means completing the
                # molecule in the selected image, not leaving boundary pieces.
                tmpl = self._templates[mol_idx]

                # Save position before adding atoms
                start_len = len(new_cc)
                placed_ok = True
                for spec, offset in zip(tmpl["species"], tmpl["offsets"]):
                    # Rotate bulk offset into slab frame, then translate
                    rotated_offset = R_bulk2slab @ offset
                    new_pos = center_cart + rotated_offset
                    # Overlap guard: skip if too close to existing atom
                    if _has_overlap(new_pos, placed_positions):
                        placed_ok = False
                        break
                    new_sp.append(spec)
                    new_cc.append(new_pos)
                    placed_positions.append(new_pos)

                if not placed_ok:
                    # Rollback: remove atoms added for this fragment
                    del new_sp[start_len:]
                    del new_cc[start_len:]
                    del placed_positions[start_len:]
                    removed += 1
                else:
                    reconstructed += 1

        # Handle any unassigned fragments (not matched to any bulk molecule image)
        for frag_idx, frag in enumerate(slab_mols):
            if frag_idx not in processed_fragments:
                # This fragment wasn't matched to any bulk molecule image
                # Check if it's well inside
                center_z = self._estimate_mol_center_cart(slab, frag)[2]
                well_inside = (
                    center_z >= fw_min + self.margin
                    and center_z <= fw_max - self.margin
                )
                if well_inside:
                    # Keep it
                    for idx in frag:
                        new_sp.append(str(slab[idx].specie))
                        new_cc.append(
                            slab.lattice.get_cartesian_coords(slab.frac_coords[idx])
                        )
                    kept += 1
                else:
                    removed += 1

        # Optionally strip inorganic atoms beyond molecular extent
        if self.strip_inorganic and kept + reconstructed > 0:
            new_sp, new_cc = self._strip_excess_inorganic(new_sp, new_cc)

        return self._assemble_slab(slab, new_sp, new_cc, kept, reconstructed, removed)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _inorganic_z_bounds(self, slab: PmgStructure) -> Tuple[float, float]:
        """Get z-range of inorganic atoms in the slab."""
        inorg_z = []
        for i, s in enumerate(slab):
            if str(s.specie) not in self.organic_elements:
                inorg_z.append(
                    slab.lattice.get_cartesian_coords(slab.frac_coords[i])[2]
                )
        if not inorg_z:
            return 0.0, slab.lattice.c
        return min(inorg_z), max(inorg_z)

    @staticmethod
    def _cluster_atoms_by_distance(
        structure: PmgStructure, indices: List[int], cutoff: float = 1.6,
    ) -> List[List[int]]:
        """Cluster atoms by pairwise distance connectivity."""
        def _pbc_distance(i: int, j: int) -> float:
            delta = np.asarray(structure.frac_coords[i]) - np.asarray(
                structure.frac_coords[j]
            )
            delta -= np.round(delta)
            cart_delta = structure.lattice.get_cartesian_coords(delta)
            return float(np.linalg.norm(cart_delta))

        visited: Set[int] = set()
        clusters: List[List[int]] = []
        for idx in indices:
            if idx in visited:
                continue
            cluster: List[int] = []
            queue = [idx]
            while queue:
                cur = queue.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                cluster.append(cur)
                for other in indices:
                    if other not in visited:
                        dist = _pbc_distance(cur, other)
                        if dist < cutoff:
                            queue.append(other)
            clusters.append(cluster)
        return clusters

    @staticmethod
    def _pbc_center(structure: PmgStructure, indices: List[int]) -> np.ndarray:
        """PBC-aware geometric center (fractional coords)."""
        return pbc_center(structure, indices)

    def _build_mol_templates(self) -> List[Dict]:
        """Extract Cartesian offset templates for each detected molecule."""
        return build_molecule_templates(self.bulk, self.molecules)

    @staticmethod
    def _compute_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
        """RMSD between two (N,3) coordinate arrays."""
        if len(P) == 0:
            return float("inf")
        return float(np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1))))

    @staticmethod
    def _kabsch(
        P: np.ndarray, Q: np.ndarray,
    ) -> np.ndarray:
        """Kabsch algorithm: optimal rotation aligning P onto Q.

        Both P and Q must be centroid-centered (N×3 arrays).
        Returns a 3×3 rotation matrix R such that ``P @ R.T ≈ Q``.
        """
        H = P.T @ Q
        U, _S, Vt = np.linalg.svd(H)  # noqa: F841 - _S intentionally unused
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
        R = Vt.T @ sign_matrix @ U.T
        return R

    def _compute_bulk_to_slab_transform(
        self, slab: PmgStructure,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute rotation R and translation t mapping bulk → slab frame.

        Matches inorganic atoms between bulk and slab by minimum-image
        distance, then solves for the rigid-body transform via the Kabsch
        algorithm.

        Returns ``(R, t)`` where ``cart_slab ≈ R @ cart_bulk + t``.
        Falls back to identity / zero when no matches are found.
        """
        inorg_bulk_idx = []
        inorg_bulk_el = []
        for i, site in enumerate(self.bulk):
            el = str(site.specie)
            if el not in self.organic_elements:
                inorg_bulk_idx.append(i)
                inorg_bulk_el.append(el)

        inorg_slab_idx = []
        inorg_slab_el = []
        for i, site in enumerate(slab):
            el = str(site.specie)
            if el not in self.organic_elements:
                inorg_slab_idx.append(i)
                inorg_slab_el.append(el)

        if not inorg_bulk_idx or not inorg_slab_idx:
            return np.eye(3), np.zeros(3)

        bulk_cart = np.array([
            self.bulk.lattice.get_cartesian_coords(
                self.bulk.frac_coords[i],
            )
            for i in inorg_bulk_idx
        ])
        slab_cart = np.array([
            slab.lattice.get_cartesian_coords(slab.frac_coords[i])
            for i in inorg_slab_idx
        ])

        # Match slab inorganic atoms to nearest bulk inorganic atom
        bulk_matched: List[np.ndarray] = []
        slab_matched: List[np.ndarray] = []
        max_search_r = 5.0

        for s_idx in range(len(inorg_slab_idx)):
            s_pos = slab_cart[s_idx]
            s_el = inorg_slab_el[s_idx]
            best_d = max_search_r
            best_b_pos = None
            for b_idx in range(len(inorg_bulk_idx)):
                if inorg_bulk_el[b_idx] != s_el:
                    continue
                b_pos = bulk_cart[b_idx]
                # Check periodic images of bulk position
                for da in range(-1, 2):
                    for db in range(-1, 2):
                        for dc in range(-1, 2):
                            shift = (
                                da * self.bulk.lattice.matrix[0]
                                + db * self.bulk.lattice.matrix[1]
                                + dc * self.bulk.lattice.matrix[2]
                            )
                            image_pos = b_pos + shift
                            d = np.linalg.norm(s_pos - image_pos)
                            if d < best_d:
                                best_d = d
                                best_b_pos = image_pos
            if best_b_pos is not None:
                bulk_matched.append(best_b_pos)
                slab_matched.append(s_pos)

        if len(bulk_matched) < 3:
            return np.eye(3), np.zeros(3)

        bulk_arr = np.array(bulk_matched)
        slab_arr = np.array(slab_matched)

        bulk_centroid = bulk_arr.mean(axis=0)
        slab_centroid = slab_arr.mean(axis=0)

        R = self._kabsch(bulk_arr - bulk_centroid, slab_arr - slab_centroid)
        t = slab_centroid - R @ bulk_centroid
        return R, t

    def _find_best_template(
        self, slab: PmgStructure, mol_indices: List[int],
    ) -> Tuple[int, float]:
        """Find best-matching template for a fragment via heavy-atom RMSD.

        Returns ``(template_index, rmsd)``.
        """
        heavy_elems = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
        frag_heavy = [
            i for i in mol_indices
            if str(slab[i].specie) in heavy_elems
        ]
        if not frag_heavy:
            return 0, float("inf")

        frag_coords = np.array([
            slab.lattice.get_cartesian_coords(slab.frac_coords[i])
            for i in frag_heavy
        ])
        frag_elems = [str(slab[i].specie) for i in frag_heavy]
        frag_centroid = frag_coords.mean(axis=0)

        best_idx = 0
        best_rmsd = float("inf")
        for t_idx, tmpl in enumerate(self._templates):
            tmpl_heavy_mask = [
                s in heavy_elems for s in tmpl["species"]
            ]
            if sum(tmpl_heavy_mask) < len(frag_heavy):
                continue
            tmpl_offsets_heavy = tmpl["offsets"][tmpl_heavy_mask]
            tmpl_elems_heavy = [
                s for s, m in zip(tmpl["species"], tmpl_heavy_mask) if m
            ]
            # Match element composition
            if sorted(tmpl_elems_heavy) != sorted(frag_elems):
                continue
            # Align by element order
            tmpl_centered = tmpl_offsets_heavy.copy()
            rmsd = self._compute_rmsd(frag_coords - frag_centroid, tmpl_centered)
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_idx = t_idx

        return best_idx, best_rmsd

    @staticmethod
    def _estimate_mol_center_cart(
        structure: PmgStructure, mol_indices: List[int],
    ) -> np.ndarray:
        """Estimate molecular center in Cartesian coordinates."""
        coords_cart = np.array([
            structure.lattice.get_cartesian_coords(structure.frac_coords[k])
            for k in mol_indices
        ])
        elems = [str(structure[k].specie) for k in mol_indices]

        if len(mol_indices) >= 8:
            return coords_cart.mean(axis=0)

        c_pos = [coords_cart[j] for j, e in enumerate(elems) if e == "C"]
        n_pos = [coords_cart[j] for j, e in enumerate(elems) if e == "N"]

        if c_pos and n_pos:
            return (np.mean(c_pos, axis=0) + np.mean(n_pos, axis=0)) / 2.0
        elif c_pos:
            return np.mean(c_pos, axis=0)
        elif n_pos:
            return np.mean(n_pos, axis=0)
        else:
            return coords_cart.mean(axis=0)

    def _strip_excess_inorganic(
        self, sp: List[str], cc: List[np.ndarray],
    ) -> Tuple[List[str], List[np.ndarray]]:
        """Remove inorganic atoms outside the organic z-range."""
        org_z = [cc[i][2] for i, s in enumerate(sp) if s in self.organic_elements]
        if not org_z:
            return sp, cc
        mol_min, mol_max = min(org_z), max(org_z)

        filtered_sp: List[str] = []
        filtered_cc: List[np.ndarray] = []
        for i, s in enumerate(sp):
            z = cc[i][2]
            if s not in self.organic_elements and (z < mol_min or z > mol_max):
                continue
            filtered_sp.append(s)
            filtered_cc.append(cc[i])
        return filtered_sp, filtered_cc

    @staticmethod
    def _assemble_slab(
        slab: PmgStructure,
        new_sp: List[str],
        new_cc: List[np.ndarray],
        kept: int,
        reconstructed: int,
        removed: int,
    ) -> Tuple[PmgStructure, Dict[str, int]]:
        """Assemble the final repaired slab structure."""
        new_cc_arr = np.array(new_cc)
        z_min = new_cc_arr[:, 2].min()
        z_max = new_cc_arr[:, 2].max()
        thickness = z_max - z_min
        vacuum = slab.lattice.c - (
            slab.cart_coords[:, 2].max() - slab.cart_coords[:, 2].min()
        )
        new_c = thickness + max(vacuum, 15.0)
        new_cc_arr[:, 2] -= z_min

        new_lat = _get_pymatgen_types()["PmgLattice"].from_parameters(
            slab.lattice.a, slab.lattice.b, new_c,
            slab.lattice.alpha, slab.lattice.beta, slab.lattice.gamma,
        )
        new_frac = new_lat.get_fractional_coords(new_cc_arr) % 1.0
        new_frac[np.isclose(new_frac, 1.0, atol=1e-8)] = 0.0

        z_ord = np.argsort(new_frac[:, 2])
        final = _get_pymatgen_types()["PmgStructure"](
            new_lat,
            [new_sp[i] for i in z_ord],
            [new_frac[i] for i in z_ord],
        )

        report = {
            "kept": kept,
            "reconstructed": reconstructed,
            "removed": removed,
            "n_atoms": len(final),
        }
        return final, report


# ===================================================================
# Surface Builder (high-level Operation)
# ===================================================================

class SurfaceBuilder(Operation):
    """Build slab models with termination control.

    Ties together ``TerminationAnalyzer``, ``MoleculeDetector``, and
    ``MoleculeRepair`` into a single ``Operation``-compatible interface.

    Parameters
    ----------
    vacuum : float
        Vacuum thickness in Angstroms.
    n_layers : int
        Number of atomic planes per slab.
    margin : float
        Surface layer margin for composition labeling.
    preserve_molecules : bool
        Detect and reconstruct cut molecules at surfaces.
    mol_tol : float
        Bond-detection tolerance for molecule detection.
    mol_min_size : int
        Minimum atoms to count as a molecule.
    mol_extent : float
        Approximate molecular radius for edge decisions.
    strip_inorganic : bool
        Strip inorganic atoms beyond molecular extent.

    Example
    -------
    >>> builder = SurfaceBuilder(n_layers=6, vacuum=15.0, preserve_molecules=True)
    >>> terminations = builder.find_terminations(bulk_path="SrTiO3.cif", miller=(0,0,1))
    >>> slab = builder.build_slab(bulk_path="SrTiO3.cif", miller=(0,0,1), termination=0)
    """

    def __init__(
        self,
        vacuum: float = 15.0,
        n_layers: int = 4,
        margin: float = 1.5,
        preserve_molecules: bool = True,
        mol_tol: float = 0.45,
        mol_min_size: int = 2,
        mol_extent: float = 2.5,
        strip_inorganic: bool = True,
    ):
        self.vacuum = vacuum
        self.n_layers = n_layers
        self.margin = margin
        self.preserve_molecules = preserve_molecules
        self.mol_tol = mol_tol
        self.mol_min_size = mol_min_size
        self.mol_extent = mol_extent
        self.strip_inorganic = strip_inorganic

    # ------------------------------------------------------------------
    # Bulk loading
    # ------------------------------------------------------------------
    @staticmethod
    def load_bulk(path: str) -> PmgStructure:
        """Load a bulk structure file and return a pymatgen ``Structure``.

        Delegates to :func:`matmod.io.reader.read_structure`, which already
        chooses pymatgen's ``CifParser`` for ``.cif`` and ASE otherwise.
        """
        from mckit.io.reader import read_structure
        from pymatgen.io.ase import AseAtomsAdaptor

        bulk = AseAtomsAdaptor().get_structure(read_structure(path))
        PmgStructure = _get_pymatgen_types()["PmgStructure"]
        return PmgStructure(
            bulk.lattice,
            [site.species for site in bulk],
            bulk.frac_coords % 1.0,
            site_properties=bulk.site_properties,
            coords_are_cartesian=False,
        )

    # ------------------------------------------------------------------
    # Termination discovery
    # ------------------------------------------------------------------
    def find_terminations(
        self,
        bulk: str | StructureLike,
        miller: Tuple[int, int, int],
    ) -> List[Termination]:
        """Load a bulk file and discover all terminations."""
        if isinstance(bulk, str):
            bulk = self.load_bulk(bulk)
        else:
            bulk = to_pymatgen_structure(bulk, copy=False)
        analyzer = TerminationAnalyzer(
            bulk=bulk,
            miller=miller,
            n_layers=self.n_layers,
            vacuum=self.vacuum,
            margin=self.margin,
        )
        return analyzer.find_terminations()

    # ------------------------------------------------------------------
    # Slab building
    # ------------------------------------------------------------------
    def build_slab(
        self,
        bulk: str | StructureLike,
        miller: Tuple[int, int, int],
        termination: str | int = 0,
        output: Optional[str] = None,
    ) -> Tuple[Atoms, str]:
        """Build a slab with a specific termination.

        Parameters
        ----------
        bulk : str or Structure
            Path to bulk structure file or a pymatgen Structure object.
        miller : tuple of int
            Miller indices.
        termination : str or int
            Termination label, index, or ``"all"``.
        output : str, optional
            Output file path.

        Returns
        -------
        slab_atoms : Atoms
        output_path : str
        """
        if isinstance(bulk, str):
            bulk = self.load_bulk(bulk)
        else:
            bulk = to_pymatgen_structure(bulk, copy=False)
        analyzer = TerminationAnalyzer(
            bulk=bulk,
            miller=miller,
            n_layers=self.n_layers,
            vacuum=self.vacuum,
            margin=self.margin,
        )
        terms = analyzer.find_terminations()

        if not terms:
            raise RuntimeError("No terminations found.")

        # Select termination
        selected = self._select_termination(terms, termination)

        # Detect molecules if needed
        molecules: List[List[int]] = []
        if self.preserve_molecules:
            detector = MoleculeDetector(tol=self.mol_tol, min_size=self.mol_min_size)
            molecules = detector.detect(bulk)

        # Build each selected slab
        results = []
        for t in selected:
            slab_atoms = t.slab  # ASE Atoms

            if self.preserve_molecules and molecules:
                AseAtomsAdaptor = _get_pymatgen_types()["AseAtomsAdaptor"]
                slab_pm = AseAtomsAdaptor().get_structure(slab_atoms)
                repair = MoleculeRepair(
                    bulk=bulk,
                    molecules=molecules,
                    mol_extent=self.mol_extent,
                    margin=self.margin,
                    strip_inorganic=self.strip_inorganic,
                )
                repaired, _ = repair.repair(slab_pm)
                slab_atoms = AseAtomsAdaptor().get_atoms(repaired)

            # Determine output path
            out_path = self._resolve_output_path(
                bulk if isinstance(bulk, str) else None, miller, t, output, len(selected) > 1,
            )
            written = write_atoms(out_path, slab_atoms)
            results.append((slab_atoms, written))

        # If single termination, return directly; else return list
        if len(results) == 1:
            return results[0]
        return results

    # ------------------------------------------------------------------
    # Operation interface
    # ------------------------------------------------------------------
    def apply(
        self,
        *,
        bulk: str | StructureLike,
        miller: Tuple[int, int, int] = (0, 0, 1),
        termination: str | int = 0,
        **_,
    ) -> Atoms:
        """Build a slab (Operation interface).

        For programmatic use.  For CLI, use ``find_terminations`` /
        ``build_slab`` directly.
        """
        slab_atoms, _path = self.build_slab(bulk, miller, termination)
        return slab_atoms

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _select_termination(
        terms: List[Termination], selection: str | int,
    ) -> List[Termination]:
        """Resolve a termination selection to a list."""
        if isinstance(selection, str) and selection == "all":
            return terms
        if isinstance(selection, int):
            if selection < 0 or selection >= len(terms):
                raise IndexError(
                    f"Index {selection} out of range (0-{len(terms) - 1})."
                )
            return [terms[selection]]
        # String label
        matched = [t for t in terms if t.top_label == selection]
        if not matched:
            available = ", ".join(t.top_label for t in terms)
            raise ValueError(
                f"No termination matches '{selection}'. Available: {available}"
            )
        return matched

    @staticmethod
    def _resolve_output_path(
        bulk_path: str,
        miller: Tuple[int, int, int],
        term: Termination,
        output: Optional[str],
        is_multi: bool,
    ) -> str:
        """Determine the output file path for a slab."""
        if is_multi:
            hkl = "".join(str(x) for x in miller)
            out_dir = Path(output).parent if output else Path(bulk_path).parent
            out_dir = out_dir or Path(".")
            return str(
                out_dir / f"{Path(bulk_path).stem}_{hkl}_{term.label}.extxyz"
            )
        return output or f"slab_{term.top_label}.extxyz"


# ===================================================================
# CLI — thin wrapper around the OOP classes
# ===================================================================

def cmd_list_terminations(args) -> None:
    """CLI handler: list all terminations."""
    builder = SurfaceBuilder(
        vacuum=args.vacuum,
        n_layers=args.layers,
        margin=args.margin,
        preserve_molecules=not args.no_preserve_molecules,
        mol_tol=args.mol_tol,
        mol_min_size=args.mol_min_size,
        mol_extent=args.mol_extent,
        strip_inorganic=not args.no_strip_inorganic,
    )

    bulk = SurfaceBuilder.load_bulk(args.input)
    miller = tuple(args.miller)

    print(f"Bulk: {args.input}")
    print(f"  Formula: {bulk.composition.formula}")
    print(f"  Lattice: {bulk.lattice.a:.3f} x {bulk.lattice.b:.3f} x {bulk.lattice.c:.3f}")
    print(f"  Atoms: {len(bulk)}")
    print(f"Surface: ({' '.join(str(x) for x in miller)})")
    print(f"  layers: {args.layers}, vacuum: {args.vacuum} A")
    if not args.no_preserve_molecules:
        print("  Molecular preservation: ON")
    print()

    analyzer = TerminationAnalyzer(
        bulk=bulk,
        miller=miller,
        n_layers=builder.n_layers,
        vacuum=builder.vacuum,
        margin=builder.margin,
    )
    terms = analyzer.find_terminations()

    if not terms:
        print("No terminations found.")
        return

    print(f"Found {len(terms)} termination(s):\n")
    hdr = (
        f"  {'#':<4} {'Top Label':<20} {'Bot Label':<16} "
        f"{'Atoms':>6} {'Sym':>5} {'Thick(A)':>9} {'Shift'}"
    )
    print(hdr)
    print(f"  {'-' * 4} {'-' * 20} {'-' * 16} {'-' * 6} {'-' * 5} {'-' * 9} {'-' * 20}")

    for i, t in enumerate(terms):
        sym_str = "Y" if t.symmetric else "N"
        print(
            f"  {i:<4} {t.top_label:<20} {t.bot_label:<16} "
            f"{t.n_atoms:>6} {sym_str:>5} {t.slab_thickness:>9.2f} {t.shift}"
        )

    print()
    for i, t in enumerate(terms):
        top = ", ".join(f"{el}:{n}" for el, n in sorted(t.top_comp.items()))
        bot = ", ".join(f"{el}:{n}" for el, n in sorted(t.bot_comp.items()))
        print(f"  [{i}] Top: {t.top_label} {{{top}}}  |  Bot: {t.bot_label} {{{bot}}}")

    if args.json:
        out = [
            {
                "index": i,
                "label": t.top_label,
                "bot_label": t.bot_label,
                "top_comp": t.top_comp,
                "bot_comp": t.bot_comp,
                "n_atoms": t.n_atoms,
                "symmetric": t.symmetric,
                "slab_thickness": t.slab_thickness,
                "shift": t.shift,
            }
            for i, t in enumerate(terms)
        ]
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nJSON saved to {args.json}")


def cmd_build_slab(args) -> None:
    """CLI handler: build a slab."""
    builder = SurfaceBuilder(
        vacuum=args.vacuum,
        n_layers=args.layers,
        margin=args.margin,
        preserve_molecules=not args.no_preserve_molecules,
        mol_tol=args.mol_tol,
        mol_min_size=args.mol_min_size,
        mol_extent=args.mol_extent,
        strip_inorganic=not args.no_strip_inorganic,
    )

    bulk = SurfaceBuilder.load_bulk(args.input)
    miller = tuple(args.miller)

    analyzer = TerminationAnalyzer(
        bulk=bulk,
        miller=miller,
        n_layers=builder.n_layers,
        vacuum=builder.vacuum,
        margin=builder.margin,
    )
    terms = analyzer.find_terminations()

    if not terms:
        print("ERROR: No terminations found.", file=sys.stderr)
        sys.exit(1)

    # Resolve selection
    selection = args.termination
    if selection == "all":
        selected = terms
    else:
        try:
            idx = int(selection)
            if idx < 0 or idx >= len(terms):
                print(
                    f"ERROR: Index {idx} out of range (0-{len(terms) - 1}).",
                    file=sys.stderr,
                )
                sys.exit(1)
            selected = [terms[idx]]
        except ValueError:
            matched = [t for t in terms if t.top_label == selection]
            if not matched:
                print(
                    f"ERROR: No termination matches '{selection}'.",
                    file=sys.stderr,
                )
                print(
                    "Available:",
                    ", ".join(t.top_label for t in terms),
                    file=sys.stderr,
                )
                sys.exit(1)
            selected = matched

    # Detect molecules in bulk
    molecules: List[List[int]] = []
    if not args.no_preserve_molecules:
        detector = MoleculeDetector(tol=args.mol_tol, min_size=args.mol_min_size)
        molecules = detector.detect(bulk)

    for t in selected:
        label = t.top_label
        is_multi = selection == "all"

        if is_multi:
            hkl = "".join(str(x) for x in miller)
            out_dir = (
                Path(args.output).parent if args.output else Path(args.input).parent
            )
            out_dir = out_dir or Path(".")
            output = str(
                out_dir / f"{Path(args.input).stem}_{hkl}_{t.label}.extxyz"
            )
        else:
            output = args.output or f"slab_{label}.extxyz"

        slab_atoms = t.slab

        if not args.no_preserve_molecules and molecules:
            AseAtomsAdaptor = _get_pymatgen_types()["AseAtomsAdaptor"]
            slab_pm = AseAtomsAdaptor().get_structure(slab_atoms)
            repair = MoleculeRepair(
                bulk=bulk,
                molecules=molecules,
                mol_extent=args.mol_extent,
                margin=args.margin,
                strip_inorganic=not args.no_strip_inorganic,
            )
            repaired, report = repair.repair(slab_pm)
            slab_atoms = AseAtomsAdaptor().get_atoms(repaired)
            path = write_atoms(output, slab_atoms)

            # Check symmetry of repaired slab
            z = repaired.cart_coords[:, 2]
            z_min, z_max = z.min(), z.max()
            top_comp: Dict[str, int] = {}
            bot_comp: Dict[str, int] = {}
            for j, site in enumerate(repaired):
                el = str(site.specie)
                if z[j] > z_max - args.margin:
                    top_comp[el] = top_comp.get(el, 0) + 1
                elif z[j] < z_min + args.margin:
                    bot_comp[el] = bot_comp.get(el, 0) + 1
            sym = "symmetric" if top_comp == bot_comp else "asymmetric"
            print(
                f"  {label}: {report['n_atoms']} atoms, {sym}, "
                f"molecules preserved -> {path}"
            )
            print(
                f"    Molecules: kept={report['kept']}, "
                f"reconstructed={report['reconstructed']}, "
                f"removed={report['removed']}"
            )
        else:
            path = write_atoms(output, slab_atoms)
            sym = "symmetric" if t.symmetric else "asymmetric"
            print(
                f"  {label}: {t.n_atoms} atoms, {sym}, "
                f"{t.slab_thickness:.2f} A -> {path}"
            )


def _add_surface_args(p) -> None:
    """Shared argument definitions for surface CLI commands."""
    p.add_argument("input", help="Bulk structure file (CIF, POSCAR, extxyz, ...)")
    p.add_argument(
        "--miller", type=int, nargs=3, required=True,
        help="Miller indices (h k l)",
    )
    p.add_argument(
        "--layers", type=int, default=4,
        help="Number of atomic layers (default: 4)",
    )
    p.add_argument(
        "--vacuum", type=float, default=15.0,
        help="Vacuum thickness (A)",
    )
    p.add_argument(
        "--margin", type=float, default=1.5,
        help="Surface layer margin (A)",
    )
    p.add_argument(
        "--no-preserve-molecules", action="store_true",
        help="Disable molecular integrity preservation",
    )
    p.add_argument(
        "--mol-tol", type=float, default=0.45,
        help="Bond detection tolerance (A)",
    )
    p.add_argument(
        "--mol-min-size", type=int, default=2,
        help="Min atoms to count as molecule",
    )
    p.add_argument(
        "--mol-extent", type=float, default=2.5,
        help="Approximate molecular radius for edge decisions (A)",
    )
    p.add_argument(
        "--no-strip-inorganic", action="store_true",
        help="Don't strip inorganic atoms beyond molecular extent",
    )


def register_cli(subparsers) -> None:
    """Register surface subcommands with the mmkit CLI."""
    surface = subparsers.add_parser(
        "surface", 
        help="Surface slab tools",
        description="Discover terminations and build slabs with termination control.",
        epilog="Hint: it is very important in many cases to choose and check the surface termination for modelling.",
        # formatter_class=argparse.RawDescriptionHelpFormatter,  # preserve newlines
    )
    surface_sub = surface.add_subparsers(dest="action", required=True)

    p_list = surface_sub.add_parser("list", help="Discover all terminations")
    _add_surface_args(p_list)
    p_list.add_argument("--json", help="Save results as JSON")
    p_list.set_defaults(handler=cmd_list_terminations)

    p_build = surface_sub.add_parser("build", help="Build slab with specific termination")
    _add_surface_args(p_build)
    p_build.add_argument(
        "--termination", required=False, default=0,
        help="Termination label, index, or 'all'",
    )
    p_build.add_argument(
        "--output", help="Output file (auto-generated for 'all')",
    )
    p_build.set_defaults(handler=cmd_build_slab)


def build_parser() -> argparse.ArgumentParser:
    """Standalone parser for ``python -m mmkit.operate.surface``."""
    parser = argparse.ArgumentParser(
        prog="surface_modeler",
        description="Surface Modeler — build slabs with termination control.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="Discover all terminations")
    _add_surface_args(p_list)
    p_list.add_argument("--json", help="Save results as JSON")
    p_list.set_defaults(handler=cmd_list_terminations)

    p_build = sub.add_parser("build", help="Build slab with specific termination")
    _add_surface_args(p_build)
    p_build.add_argument(
        "--termination", required=False, default=0,
        help="Termination label, index, or 'all'",
    )
    p_build.add_argument(
        "--output", help="Output file (auto-generated for 'all')",
    )
    p_build.set_defaults(handler=cmd_build_slab)

    return parser


def main():
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
