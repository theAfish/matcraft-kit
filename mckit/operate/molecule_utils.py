"""Shared molecule utilities for operate builders."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

_PSEUDO_SYMBOL = "X"
_dummy_patched = False


def _get_pymatgen_types():
    """Import pymatgen types lazily so the module loads without pymatgen."""
    from pymatgen.core import Structure as PmgStructure
    return {"PmgStructure": PmgStructure}


def _ensure_dummy_patched() -> None:
    """Ensure DummySpecies provides atomic_mass for interface building."""
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


def pbc_center(structure, indices: List[int]) -> np.ndarray:
    """PBC-aware geometric center in fractional coordinates."""
    coords = structure.frac_coords[indices].copy()
    ref = coords[0]
    for j in range(3):
        diff = coords[:, j] - ref[j]
        coords[:, j] = ref[j] + (diff + 0.5) % 1.0 - 0.5
    return coords.mean(axis=0) % 1.0


def build_molecule_templates(
    bulk,
    molecules: List[List[int]],
) -> List[Dict]:
    """Extract Cartesian offset templates for each detected molecule."""
    templates = []
    for mol in molecules:
        center_frac = pbc_center(bulk, mol)
        center_cart = bulk.lattice.get_cartesian_coords(center_frac)
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
            "center": center_frac,
        })
    return templates


def create_pseudo_structure(
    bulk,
    molecules: List[List[int]],
    templates: List[Dict],
):
    """Replace each detected molecule with a single X pseudo-atom."""
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


class MoleculeDetector:
    """Detect molecular fragments in a bulk structure via connectivity.

    The detector is intentionally context-aware around halides.  Halides can
    be real molecular atoms, for example in aryl iodides, but in hybrid
    perovskites framework halides can sit close enough to organic cations to
    pass a simple covalent-radius cutoff.  We therefore build a valence-limited
    covalent graph first, then prune halide contacts that look shared or
    framework-like rather than terminal.
    """

    _HALOGENS: Set[str] = {"F", "Cl", "Br", "I", "At"}
    _ORGANIC_HEAVY: Set[str] = {"B", "C", "N", "O", "P", "S", "Se", "Si"}
    _NONMETALS: Set[str] = {
        "H", "He", "B", "C", "N", "O", "F", "Ne", "P", "S", "Cl", "Ar",
        "Se", "Br", "Kr", "I", "Xe", "Te", "At", "Rn",
    }
    _MAX_BONDS: Dict[str, int] = {
        "H": 1,
        "F": 1,
        "Cl": 1,
        "Br": 1,
        "I": 1,
        "At": 1,
        "O": 2,
        "S": 2,
        "Se": 2,
        "N": 3,
        "P": 3,
        "B": 3,
        "C": 4,
        "Si": 4,
    }

    def __init__(
        self,
        tol: float = 0.45,
        min_size: int = 2,
        framework_halogen_min_neighbors: int = 2,
    ):
        self.tol = tol
        self.min_size = min_size
        self.framework_halogen_min_neighbors = framework_halogen_min_neighbors

    def detect(self, bulk) -> List[List[int]]:
        """Detect and return molecular fragments as lists of atom indices."""
        n = len(bulk)
        if n == 0:
            return []

        from ase.data import covalent_radii as _ase_radii, atomic_numbers

        def _cov_r(sym: str) -> float:
            z = atomic_numbers.get(sym, 0)
            return float(_ase_radii[z]) if z > 0 else 1.5

        def _is_metal(el: str) -> bool:
            return el not in self._NONMETALS

        def _max_bonds(el: str) -> int:
            return self._MAX_BONDS.get(el, 4)

        def _bond_tolerance(el_i: str, el_j: str) -> float:
            # Avoid treating framework halides as molecular ligands through
            # weak H...X contacts while still allowing real short H-X bonds.
            if "H" in (el_i, el_j):
                other = el_j if el_i == "H" else el_i
                if other in self._HALOGENS:
                    return min(self.tol, 0.05)
            return self.tol

        def _can_consider_bond(el_i: str, el_j: str) -> bool:
            if _is_metal(el_i) or _is_metal(el_j):
                return False
            return True

        def _bond_priority(el_i: str, el_j: str) -> int:
            if "H" in (el_i, el_j):
                other = el_j if el_i == "H" else el_i
                if other in self._ORGANIC_HEAVY:
                    return 0
                if other in self._HALOGENS:
                    return 2
                return 1
            if el_i in self._HALOGENS or el_j in self._HALOGENS:
                return 2
            return 1

        max_cov_r = max(_cov_r(str(site.specie)) for site in bulk)
        cov_cutoff = 2 * (max_cov_r + self.tol)

        def _pbc_distance(i: int, j: int) -> float:
            delta = np.asarray(bulk.frac_coords[i]) - np.asarray(bulk.frac_coords[j])
            delta -= np.round(delta)
            cart_delta = bulk.lattice.get_cartesian_coords(delta)
            return float(np.linalg.norm(cart_delta))

        candidates = []
        for i in range(n):
            el_i = str(bulk[i].specie)
            if _is_metal(el_i):
                continue
            ri = _cov_r(el_i)
            for j in range(i + 1, n):
                dist = _pbc_distance(i, j)
                if dist > cov_cutoff:
                    continue
                el_j = str(bulk[j].specie)
                if _is_metal(el_j):
                    continue
                if not _can_consider_bond(el_i, el_j):
                    continue
                rj = _cov_r(el_j)
                if dist <= ri + rj + _bond_tolerance(el_i, el_j):
                    candidates.append((
                        _bond_priority(el_i, el_j),
                        dist,
                        i,
                        j,
                    ))

        pruned_candidates = self._prune_framework_halogen_candidates(
            bulk,
            candidates,
            _cov_r,
            _bond_tolerance,
        )

        adj: List[Set[int]] = [set() for _ in range(n)]
        bond_counts = [0] * n
        for _, _, i, j in sorted(pruned_candidates):
            el_i = str(bulk[i].specie)
            el_j = str(bulk[j].specie)
            if bond_counts[i] >= _max_bonds(el_i):
                continue
            if bond_counts[j] >= _max_bonds(el_j):
                continue
            adj[i].add(j)
            adj[j].add(i)
            bond_counts[i] += 1
            bond_counts[j] += 1

        visited = [False] * n
        clusters: List[List[int]] = []
        for start in range(n):
            if visited[start] or not adj[start]:
                continue
            queue = [start]
            visited[start] = True
            cluster = [start]
            while queue:
                node = queue.pop()
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        cluster.append(nb)
                        queue.append(nb)
            clusters.append(cluster)

        molecules = []
        for cluster in clusters:
            if len(cluster) < self.min_size:
                continue
            coords = np.array([bulk[i].frac_coords for i in cluster])
            extent = []
            for dim in range(3):
                c = coords[:, dim]
                c_sorted = np.sort(c % 1.0)
                gaps = np.diff(np.concatenate([c_sorted, [c_sorted[0] + 1.0]]))
                max_gap = gaps.max()
                span = 1.0 - max_gap
                extent.append(span * bulk.lattice.abc[dim])
            max_extent = max(extent)
            cell_min = min(bulk.lattice.abc)
            if max_extent < cell_min * 0.8:
                molecules.append(cluster)

        return molecules

    def _prune_framework_halogen_candidates(
        self,
        bulk,
        candidates,
        cov_radius,
        bond_tolerance,
    ):
        """Remove halogen contacts that look inorganic or shared.

        A single terminal C-I/C-Br/etc. contact is molecular.  A halide with
        nearby metal coordination, or with several competing organic-heavy
        contacts, is more likely a framework/counter-ion contact and should
        not merge an organic fragment into the inorganic network.
        """
        if not candidates:
            return candidates

        symbols = [str(site.specie) for site in bulk]
        halogens = {
            i for i, sym in enumerate(symbols)
            if sym in self._HALOGENS
        }
        if not halogens:
            return candidates

        max_cov_r = max(cov_radius(sym) for sym in symbols)
        neighbor_cutoff = 2 * (max_cov_r + self.tol)
        framework_halogen: Set[int] = set()

        for hidx in halogens:
            hsym = symbols[hidx]
            organic_contacts = 0
            metal_contacts = 0
            for nn in bulk.get_neighbors(bulk[hidx], neighbor_cutoff):
                jsym = symbols[int(nn.index)]
                cutoff = cov_radius(hsym) + cov_radius(jsym) + bond_tolerance(hsym, jsym)
                if float(nn.nn_distance) > cutoff:
                    continue
                if jsym not in self._NONMETALS:
                    metal_contacts += 1
                elif jsym in self._ORGANIC_HEAVY:
                    organic_contacts += 1

            if metal_contacts > 0:
                framework_halogen.add(hidx)
            elif organic_contacts >= self.framework_halogen_min_neighbors:
                framework_halogen.add(hidx)

        if not framework_halogen:
            return candidates

        pruned = []
        for item in candidates:
            _, _, i, j = item
            if i in framework_halogen or j in framework_halogen:
                continue
            pruned.append(item)
        return pruned
