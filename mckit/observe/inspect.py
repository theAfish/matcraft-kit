"""Extract summary information from a structure.

The module is organized around *info sections* — small, self-contained
inspectors that each report one aspect of a structure.  ``StructureInfo``
aggregates all registered sections so adding a new category of information
only requires writing a new ``InfoSection`` subclass and registering it.

Built-in sections
-----------------
* **basic** — lattice, volume, composition, density
* **vacuum** — vacuum-layer detection along all lattice directions
* **slabs** — per-slab composition and element distribution (when vacuum is present)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mckit.core.structure import Structure
from mckit.core.tool import Observation


# ---------------------------------------------------------------------------
# Section base class
# ---------------------------------------------------------------------------

class InfoSection(ABC):
    """One category of structural information.

    Subclass this to add a new information category.  At minimum you must
    set ``key`` and implement :meth:`observe`.  Override :meth:`print_section`
    for custom formatting (the default prints the raw dict).
    """

    key: str = ""

    @abstractmethod
    def observe(self, structure: Structure) -> Dict[str, Any]:
        """Return a dict of information for *structure*."""

    def print_section(
        self,
        info: Dict[str, Any],
        structure: Structure,
        all_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Pretty-print this section.  *info* is the dict from :meth:`observe`."""
        print(f"  [{self.key}]")
        for k, v in info.items():
            print(f"    {k}: {v}")


# ---------------------------------------------------------------------------
# Built-in sections
# ---------------------------------------------------------------------------

class BasicInfo(InfoSection):
    """Lattice parameters, volume, composition, density."""

    key = "basic"

    def observe(self, structure: Structure) -> Dict[str, Any]:
        atoms = structure.atoms
        a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
        return {
            "lattice": {
                "a": float(a), "b": float(b), "c": float(c),
                "alpha": float(alpha), "beta": float(beta), "gamma": float(gamma),
            },
            "volume": float(atoms.cell.volume),
            "num_atoms": len(atoms),
            "composition": dict(Counter(atoms.get_chemical_symbols())),
            "density_g_cm3": structure.density,
            "total_mass_amu": float(atoms.get_masses().sum()),
        }

    def print_section(
        self,
        info: Dict[str, Any],
        structure: Structure,
        all_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        has_cell = info["volume"] > 1e-10
        print("[basic]")
        if has_cell:
            lat = info["lattice"]
            print(f"  Lattice : a={lat['a']:.4f}  b={lat['b']:.4f}  c={lat['c']:.4f} A")
            print(f"            alpha={lat['alpha']:.2f}  beta={lat['beta']:.2f}  gamma={lat['gamma']:.2f}")
        else:
            print("  Lattice : (no cell -- isolated atoms)")
        print(f"  Atoms   : {info['num_atoms']}")
        print(f"  Comp.   : {info['composition']}")
        if has_cell:
            print(f"  Density : {info['density_g_cm3']:.4f} g/cm3")

        # Show classification if available (only meaningful with a cell)
        if has_cell and all_info and "vacuum" in all_info:
            vac_info = all_info["vacuum"]
            classification = vac_info.get("classification", "")
            threshold = vac_info.get("threshold", 3.0)
            print(f"  Class.  : {classification} (vacuum threshold {threshold:.1f} A)")


class MoleculeInfo(InfoSection):
    """Molecular fragments detected by non-metal connectivity."""

    key = "molecules"

    def __init__(self, tol: float = 0.45, min_size: int = 2) -> None:
        self.tol = tol
        self.min_size = min_size

    @staticmethod
    def _composition_key(composition: Dict[str, int]) -> Tuple[Tuple[str, int], ...]:
        return tuple(sorted(composition.items()))

    @staticmethod
    def _format_formula(composition: Dict[str, int]) -> str:
        parts = []
        for elem in sorted(composition):
            count = composition[elem]
            parts.append(elem if count == 1 else f"{elem}{count}")
        return "".join(parts)

    def observe(self, structure: Structure) -> Dict[str, Any]:
        try:
            from mckit.operate.molecule_utils import MoleculeDetector, pbc_center
        except ImportError as exc:
            return {
                "available": False,
                "reason": str(exc),
                "num_molecules": 0,
                "molecules": [],
                "types": [],
            }

        try:
            bulk = structure.to_pymatgen()
            detected = MoleculeDetector(tol=self.tol, min_size=self.min_size).detect(bulk)
        except ImportError as exc:
            return {
                "available": False,
                "reason": str(exc),
                "num_molecules": 0,
                "molecules": [],
                "types": [],
            }

        symbols = structure.symbols

        molecules: List[Dict[str, Any]] = []
        type_counts: Counter = Counter()
        for mol in detected:
            indices = sorted(int(i) for i in mol)
            composition = dict(Counter(symbols[i] for i in indices))
            center = pbc_center(bulk, indices)
            key = self._composition_key(composition)
            type_counts[key] += 1
            molecules.append({
                "indices": indices,
                "num_atoms": len(indices),
                "composition": composition,
                "formula": self._format_formula(composition),
                "center_frac": [round(float(x), 4) for x in center],
            })

        types = []
        for comp_key, count in type_counts.items():
            composition = dict(comp_key)
            types.append({
                "formula": self._format_formula(composition),
                "composition": composition,
                "count": int(count),
            })

        types.sort(key=lambda item: (item["formula"], item["count"]))
        return {
            "available": True,
            "num_molecules": len(molecules),
            "num_molecular_atoms": sum(mol["num_atoms"] for mol in molecules),
            "types": types,
            "molecules": molecules,
            "tol": self.tol,
            "min_size": self.min_size,
        }

    def print_section(
        self,
        info: Dict[str, Any],
        structure: Structure,
        all_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not info.get("available", True):
            print("[Molecules]")
            print(f"  Detection unavailable: {info.get('reason', 'unknown error')}")
            return

        if info.get("num_molecules", 0) == 0:
            return

        print("[Molecules detected]")
        print(f"  Total : {info['num_molecules']} molecule(s), "
              f"{info['num_molecular_atoms']} atoms")
        print("  Types :")
        for item in info.get("types", []):
            print(f"    {item['formula']}: {item['count']} molecule(s), "
                  f"{item['composition']}")
        print("  Centers:")
        for i, mol in enumerate(info.get("molecules", [])):
            print(f"    {mol['formula']}: {mol['num_atoms']} atoms, "
                  f"center(frac)={mol['center_frac']}")


# ---------------------------------------------------------------------------
# Shared helpers for multi-direction vacuum/slab analysis
# ---------------------------------------------------------------------------

_DIRECTION_LABELS = ("a", "b", "c")


def _direction_normals(
    cell_vectors: np.ndarray,
) -> List[Tuple[str, Optional[np.ndarray], float]]:
    """Compute perpendicular normal and cell height for each lattice direction.

    For direction *d*, the normal is perpendicular to the plane formed by the
    other two lattice vectors.  This is the physically meaningful direction
    for slab/vacuum detection, and works for any cell shape (orthogonal or
    triclinic).

    Parameters
    ----------
    cell_vectors
        3×3 array whose rows are the lattice vectors **a**, **b**, **c**.

    Returns
    -------
    list of (label, normal, height)
        *label* is ``'a'``, ``'b'``, or ``'c'``.
        *normal* is the unit normal vector (or ``None`` for degenerate cells).
        *height* is the perpendicular cell extent in Å along that normal.
    """
    vectors = np.asarray(cell_vectors, dtype=np.float64)
    result: List[Tuple[str, Optional[np.ndarray], float]] = []

    for d in range(3):
        others = [vectors[i] for i in range(3) if i != d]
        cross = np.cross(others[0], others[1])
        cross_len = float(np.linalg.norm(cross))
        if cross_len < 1e-10:
            result.append((_DIRECTION_LABELS[d], None, 0.0))
        else:
            normal = cross / cross_len
            height = float(abs(np.dot(vectors[d], normal)))
            result.append((_DIRECTION_LABELS[d], normal, height))

    return result


def _detect_vacuum_1d(
    cart_positions: np.ndarray,
    normal: np.ndarray,
    cell_height: float,
    threshold: float,
) -> Tuple[List[List[int]], np.ndarray, np.ndarray, np.ndarray]:
    """Detect vacuum gaps and group atoms into slab regions along one direction.

    Parameters
    ----------
    cart_positions
        (N, 3) array of Cartesian atomic positions.
    normal
        Unit vector perpendicular to the slab plane for this direction.
    cell_height
        Cell extent along *normal* in Å.
    threshold
        Minimum gap size (Å) to count as vacuum.

    Returns
    -------
    slabs : list of list of int
        Each inner list holds original atom indices belonging to one slab.
    all_gaps_A : ndarray
        All inter-atomic gaps in Å, including the periodic wrap-around gap.
    vacuum_mask : ndarray of bool
        Which gaps exceed *threshold*.
    sorted_indices : ndarray of int
        Original atom indices sorted by projected coordinate.
    """
    n = len(cart_positions)
    if n == 0 or cell_height < 1e-10:
        return [], np.array([]), np.array([], dtype=bool), np.array([], dtype=int)

    # Project onto normal and wrap to [0, cell_height)
    proj = (cart_positions @ normal) % cell_height

    # Sort by projection
    sorted_indices = np.argsort(proj)
    sorted_proj = proj[sorted_indices]

    # Compute gaps (including periodic wrap)
    gaps = np.diff(sorted_proj)
    periodic_gap = cell_height - sorted_proj[-1] + sorted_proj[0]
    all_gaps = np.append(gaps, periodic_gap)

    vacuum_mask = all_gaps > threshold

    if not vacuum_mask.any():
        # No vacuum → one slab containing all atoms
        return [sorted_indices.tolist()], all_gaps, vacuum_mask, sorted_indices

    # Group atoms into slabs between consecutive vacuum gaps.
    # Uses modular (circular) indexing to handle periodic wrapping.
    vacuum_positions = np.where(vacuum_mask)[0]
    slabs: List[List[int]] = []
    for k in range(len(vacuum_positions)):
        # Start after this vacuum gap, collect until next vacuum gap
        start = (vacuum_positions[k] + 1) % n
        end = vacuum_positions[(k + 1) % len(vacuum_positions)]

        slab_idx: List[int] = []
        idx = start
        while True:
            slab_idx.append(int(sorted_indices[idx]))
            if idx == end:
                break
            idx = (idx + 1) % n

        if slab_idx:
            slabs.append(slab_idx)

    return slabs, all_gaps, vacuum_mask, sorted_indices


def _detect_layers(
    atom_indices: List[int],
    cart_positions: np.ndarray,
    normal: np.ndarray,
    cell_height: float,
    tolerance: float = 0.5,
) -> List[Tuple[List[int], np.ndarray]]:
    """Cluster atoms within a slab into layers by projected coordinate.

    Atoms whose projections differ by less than *tolerance* Å are grouped
    into the same layer.  Handles periodic wrapping by finding the largest
    circular gap (where the slab is *not*) and linearizing from there.

    Parameters
    ----------
    atom_indices
        Original atom indices belonging to one slab.
    cart_positions
        (N, 3) array of Cartesian positions for the full structure.
    normal
        Unit normal for the slab direction.
    cell_height
        Cell extent along *normal* in Å.
    tolerance
        Maximum projection gap (Å) for two atoms to be in the same layer.

    Returns
    -------
    list of (atom_indices, linearized_projections)
        Each entry holds original atom indices and their unwrapped projected
        coordinates (Å) for one layer, ordered from low to high.
    """
    if not atom_indices or cell_height < 1e-10:
        return []

    idx = np.asarray(atom_indices, dtype=int)
    proj = (cart_positions[idx] @ normal) % cell_height
    # Fix floating-point edge: atoms at exactly 0 or cell_height may get
    # mod_proj ≈ cell_height instead of 0
    proj[proj >= cell_height - 1e-4] = 0.0
    n = len(idx)

    if n <= 1:
        return [(atom_indices[:], proj)]

    # Sort by projection
    order = np.argsort(proj)
    sorted_proj = proj[order]
    sorted_idx = idx[order]

    # Compute circular gaps (including periodic wrap-around)
    gaps = np.diff(sorted_proj)
    wrap_gap = cell_height - sorted_proj[-1] + sorted_proj[0]
    all_gaps = np.append(gaps, wrap_gap)

    # The largest gap is "where the slab is not" — cut there to linearize
    cut = int(np.argmax(all_gaps))

    if cut == n - 1:
        # Largest gap is the wrap-around → already linear, no unwrapping needed
        lin_proj = sorted_proj.copy()
        lin_idx = sorted_idx.copy()
    else:
        # Rotate so the cut is at the start, unwrap the wrapped segment
        lin_order = np.roll(order, -(cut + 1))
        lin_idx = idx[lin_order]
        lin_proj = proj[lin_order].copy()
        # Add cell_height to atoms that wrapped around
        lin_proj[lin_proj < lin_proj[0]] += cell_height

    # Group consecutive atoms into layers
    layers: List[Tuple[List[int], np.ndarray]] = []
    current_indices: List[int] = [int(lin_idx[0])]
    current_projs: List[float] = [float(lin_proj[0])]

    for i in range(1, n):
        if lin_proj[i] - lin_proj[i - 1] >= tolerance:
            layers.append((current_indices, np.array(current_projs) % cell_height))
            current_indices = [int(lin_idx[i])]
            current_projs = [float(lin_proj[i])]
        else:
            current_indices.append(int(lin_idx[i]))
            current_projs.append(float(lin_proj[i]))

    layers.append((current_indices, np.array(current_projs) % cell_height))
    return layers


def _molecule_aware_region_compositions(
    regions: List[List[int]],
    symbols: List[str],
    molecule_info: Optional[Dict[str, Any]],
) -> List[Dict[str, int]]:
    """Collapse detected molecular atoms into formula units for print output."""
    compositions = [Counter(symbols[i] for i in region) for region in regions]
    if not molecule_info or not molecule_info.get("available", True):
        return [dict(comp) for comp in compositions]

    molecules = molecule_info.get("molecules", [])
    if not molecules:
        return [dict(comp) for comp in compositions]

    region_sets = [set(region) for region in regions]
    for mol in molecules:
        mol_indices = set(int(i) for i in mol.get("indices", []))
        if not mol_indices:
            continue

        overlaps = [len(region & mol_indices) for region in region_sets]
        best_overlap = max(overlaps) if overlaps else 0
        if best_overlap == 0:
            continue

        best_region = int(np.argmax(overlaps))
        for ri, region in enumerate(region_sets):
            for atom_idx in region & mol_indices:
                sym = symbols[atom_idx]
                compositions[ri][sym] -= 1
                if compositions[ri][sym] <= 0:
                    del compositions[ri][sym]

        formula = mol.get("formula")
        if formula:
            compositions[best_region][f"[{formula}]"] += 1

    return [dict(comp) for comp in compositions]


def _classify(directions: Dict[str, Dict[str, Any]]) -> str:
    """Heuristic classification based on vacuum across all directions."""
    vacuum_dirs = [d for d in _DIRECTION_LABELS
                   if directions.get(d, {}).get("has_vacuum", False)]
    n = len(vacuum_dirs)

    if n == 0:
        return "bulk"

    dir_str = ", ".join(vacuum_dirs)

    if n == 1:
        d = vacuum_dirs[0]
        n_layers = directions[d]["num_vacuum_layers"]
        if n_layers == 1:
            return f"surface slab (vacuum along {d})"
        if n_layers == 2:
            return f"interface / thin film (vacuum along {d})"
        return f"multi-gap slab ({n_layers} vacuum layers along {d})"

    if n == 2:
        return f"nanowire / 1D (vacuum along {dir_str})"

    return f"nanocluster / 0D (vacuum along {dir_str})"


# ---------------------------------------------------------------------------
# Built-in sections
# ---------------------------------------------------------------------------

class VacuumInfo(InfoSection):
    """Detect vacuum layers along lattice directions.

    For each lattice direction (a, b, c), projects atomic positions onto the
    perpendicular normal and identifies gaps exceeding ``threshold`` Å,
    including the periodic wrap-around gap.

    Parameters
    ----------
    threshold
        Minimum gap size (Å) to count as a vacuum layer.
    directions
        Which lattice directions to check.  ``None`` (default) checks all
        three (a, b, c).  Pass e.g. ``['c']`` to restrict.
    """

    key = "vacuum"

    def __init__(
        self,
        threshold: float = 3.0,
        directions: Optional[List[str]] = None,
    ) -> None:
        self.threshold = threshold
        self._check_dirs = set(directions) if directions else set(_DIRECTION_LABELS)

    def observe(self, structure: Structure) -> Dict[str, Any]:
        cell = np.asarray(structure.atoms.cell.array)
        cart_pos = structure.cart_positions
        dir_info = _direction_normals(cell)

        directions: Dict[str, Dict[str, Any]] = {}
        vacuum_dirs: List[str] = []

        for label, normal, height in dir_info:
            if label not in self._check_dirs or normal is None:
                directions[label] = {
                    "has_vacuum": False,
                    "num_vacuum_layers": 0,
                    "vacuum_sizes_A": [],
                    "total_vacuum_A": 0.0,
                    "slab_thickness_A": round(height, 4),
                    "cell_height_A": round(height, 4),
                }
                continue

            _slabs, gaps, vac_mask, _ = _detect_vacuum_1d(
                cart_pos, normal, height, self.threshold
            )

            vac_sizes = gaps[vac_mask]
            num_vac = int(vac_mask.sum())
            total_vac = float(vac_sizes.sum()) if num_vac else 0.0

            if num_vac > 0:
                vacuum_dirs.append(label)

            directions[label] = {
                "has_vacuum": num_vac > 0,
                "num_vacuum_layers": num_vac,
                "vacuum_sizes_A": [round(float(s), 4) for s in vac_sizes],
                "total_vacuum_A": round(total_vac, 4),
                "slab_thickness_A": round(height - total_vac, 4),
                "cell_height_A": round(height, 4),
            }

        classification = _classify(directions)

        return {
            "directions": directions,
            "num_vacuum_directions": len(vacuum_dirs),
            "vacuum_directions": vacuum_dirs,
            "classification": classification,
            "threshold": self.threshold,
        }

    def print_section(
        self,
        info: Dict[str, Any],
        structure: Structure,
        all_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Skip if no vacuum detected (classification shown in BasicInfo)
        if info["num_vacuum_directions"] == 0:
            return

        print("[Vacuum layer(s) detected]")

        for d in info["vacuum_directions"]:
            dinfo = info["directions"][d]
            print(f"  Direction {d}:")
            print(f"    Slab    : {dinfo['slab_thickness_A']:.2f} A thick")
            print(f"    Cell    : {dinfo['cell_height_A']:.2f} A")
            n = dinfo["num_vacuum_layers"]
            print(f"    Vacuum  : {n} layer{'s' if n != 1 else ''}, "
                  f"total {dinfo['total_vacuum_A']:.2f} A")
            # print layer sizes if n layers > 1
            if n > 1:
                for i, size in enumerate(dinfo["vacuum_sizes_A"], 1):
                    print(f"      layer {i}: {size:.2f} A")


class SlabCompositionInfo(InfoSection):
    """Per-slab composition and element distribution.

    Only reports when vacuum layers are detected.  For each vacuum direction,
    splits atoms into slab regions and computes composition, atom count, and
    extent along that direction.  When ``show_layers`` is enabled, atoms
    within each slab are further clustered into layers by their projected
    coordinate.

    Parameters
    ----------
    threshold
        Minimum gap size (Å) to count as vacuum (should match VacuumInfo).
    directions
        Which lattice directions to check.  ``None`` (default) checks all.
    layer_tolerance
        Maximum projection gap (Å) for atoms to belong to the same layer
        within a slab.  Default 0.5 Å.
    show_layers
        Whether to compute and display per-layer composition within each
        slab.  Default ``True``.
    """

    key = "slabs"

    def __init__(
        self,
        threshold: float = 3.0,
        directions: Optional[List[str]] = None,
        layer_tolerance: float = 2.0,
        show_layers: bool = True,
        show_bulk_layers: bool = False,
    ) -> None:
        self.threshold = threshold
        self.layer_tolerance = layer_tolerance
        self.show_layers = show_layers
        self.show_bulk_layers = show_bulk_layers
        self._check_dirs = set(directions) if directions else set(_DIRECTION_LABELS)

    def observe(self, structure: Structure) -> Dict[str, Any]:
        cell = np.asarray(structure.atoms.cell.array)
        cart_pos = structure.cart_positions
        symbols = structure.symbols
        dir_info = _direction_normals(cell)

        result: Dict[str, Any] = {"directions": {}, "has_vacuum": False}

        if len(cart_pos) == 0:
            return result

        for label, normal, height in dir_info:
            if label not in self._check_dirs or normal is None:
                continue

            slabs, _, vac_mask, _ = _detect_vacuum_1d(
                cart_pos, normal, height, self.threshold
            )
            has_direction_vacuum = bool(vac_mask.any())

            if has_direction_vacuum:
                result["has_vacuum"] = True

            # Projected coordinates for extent calculation
            proj = (cart_pos @ normal) % height

            slab_info: List[Dict[str, Any]] = []
            for idx_list in slabs:
                slab_symbols = [symbols[i] for i in idx_list]
                slab_proj = proj[idx_list]
                entry: Dict[str, Any] = {
                    "num_atoms": len(idx_list),
                    "atom_indices": [int(i) for i in idx_list],
                    "composition": dict(Counter(slab_symbols)),
                    "extent_A": (
                        round(float(slab_proj.min()), 4),
                        round(float(slab_proj.max()), 4),
                    ),
                }

                # Per-layer decomposition within this slab
                if self.show_layers:
                    layers = _detect_layers(
                        idx_list, cart_pos, normal, height, self.layer_tolerance
                    )
                    layer_list: List[Dict[str, Any]] = []
                    for layer_idx, layer_lin_proj in layers:
                        layer_symbols = [symbols[i] for i in layer_idx]
                        ext_min = round(float(layer_lin_proj.min()), 4)
                        ext_max = round(float(layer_lin_proj.max()), 4)
                        layer_list.append({
                            "num_atoms": len(layer_idx),
                            "atom_indices": [int(i) for i in layer_idx],
                            "composition": dict(Counter(layer_symbols)),
                            "extent_A": (ext_min, ext_max),
                            "center_A": round((ext_min + ext_max) / 2, 4),
                        })
                    entry["layers"] = layer_list

                slab_info.append(entry)

            # Element distribution across slabs
            all_elements = sorted(set(symbols))
            element_dist: Dict[str, List[Dict[str, Any]]] = {}
            for elem in all_elements:
                elem_dist: List[Dict[str, Any]] = []
                for si, slab in enumerate(slab_info):
                    if elem in slab["composition"]:
                        elem_dist.append({
                            "slab": si,
                            "count": slab["composition"][elem],
                        })
                element_dist[elem] = elem_dist

            result["directions"][label] = {
                "slabs": slab_info,
                "element_distribution": element_dist,
                "cell_height_A": round(height, 4),
                "has_vacuum": has_direction_vacuum,
            }

        return result

    def print_section(
        self,
        info: Dict[str, Any],
        structure: Structure,
        all_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not info.get("directions"):
            return

        molecule_info = all_info.get("molecules") if all_info else None
        symbols = structure.symbols

        for label, dinfo in info["directions"].items():
            if not dinfo.get("has_vacuum", False) and not self.show_bulk_layers:
                continue

            slabs = dinfo["slabs"]
            n_slabs = len(slabs)
            slab_regions = [slab.get("atom_indices", []) for slab in slabs]
            slab_compositions = _molecule_aware_region_compositions(
                slab_regions, symbols, molecule_info
            )

            # Omit directions with no interesting structure — a single
            # region with ≤1 layer adds nothing beyond [basic].
            layers = slabs[0].get("layers", []) if slabs else []
            if n_slabs == 1 and len(layers) <= 1:
                continue

            # Header: different phrasing for vacuum vs non-vacuum directions
            if info.get("has_vacuum"):
                header = f"Slab composition"
            else:
                header = f"Layer composition"

            print(f"[{header} — direction {label} "
                  f"({n_slabs} region{'s' if n_slabs != 1 else ''})]")
            for i, slab in enumerate(slabs):
                ext_min, ext_max = slab["extent_A"]
                print(f"  Slab {i}: {slab['num_atoms']} atoms, "
                      f"extent=[{ext_min:.2f}, {ext_max:.2f}] A, "
                      f"{ext_max - ext_min:.2f} A thick")
                print(f"    Comp. : {slab_compositions[i]}")

                # Per-layer breakdown
                layers = slab.get("layers", [])
                if self.show_layers and len(layers) > 1:
                    layer_regions = [layer.get("atom_indices", []) for layer in layers]
                    layer_compositions = _molecule_aware_region_compositions(
                        layer_regions, symbols, molecule_info
                    )
                    print(f"    Layers ({len(layers)}):")
                    for li, layer in enumerate(layers):
                        lmin, lmax = layer["extent_A"]
                        print(f"      Layer {li}: {layer['num_atoms']:>3d} atoms, "
                              f"z=[{lmin:.2f}, {lmax:.2f}] A, "
                              f"center={layer['center_A']:.2f} A  "
                              f"{layer_compositions[li]}")

            # Element distribution:
            # elem_dist = dinfo["element_distribution"]
            # print("  Element distribution:")
            # for elem in sorted(elem_dist.keys()):
            #     locations = elem_dist[elem]
            #     if len(locations) == 1:
            #         loc = locations[0]
            #         print(f"    {elem:2s}: only in slab {loc['slab']} "
            #               f"({loc['count']} atoms)")
            #     else:
            #         parts = [f"slab {loc['slab']}: {loc['count']}" for loc in locations]
            #         print(f"    {elem:2s}: {'   '.join(parts)}")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

# Default sections used when none are specified.
_DEFAULT_SECTIONS: List[InfoSection] = [
    BasicInfo(),
    MoleculeInfo(),
    VacuumInfo(),
    SlabCompositionInfo(),
]


class StructureInspect(Observation):
    """Collect key information about a structure.

    Runs every registered :class:`InfoSection` and merges the results under
    their ``key``.

    Parameters
    ----------
    sections
        Sections to run.  Defaults to :class:`BasicInfo`
        + :class:`MoleculeInfo` + :class:`VacuumInfo`
        + :class:`SlabCompositionInfo`.
    """

    def __init__(self, sections: Optional[List[InfoSection]] = None) -> None:
        self.sections = list(sections) if sections is not None else list(_DEFAULT_SECTIONS)

    def observe(self, structure: Structure, **kwargs) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for section in self.sections:
            result[section.key] = section.observe(structure)
        return result

    def print_summary(self, structure: Structure) -> None:
        """Pretty-print all sections to stdout."""
        info = self.observe(structure)
        # print("=== Structure Info ===")
        for section in self.sections:
            section_data = info.get(section.key, {})
            section.print_section(section_data, structure, all_info=info)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_info(args):
    """CLI handler: print structure info."""
    import json
    from mckit.io import read_structure

    atoms = read_structure(args.input)
    structure = Structure.from_ase_atoms(atoms)

    if args.json:
        info = StructureInspect().observe(structure)
        print(json.dumps(info, indent=2, default=str))
    else:
        StructureInspect().print_summary(structure)


def register_cli(subparsers) -> None:
    """Register inspect subcommands with the mmkit CLI."""
    p = subparsers.add_parser("inspect", help="Print structure information")
    p.add_argument("input", help="Structure file (CIF, POSCAR, extxyz, ...)")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.set_defaults(handler=_cmd_info)
