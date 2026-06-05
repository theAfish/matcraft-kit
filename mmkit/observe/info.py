"""Extract summary information from a structure.

The module is organized around *info sections* — small, self-contained
inspectors that each report one aspect of a structure.  ``StructureInfo``
aggregates all registered sections so adding a new category of information
only requires writing a new ``InfoSection`` subclass and registering it.

Built-in sections
-----------------
* **basic** — lattice, volume, composition, density
* **vacuum** — vacuum-layer detection along the *c*-direction
* **slabs** — per-slab composition and element distribution (when vacuum is present)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from mmkit.core.structure import Structure
from mmkit.core.tool import Observation


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

    def print_section(self, info: Dict[str, Any], structure: Structure) -> None:
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

    def print_section(self, info: Dict[str, Any], structure: Structure) -> None:
        lat = info["lattice"]
        print("[basic]")
        print(f"  Lattice : a={lat['a']:.4f}  b={lat['b']:.4f}  c={lat['c']:.4f} A")
        print(f"            alpha={lat['alpha']:.2f}  beta={lat['beta']:.2f}  gamma={lat['gamma']:.2f}")
        # print(f"  Volume  : {info['volume']:.4f} A^3")
        print(f"  Atoms   : {info['num_atoms']}")
        print(f"  Comp.   : {info['composition']}")
        print(f"  Density : {info['density_g_cm3']:.4f} g/cm3")


# ---------------------------------------------------------------------------
# Shared helpers for vacuum/slab analysis
# ---------------------------------------------------------------------------

def _find_slab_groups(
    structure: Structure,
    threshold: float,
) -> tuple[List[List[int]], np.ndarray, np.ndarray, float]:
    """Group atoms into slab regions separated by vacuum gaps along *c*.

    Parameters
    ----------
    structure
        The structure to analyse.
    threshold
        Minimum gap size (Å) to count as vacuum.

    Returns
    -------
    slabs : list of list of int
        Each inner list holds original atom indices belonging to one slab,
        ordered from bottom to top (by fractional *z*).
    sorted_indices : ndarray
        Original atom indices sorted by fractional *z*.
    all_gaps_A : ndarray
        All inter-atomic gaps in Å, including the periodic wrap-around gap.
    c_length : float
        Cell height along *c* in Å.
    """
    frac_z = structure.positions[:, 2]
    c_length = float(structure.lattice.c)
    n_atoms = len(frac_z)

    if n_atoms == 0:
        return [], np.array([], dtype=int), np.array([]), c_length

    sorted_indices = np.argsort(frac_z)
    sorted_z = frac_z[sorted_indices]

    gaps_frac = np.diff(sorted_z)
    periodic_gap_frac = 1.0 - sorted_z[-1] + sorted_z[0]
    all_gaps_frac = np.append(gaps_frac, periodic_gap_frac)
    all_gaps_A = all_gaps_frac * c_length

    vacuum_mask = all_gaps_A > threshold

    if not vacuum_mask.any():
        # No vacuum → one slab containing all atoms
        return [sorted_indices.tolist()], sorted_indices, all_gaps_A, c_length

    vacuum_positions = np.where(vacuum_mask)[0]

    # Build slabs by collecting atoms between consecutive vacuum gaps.
    # Use modular indexing to handle periodic wrapping correctly.
    slabs: List[List[int]] = []
    for k in range(len(vacuum_positions)):
        # Start after this vacuum gap, collect until next vacuum gap
        start = (vacuum_positions[k] + 1) % n_atoms
        end = vacuum_positions[(k + 1) % len(vacuum_positions)]

        slab_idx = []
        idx = start
        while idx != end:
            slab_idx.append(int(sorted_indices[idx]))
            idx = (idx + 1) % n_atoms

        if slab_idx:
            slabs.append(slab_idx)

    return slabs, sorted_indices, all_gaps_A, c_length


# ---------------------------------------------------------------------------
# Built-in sections
# ---------------------------------------------------------------------------

class VacuumInfo(InfoSection):
    """Detect vacuum layers along the *c*-direction.

    A *vacuum layer* is any gap between consecutive atoms (projected onto
    the *c*-vector) that exceeds ``threshold`` Å, including the periodic
    gap that wraps from the topmost atom back to the bottommost.

    Parameters
    ----------
    threshold
        Minimum gap size (Å) to count as a vacuum layer.
    """

    key = "vacuum"

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold

    def observe(self, structure: Structure) -> Dict[str, Any]:
        frac_z = structure.positions[:, 2]
        c_length = structure.lattice.c

        if len(frac_z) == 0:
            return {
                "has_vacuum": False,
                "num_vacuum_layers": 0,
                "vacuum_sizes_A": [],
                "total_vacuum_A": 0.0,
                "slab_thickness_A": 0.0,
                "cell_height_A": float(c_length),
                "classification": "empty",
            }

        _slabs, _, all_gaps_A, c_length = _find_slab_groups(
            structure, self.threshold
        )

        # Identify vacuum layers
        vacuum_mask = all_gaps_A > self.threshold
        vacuum_sizes = all_gaps_A[vacuum_mask]
        num_vacuum = int(vacuum_mask.sum())
        total_vacuum = float(vacuum_sizes.sum()) if num_vacuum else 0.0

        # Slab thickness = cell height minus all vacuum gaps.
        slab_thickness = float(c_length) - total_vacuum

        # Heuristic classification
        classification = _classify(num_vacuum)

        return {
            "has_vacuum": num_vacuum > 0,
            "num_vacuum_layers": num_vacuum,
            "vacuum_sizes_A": [round(float(s), 4) for s in vacuum_sizes],
            "total_vacuum_A": round(total_vacuum, 4),
            "slab_thickness_A": round(slab_thickness, 4),
            "cell_height_A": round(float(c_length), 4),
            "classification": classification,
        }

    def print_section(self, info: Dict[str, Any], structure: Structure) -> None:
        cls = info["classification"]
        print("[Vacuum layer(s) detected]")
        print(f"  Type    : {cls}")
        if not info["has_vacuum"]:
            print(f"  Vacuum  : none detected (threshold {self.threshold:.1f} A)")
            return

        # print(f"  Slab    : {info['slab_thickness_A']:.2f} A thick")
        # print(f"  Cell c  : {info['cell_height_A']:.2f} A")
        n = info["num_vacuum_layers"]
        print(f"  Vacuum  : {n} layer{'s' if n != 1 else ''}, "
              f"total {info['total_vacuum_A']:.2f} A")
        for i, size in enumerate(info["vacuum_sizes_A"], 1):
            print(f"    layer {i}: {size:.2f} A")


class SlabCompositionInfo(InfoSection):
    """Per-slab composition and element distribution.

    Only reports when vacuum layers are detected.  For each slab region
    (separated by vacuum gaps), computes composition, atom count, and
    *z*-range.  Also provides statistical summary of element distribution.

    Parameters
    ----------
    threshold
        Minimum gap size (Å) to count as vacuum (should match VacuumInfo).
    """

    key = "slabs"

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold

    def observe(self, structure: Structure) -> Dict[str, Any]:
        frac_z = structure.positions[:, 2]
        c_length = float(structure.lattice.c)
        symbols = structure.symbols

        if len(frac_z) == 0:
            return {"has_vacuum": False, "slabs": [], "element_distribution": {}}

        slabs, _, all_gaps_A, c_length = _find_slab_groups(
            structure, self.threshold
        )

        has_vacuum = bool((all_gaps_A > self.threshold).any())

        slab_info = []
        for idx_list in slabs:
            slab_symbols = [symbols[i] for i in idx_list]
            slab_frac_z = frac_z[idx_list]
            slab_cart_z = slab_frac_z * c_length
            slab_info.append({
                "num_atoms": len(idx_list),
                "composition": dict(Counter(slab_symbols)),
                "z_range_A": (round(float(slab_cart_z.min()), 4),
                              round(float(slab_cart_z.max()), 4)),
            })

        # Element distribution: for each element, list (slab_index, count)
        element_dist: Dict[str, List[dict]] = {}
        all_elements = sorted(set(symbols))
        for elem in all_elements:
            elem_dist = []
            for si, slab in enumerate(slab_info):
                if elem in slab["composition"]:
                    elem_dist.append({
                        "slab": si,
                        "count": slab["composition"][elem],
                    })
            element_dist[elem] = elem_dist

        return {
            "has_vacuum": has_vacuum,
            "slabs": slab_info,
            "element_distribution": element_dist,
        }

    def print_section(self, info: Dict[str, Any], structure: Structure) -> None:
        if not info["has_vacuum"]:
            # Don't print anything for bulk structures
            return

        slabs = info["slabs"]
        print(f"[Slab composition ({len(slabs)} region{'s' if len(slabs) != 1 else ''})]")
        for i, slab in enumerate(slabs):
            z_min, z_max = slab["z_range_A"]
            print(f"  Slab {i}: {slab['num_atoms']} atoms, "
                  f"z=[{z_min:.2f}, {z_max:.2f}] A, {z_max - z_min:.2f} A thick")
            print(f"    Comp. : {slab['composition']}")

        # Statistical summary: element distribution
        # print("  Element distribution:")
        # elem_dist = info["element_distribution"]
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
# Helpers
# ---------------------------------------------------------------------------

def _classify(num_vacuum: int) -> str:
    """Heuristic structure classification based on vacuum layer count."""
    if num_vacuum == 0:
        return "bulk"
    if num_vacuum == 1:
        return "surface slab"
    if num_vacuum == 2:
        return "interface / thin film"
    return f"multi-gap ({num_vacuum} vacuum layers)"


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

# Default sections used when none are specified.
_DEFAULT_SECTIONS: List[InfoSection] = [
    BasicInfo(),
    VacuumInfo(),
    SlabCompositionInfo(),
]


class StructureInfo(Observation):
    """Collect key information about a structure.

    Runs every registered :class:`InfoSection` and merges the results under
    their ``key``.

    Parameters
    ----------
    sections
        Sections to run.  Defaults to :class:`BasicInfo` + :class:`VacuumInfo`.
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
            section.print_section(section_data, structure)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_info(args):
    """CLI handler: print structure info."""
    import json
    from mmkit.io import read_structure

    atoms = read_structure(args.input)
    structure = Structure.from_ase_atoms(atoms)

    if args.json:
        info = StructureInfo().observe(structure)
        print(json.dumps(info, indent=2, default=str))
    else:
        StructureInfo().print_summary(structure)


def register_cli(subparsers) -> None:
    """Register info subcommands with the mmkit CLI."""
    p = subparsers.add_parser("info", help="Print structure information")
    p.add_argument("input", help="Structure file (CIF, POSCAR, extxyz, ...)")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.set_defaults(handler=_cmd_info)
