"""Extract summary information from a structure."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from mmkit.core.structure import Structure
from mmkit.core.tool import Observation


class StructureInfo(Observation):
    """Collect key information about a structure (cellpar, composition, density)."""

    def observe(self, structure: Structure, **kwargs) -> Dict[str, Any]:
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

    def print_summary(self, structure: Structure) -> None:
        """Pretty-print a summary to stdout."""
        info = self.observe(structure)
        lat = info["lattice"]
        print("=== Structure Info ===")
        print(f"  Lattice : a={lat['a']:.4f}  b={lat['b']:.4f}  c={lat['c']:.4f} A")
        print(f"            alpha={lat['alpha']:.2f}  beta={lat['beta']:.2f}  gamma={lat['gamma']:.2f}")
        print(f"  Volume  : {info['volume']:.4f} A^3")
        print(f"  Atoms   : {info['num_atoms']}")
        print(f"  Comp.   : {info['composition']}")
        print(f"  Density : {info['density_g_cm3']:.4f} g/cm3")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_info(args):
    """CLI handler: print structure info."""
    import json
    from mmkit.io import read_structure
    from mmkit.core.structure import Structure

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
