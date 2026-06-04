"""Validate / sanity-check a structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from mmkit.core.structure import Structure
from mmkit.core.tool import Observation


@dataclass
class CheckResult:
    """Result of a structure check."""

    passed: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"CheckResult({status}, warnings={len(self.warnings)}, errors={len(self.errors)})"


class StructureCheck(Observation):
    """Run sanity checks on a structure.

    Checks:
      1. Overlapping atoms (distance < ``min_dist`` Å, MIC)
      2. Fractional coords outside [0, 1) (warning only)
      3. Density outside ``density_bounds`` g/cm³
    """

    def __init__(
        self,
        min_dist: float = 0.5,
        density_bounds: Tuple[float, float] = (0.01, 30.0),
    ) -> None:
        self.min_dist = min_dist
        self.density_bounds = density_bounds

    def observe(self, structure: Structure, **kwargs) -> CheckResult:
        result = CheckResult(passed=True)
        self._check_overlaps(structure, result.errors)
        self._check_bounds(structure, result.warnings)
        self._check_density(structure, result.warnings)
        result.passed = not result.errors
        return result

    # ------------------------------------------------------------------
    def _check_overlaps(self, struct: Structure, errors: List[str]) -> None:
        atoms = struct.atoms
        n = len(atoms)
        if n < 2:
            return
        d = atoms.get_all_distances(mic=True)
        iu, ju = np.triu_indices(n, k=1)
        mask = d[iu, ju] < self.min_dist
        symbols = atoms.get_chemical_symbols()
        for i, j in zip(iu[mask], ju[mask]):
            errors.append(
                f"Overlap: atom {i} ({symbols[i]}) and atom {j} ({symbols[j]}) "
                f"are {d[i, j]:.4f} A apart."
            )

    def _check_bounds(self, struct: Structure, warnings: List[str]) -> None:
        eps = 1e-6
        frac = struct.positions
        out = np.any((frac < -eps) | (frac > 1.0 + eps), axis=1)
        if not out.any():
            return
        symbols = struct.symbols
        for i in np.nonzero(out)[0]:
            warnings.append(
                f"Atom {i} ({symbols[i]}) has fractional coords {frac[i]} "
                "outside [0, 1). Use structure.wrap_to_cell()."
            )

    def _check_density(self, struct: Structure, warnings: List[str]) -> None:
        d = struct.density
        lo, hi = self.density_bounds
        if not (lo <= d <= hi):
            warnings.append(
                f"Density {d:.4f} g/cm³ is outside expected range [{lo}, {hi}]."
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_check(args):
    """CLI handler: validate a structure."""
    import json
    import sys
    from mmkit.io import read_structure
    from mmkit.core.structure import Structure

    atoms = read_structure(args.input)
    structure = Structure.from_ase_atoms(atoms)

    checker = StructureCheck(
        min_dist=args.min_dist,
        density_bounds=tuple(args.density_bounds),
    )
    result = checker.observe(structure)

    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        status = "PASS" if result.passed else "FAIL"
        print(f"Check: {status}")
        if result.warnings or args.verbose:
            for w in result.warnings:
                print(f"  WARNING: {w}")
        for e in result.errors:
            print(f"  ERROR: {e}")

    if not result.passed:
        sys.exit(1)


def register_cli(subparsers) -> None:
    """Register check subcommands with the mmkit CLI."""
    p = subparsers.add_parser("check", help="Validate / sanity-check a structure")
    p.add_argument("input", help="Structure file (CIF, POSCAR, extxyz, ...)")
    p.add_argument("--min-dist", type=float, default=0.5,
                   help="Minimum allowed distance (A, default: 0.5)")
    p.add_argument("--density-bounds", type=float, nargs=2, default=[0.01, 30.0],
                   help="Allowed density range (default: 0.01 30.0)")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.add_argument("--verbose", "-v", action="store_true", help="Show all warnings")
    p.set_defaults(handler=_cmd_check)
