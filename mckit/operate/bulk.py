"""Build standard bulk crystals via ``ase.build.bulk``.

ASE's ``bulk`` already handles every structure type we care about
(``fcc``, ``bcc``, ``hcp``, ``diamond``, ``zincblende``, ``rocksalt``,
``cesiumchloride``, ``fluorite``, ``wurtzite``, ...) so this class is just
a thin matmod-flavored wrapper.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from ase import Atoms
from ase.build import bulk as ase_bulk

from mckit.core.conversion import to_ase_atoms, to_pymatgen_structure
from mckit.core.tool import Operation


class BulkBuilder(Operation):
    """Build a bulk crystal of the requested structure type.

    Example
    -------
    >>> BulkBuilder().apply(
    ...     structure_type="fcc", element="Cu", a=3.61,
    ... )
    """

    # Whitelist of structure types that ASE's bulk() understands.
    SUPPORTED = {
        "sc", "fcc", "bcc", "tetragonal", "bct", "hcp", "rhombohedral",
        "orthorhombic", "mcl", "diamond", "zincblende", "rocksalt",
        "cesiumchloride", "fluorite", "wurtzite",
    }
    _FORMULA_PART_RE = re.compile(r"([A-Za-z][a-z]?)(\d*)")

    def apply(
        self,
        *,
        structure_type: str = "fcc",
        element: Optional[str] = None,
        elements: Optional[Sequence[str]] = None,
        a: float = 1.0,
        c: Optional[float] = None,
        conventional_unit_cell: bool = True,
        **extra,
    ) -> Atoms:
        """Build a bulk structure.

        Parameters
        ----------
        structure_type
            Any crystal name supported by ``ase.build.bulk``.
        element
            Single element symbol or formula, e.g. ``"Cu"`` or ``"ZrO2"``.
        elements
            One or more element symbols, e.g. ``["Ga", "As"]`` or
            ``["Zr", "O", "O"]``.
        a, c
            Lattice parameters (Å). ``c`` defaults to the ideal value for
            ``hcp`` when omitted.
        **extra
            Forwarded to ``ase.build.bulk`` (e.g. ``orthorhombic=True``).
        """
        stype = structure_type.lower()
        if stype not in self.SUPPORTED:
            raise ValueError(
                f"Unknown structure type {structure_type!r}. "
                f"Supported: {sorted(self.SUPPORTED)}"
            )

        if elements is not None:
            if not elements:
                raise ValueError("`elements` cannot be empty.")
            name = "".join(self._normalize_formula_part(e) for e in elements)
        elif element is not None:
            name = self._normalize_formula(element)
        else:
            raise ValueError("Provide either `element` or `elements`.")

        kwargs = dict(extra)
        kwargs["crystalstructure"] = stype
        kwargs["a"] = a
        if c is not None:
            kwargs["c"] = c

        atoms = ase_bulk(name, **kwargs)
        if conventional_unit_cell:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            analyzer = SpacegroupAnalyzer(to_pymatgen_structure(atoms))
            atoms = to_ase_atoms(
                analyzer.get_conventional_standard_structure(),
            )
        return atoms

    @staticmethod
    def _normalize_formula_part(value) -> str:
        """Accept a single formula part, such as ``Cu`` or ``O2``."""
        if isinstance(value, str):
            symbol = value.strip()
        else:
            raise TypeError("Element inputs must be symbol strings like 'Cu'.")
        if not symbol:
            raise ValueError("Element symbol cannot be empty.")
        return BulkBuilder._normalize_formula(symbol)

    @classmethod
    def _normalize_formula(cls, value: str) -> str:
        """Normalize a chemical symbol or stoichiometric formula for ASE."""
        if not isinstance(value, str):
            raise TypeError("Element inputs must be symbol strings like 'Cu'.")

        formula = value.strip()
        if not formula:
            raise ValueError("Element symbol cannot be empty.")

        parts = []
        index = 0
        for match in cls._FORMULA_PART_RE.finditer(formula):
            if match.start() != index:
                raise ValueError(f"Invalid chemical formula {value!r}.")
            symbol, count = match.groups()
            parts.append(symbol.capitalize() + count)
            index = match.end()

        if index != len(formula):
            raise ValueError(f"Invalid chemical formula {value!r}.")
        return "".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_build(args):
    """CLI handler: build a bulk crystal."""
    from mckit.io import write_atoms

    builder = BulkBuilder()
    kwargs = dict(structure_type=args.type, a=args.a)
    if args.element:
        kwargs["element"] = args.element
    if args.elements:
        kwargs["elements"] = args.elements
    if args.c is not None:
        kwargs["c"] = args.c

    atoms = builder.apply(**kwargs)
    output = args.output or f"bulk_{args.type}.extxyz"
    path = write_atoms(output, atoms)
    print(f"Built {args.type} -> {path}  ({len(atoms)} atoms)")


def register_cli(subparsers) -> None:
    """Register bulk subcommands with the mmkit CLI."""
    bulk = subparsers.add_parser("bulk", help="Build bulk crystals")
    bulk_sub = bulk.add_subparsers(dest="action", required=True)

    p = bulk_sub.add_parser("build", help="Build a bulk crystal")
    p.add_argument("--type", required=True, choices=sorted(BulkBuilder.SUPPORTED),
                   help="Crystal structure type")
    species = p.add_mutually_exclusive_group(required=True)
    species.add_argument(
        "--element",
        help="Element symbol or formula, e.g. Cu or ZrO2",
    )
    species.add_argument(
        "--elements",
        nargs="+",
        help="One or more element symbols, e.g. Ga As or Zr O O",
    )
    p.add_argument("--a", type=float, required=True, help="Lattice parameter a (A)")
    p.add_argument("--c", type=float, help="Lattice parameter c (A, hcp only)")
    p.add_argument("--output", "-o", help="Output file (default: bulk_<type>.extxyz)")
    p.set_defaults(handler=_cmd_build)
