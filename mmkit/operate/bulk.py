"""Build standard bulk crystals via ``ase.build.bulk``.

ASE's ``bulk`` already handles every structure type we care about
(``fcc``, ``bcc``, ``hcp``, ``diamond``, ``zincblende``, ``rocksalt``,
``cesiumchloride``, ``fluorite``, ``wurtzite``, ...) so this class is just
a thin matmod-flavored wrapper.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ase.build import bulk as ase_bulk

from mmkit.core.structure import Structure
from mmkit.core.tool import Operation


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
    ) -> Structure:
        """Build a bulk structure.

        Parameters
        ----------
        structure_type
            Any crystal name supported by ``ase.build.bulk``.
        element
            Single element symbol, e.g. ``"Cu"``.
        elements
            Two element symbols for binary types, e.g. ``["Ga", "As"]``.
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
            if len(elements) != 2:
                raise ValueError("`elements` must have length 2.")
            name = "".join(self._normalize_element_symbol(e) for e in elements)
        elif element is not None:
            name = self._normalize_element_symbol(element)
        else:
            raise ValueError("Provide either `element` or `elements`.")

        kwargs = dict(extra)
        kwargs["crystalstructure"] = stype
        kwargs["a"] = a
        if c is not None:
            kwargs["c"] = c

        atoms = ase_bulk(name, **kwargs)
        structure = Structure.from_ase_atoms(atoms).to_pymatgen()
        if conventional_unit_cell:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            analyzer = SpacegroupAnalyzer(structure)
            structure = analyzer.get_conventional_standard_structure()
            # structure = Structure.from_pymatgen(pmg_conv)
        return structure

    @staticmethod
    def _normalize_element_symbol(value) -> str:
        """Accept and normalize a non-empty chemical symbol string."""
        if isinstance(value, str):
            symbol = value.strip()
        else:
            raise TypeError("Element inputs must be symbol strings like 'Cu'.")
        if not symbol:
            raise ValueError("Element symbol cannot be empty.")
        return symbol.capitalize()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_build(args):
    """CLI handler: build a bulk crystal."""
    from mmkit.io import write_atoms

    builder = BulkBuilder()
    kwargs = dict(structure_type=args.type, a=args.a)
    if args.element:
        kwargs["element"] = args.element
    if args.elements:
        kwargs["elements"] = args.elements
    if args.c is not None:
        kwargs["c"] = args.c

    structure = builder.apply(**kwargs)
    output = args.output or f"bulk_{args.type}.extxyz"
    path = write_atoms(output, structure.atoms)
    print(f"Built {args.type} -> {path}  ({len(structure.atoms)} atoms)")


def register_cli(subparsers) -> None:
    """Register bulk subcommands with the mmkit CLI."""
    bulk = subparsers.add_parser("bulk", help="Build bulk crystals")
    bulk_sub = bulk.add_subparsers(dest="action", required=True)

    p = bulk_sub.add_parser("build", help="Build a bulk crystal")
    p.add_argument("--type", required=True, choices=sorted(BulkBuilder.SUPPORTED),
                   help="Crystal structure type")
    p.add_argument("--element", help="Element symbol, e.g. Cu")
    p.add_argument("--elements", nargs=2, help="Two element symbols for binary types, e.g. Ga As")
    p.add_argument("--a", type=float, required=True, help="Lattice parameter a (A)")
    p.add_argument("--c", type=float, help="Lattice parameter c (A, hcp only)")
    p.add_argument("--output", "-o", help="Output file (default: bulk_<type>.extxyz)")
    p.set_defaults(handler=_cmd_build)
