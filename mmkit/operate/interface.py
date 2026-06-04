"""Build coherent ZSL-matched interfaces between two bulk crystals.

Uses pymatgen's ``SubstrateAnalyzer`` to find the best lattice match
(lowest von Mises strain), then ``CoherentInterfaceBuilder`` to construct
the interface with the specified gap, vacuum, and slab thicknesses.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

from ase import Atoms
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.core import Structure as PmgStructure
from pymatgen.io.ase import AseAtomsAdaptor

from mmkit.core.structure import Structure
from mmkit.core.tool import Operation


class InterfaceBuilder(Operation):
    """Build a coherent ZSL-matched interface between two bulk crystals.

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
        max_area: Optional[float] = 400.0,
        max_length_tol: float = 0.03,
        max_angle_tol: float = 0.01,
        gap: float = 2.5,
        vacuum_between: Optional[float] = 0.0,
        thickness_film: int = 2,
        thickness_substrate: int = 2,
        in_layers: bool = True,
    ) -> PmgStructure:
        """Build a coherent interface from two bulk structures.

        Parameters
        ----------
        film, substrate
            The film and substrate bulk structures.  Accepts ``ase.Atoms``,
            ``pymatgen.Structure``, or ``mmkit.Structure``.
        miller_film, miller_substrate
            Miller indices of the surfaces to expose.
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
        """
        film_pmg = self._to_pymatgen(film)
        substrate_pmg = self._to_pymatgen(substrate)

        # --- ZSL matching ---------------------------------------------------
        analyzer = SubstrateAnalyzer(
            max_area_ratio_tol=0.09,
            max_area=max_area,
            max_length_tol=max_length_tol,
            max_angle_tol=max_angle_tol,
        )
        matches = list(analyzer.calculate(
            film=film_pmg,
            substrate=substrate_pmg,
            film_millers=[miller_film],
            substrate_millers=[miller_substrate],
        ))
        if not matches:
            raise ValueError(
                "No lattice matches found. Try adjusting tolerances or Miller indices."
            )

        match = sorted(matches, key=lambda m: m.von_mises_strain)[0]

        # --- Interface construction -----------------------------------------
        builder = CoherentInterfaceBuilder(
            film_structure=film_pmg,
            substrate_structure=substrate_pmg,
            film_miller=match.film_miller,
            substrate_miller=match.substrate_miller,
            zslgen=analyzer,
        )
        terminations = builder.terminations
        if not terminations:
            raise ValueError("No terminations available for the selected slabs.")

        effective_vacuum = vacuum_between if vacuum_between != 0 else gap
        interfaces = list(builder.get_interfaces(
            termination=terminations[0],
            gap=gap,
            vacuum_over_film=effective_vacuum,
            film_thickness=thickness_film,
            substrate_thickness=thickness_substrate,
            in_layers=in_layers,
        ))
        if not interfaces:
            raise ValueError("No interfaces generated. Check parameters.")

        interface = interfaces[0]

        # Wrap all sites back into the unit cell [0, 1).
        species = [site.specie for site in interface]
        frac_coords = [site.frac_coords % 1.0 for site in interface]
        interface = PmgStructure(interface.lattice, species, frac_coords)

        return interface

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
        max_area=args.max_area,
        max_length_tol=args.max_length_tol,
        max_angle_tol=args.max_angle_tol,
        gap=args.gap,
        vacuum_between=args.vacuum,
        thickness_film=args.thickness_film,
        thickness_substrate=args.thickness_substrate,
        in_layers=not args.angstrom_thickness,
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
    p.add_argument("--film", required=True, help="Film bulk structure file")
    p.add_argument("--substrate", required=True, help="Substrate bulk structure file")
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
