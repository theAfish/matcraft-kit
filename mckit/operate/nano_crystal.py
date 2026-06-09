"""Build finite nanocrystals by cutting geometric shapes from a bulk crystal."""

from __future__ import annotations

import argparse
from functools import reduce
from itertools import combinations, product
from math import gcd
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms

from mckit.core.conversion import StructureLike, to_ase_atoms
from mckit.core.structure import Structure
from mckit.core.tool import Operation

_SHAPE_ALIASES = {
    "sphere": "sphere",
    "spherical": "sphere",
    "cube": "cube",
    "box": "box",
    "cuboid": "box",
    "ellipsoid": "ellipsoid",
    "ellipsoidal": "ellipsoid",
    "cylinder": "cylinder",
    "cylindrical": "cylinder",
    "polyhedron": "polyhedron",
    "faceted": "polyhedron",
}


def _positive_values(
    value: float | Sequence[float],
    *,
    count: int,
    name: str,
    scalar_allowed: bool = False,
) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        if count != 1 and not scalar_allowed:
            raise ValueError(f"{name} must contain {count} values.")
        values = np.repeat(float(values), count)
    values = values.reshape(-1)
    if values.size != count:
        raise ValueError(f"{name} must contain {count} values, got {values.size}.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{name} values must be finite and greater than zero.")
    return values


def _polyhedron_geometry(
    cell: np.ndarray,
    miller_indices: Sequence[Sequence[int]],
    facet_distances: Sequence[float],
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    miller = np.asarray(miller_indices, dtype=float)
    if miller.ndim != 2 or miller.shape[1] != 3 or len(miller) < 3:
        raise ValueError("miller_indices must contain at least three (h, k, l) planes.")
    if np.any(np.all(np.isclose(miller, 0.0), axis=1)):
        raise ValueError("Miller index (0, 0, 0) is not a valid facet.")

    distances = _positive_values(
        facet_distances, count=len(miller), name="facet_distances",
    )
    reciprocal_normals = miller @ np.linalg.inv(cell).T
    normals = reciprocal_normals / np.linalg.norm(
        reciprocal_normals, axis=1, keepdims=True,
    )

    # The symmetric half-spaces must enclose a bounded volume. Determine its
    # vertices both to validate that condition and to size the source supercell.
    vertices = []
    for indices in combinations(range(len(normals)), 3):
        matrix = normals[list(indices)]
        if abs(np.linalg.det(matrix)) < 1e-10:
            continue
        selected_distances = distances[list(indices)]
        for signs in product((-1.0, 1.0), repeat=3):
            vertex = np.linalg.solve(matrix, selected_distances * signs)
            if np.all(np.abs(normals @ vertex) <= distances + tolerance):
                vertices.append(vertex)
    if not vertices:
        raise ValueError(
            "The Miller facets do not define a bounded three-dimensional polyhedron."
        )

    vertices_array = np.asarray(vertices)
    extent = float(np.linalg.norm(vertices_array, axis=1).max())
    return normals, distances, extent


def _canonical_miller(miller: Sequence[int]) -> tuple[int, int, int]:
    values = np.asarray(miller, dtype=int)
    divisor = reduce(gcd, (abs(int(value)) for value in values if value), 0)
    if divisor:
        values //= divisor
    first_nonzero = next((int(value) for value in values if value), 1)
    if first_nonzero < 0:
        values *= -1
    return tuple(int(value) for value in values)


def _expand_miller_families(
    atoms: Atoms,
    miller_indices: Sequence[Sequence[int]],
    facet_distances: Sequence[float],
    *,
    symprec: float,
) -> tuple[list[list[int]], list[float]]:
    """Expand Miller families using the input bulk's reciprocal-lattice symmetry."""
    miller = np.asarray(miller_indices, dtype=int)
    if miller.ndim != 2 or miller.shape[1] != 3:
        raise ValueError("miller_indices must contain (h, k, l) triplets.")
    distances = _positive_values(
        facet_distances, count=len(miller), name="facet_distances",
    )

    from pymatgen.core import Lattice

    lattice = Lattice(atoms.cell.array)
    reciprocal = lattice.reciprocal_lattice.matrix
    inverse_reciprocal = np.linalg.inv(reciprocal)
    operations = lattice.get_recp_symmetry_operation(symprec=symprec)

    expanded: dict[tuple[int, int, int], float] = {}
    for family, distance in zip(miller, distances):
        if not np.any(family):
            raise ValueError("Miller index (0, 0, 0) is not a valid facet.")
        reciprocal_vector = family @ reciprocal
        for operation in operations:
            transformed_vector = reciprocal_vector @ operation.rotation_matrix.T
            transformed = transformed_vector @ inverse_reciprocal
            transformed_int = np.rint(transformed).astype(int)
            if not np.allclose(transformed, transformed_int, atol=1e-6):
                continue
            key = _canonical_miller(transformed_int)
            expanded[key] = min(expanded.get(key, float("inf")), float(distance))

    return [list(key) for key in expanded], list(expanded.values())


class NanoCrystalBuilder(Operation):
    """Cut a finite nanocrystal from an arbitrary periodic bulk structure.

    ``size`` gives full dimensions in Angstrom: a sphere diameter, cube edge,
    box/ellipsoid ``(x, y, z)`` dimensions, or cylinder
    ``(diameter, length)``. For ``polyhedron``, provide Miller planes and their
    center-to-facet distances instead.
    """

    def apply(
        self,
        *,
        structure: StructureLike,
        shape: str,
        size: float | Sequence[float] | None = None,
        center: Sequence[float] = (0.0, 0.0, 0.0),
        vacuum: float = 10.0,
        axis: Sequence[float] = (0.0, 0.0, 1.0),
        miller_indices: Sequence[Sequence[int]] | None = None,
        facet_distances: Sequence[float] | None = None,
        expand_symmetry: bool = True,
        symprec: float = 1e-3,
        tolerance: float = 1e-8,
    ) -> Structure:
        """Build and return a nonperiodic nanocrystal."""
        atoms = to_ase_atoms(structure)
        if len(atoms) == 0:
            raise ValueError("The input bulk structure contains no atoms.")
        if not np.all(atoms.pbc):
            raise ValueError("The input bulk structure must be periodic in all directions.")

        cell = np.asarray(atoms.cell.array, dtype=float)
        if abs(np.linalg.det(cell)) < 1e-10:
            raise ValueError("The input bulk structure must have a non-singular cell.")

        shape_key = _SHAPE_ALIASES.get(shape.lower())
        if shape_key is None:
            choices = ", ".join(sorted(set(_SHAPE_ALIASES.values())))
            raise ValueError(f"Unknown shape {shape!r}. Choose from: {choices}.")
        if not np.isfinite(vacuum) or vacuum < 0:
            raise ValueError("vacuum must be finite and non-negative.")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative.")

        center_frac = np.asarray(center, dtype=float).reshape(-1)
        if center_frac.size != 3 or not np.all(np.isfinite(center_frac)):
            raise ValueError("center must contain three finite fractional coordinates.")
        center_frac %= 1.0
        center_cart = center_frac @ cell

        normals = distances = None
        if shape_key == "sphere":
            dimensions = _positive_values(size, count=1, name="size")
            radius = dimensions[0] / 2.0
            extent = radius
        elif shape_key == "cube":
            dimensions = _positive_values(
                size, count=3, name="size", scalar_allowed=True,
            )
            if not np.allclose(dimensions, dimensions[0]):
                raise ValueError("cube size must be a scalar or three equal values.")
            half_lengths = dimensions / 2.0
            extent = float(np.linalg.norm(half_lengths))
        elif shape_key in ("box", "ellipsoid"):
            dimensions = _positive_values(
                size, count=3, name="size", scalar_allowed=True,
            )
            half_lengths = dimensions / 2.0
            extent = (
                float(np.linalg.norm(half_lengths))
                if shape_key == "box"
                else float(half_lengths.max())
            )
        elif shape_key == "cylinder":
            dimensions = _positive_values(size, count=2, name="size")
            radius, half_length = dimensions[0] / 2.0, dimensions[1] / 2.0
            axis_uvw = np.asarray(axis, dtype=float).reshape(-1)
            if (
                axis_uvw.size != 3
                or not np.all(np.isfinite(axis_uvw))
                or np.linalg.norm(axis_uvw) < 1e-12
            ):
                raise ValueError("axis must contain three finite, non-zero values.")
            axis_cart = axis_uvw @ cell
            axis_unit = axis_cart / np.linalg.norm(axis_cart)
            extent = float(np.hypot(radius, half_length))
        else:
            if size is not None:
                raise ValueError("size is not used for polyhedron shapes.")
            if miller_indices is None or facet_distances is None:
                raise ValueError(
                    "polyhedron requires Miller families and facet distances. "
                    "CLI example: --miller-indices 1 1 1 --facet-distances 5"
                )
            if not np.isfinite(symprec) or symprec <= 0:
                raise ValueError("symprec must be finite and greater than zero.")
            if expand_symmetry:
                miller_indices, facet_distances = _expand_miller_families(
                    atoms,
                    miller_indices,
                    facet_distances,
                    symprec=symprec,
                )
            normals, distances, extent = _polyhedron_geometry(
                cell, miller_indices, facet_distances, tolerance,
            )

        inverse_cell = np.linalg.inv(cell)
        repeats_each_side = (
            np.ceil(extent * np.linalg.norm(inverse_cell, axis=0)).astype(int) + 1
        )
        repeats = tuple(2 * repeats_each_side + 1)
        candidates = atoms.repeat(repeats)
        central_translation = repeats_each_side @ cell
        relative = candidates.positions - central_translation - center_cart

        if shape_key == "sphere":
            mask = np.linalg.norm(relative, axis=1) <= radius + tolerance
        elif shape_key in ("cube", "box"):
            mask = np.all(np.abs(relative) <= half_lengths + tolerance, axis=1)
        elif shape_key == "ellipsoid":
            mask = np.sum((relative / half_lengths) ** 2, axis=1) <= 1.0 + tolerance
        elif shape_key == "cylinder":
            axial = relative @ axis_unit
            radial = relative - np.outer(axial, axis_unit)
            mask = (
                (np.abs(axial) <= half_length + tolerance)
                & (np.linalg.norm(radial, axis=1) <= radius + tolerance)
            )
        else:
            mask = np.all(
                np.abs(relative @ normals.T) <= distances + tolerance, axis=1,
            )

        cluster = candidates[mask]
        if len(cluster) == 0:
            raise ValueError(
                "The requested cut contains no atoms. Increase the size or change center."
            )

        cluster.positions = relative[mask]
        spans = np.ptp(cluster.positions, axis=0)
        cell_lengths = np.maximum(spans + 2.0 * vacuum, 1e-6)
        cluster.set_cell(np.diag(cell_lengths), scale_atoms=False)
        cluster.positions -= cluster.positions.min(axis=0)
        cluster.positions += vacuum
        cluster.pbc = False
        cluster.info.update(
            {
                "nanocrystal_shape": shape_key,
                "nanocrystal_center_fractional": center_frac.tolist(),
                "nanocrystal_vacuum": float(vacuum),
            }
        )
        return Structure(atoms=cluster)


def _parse_facets(values: Sequence[Sequence[str]]) -> tuple[list[list[int]], list[float]]:
    miller_indices = []
    distances = []
    for h, k, l, distance in values:
        miller_indices.append([int(h), int(k), int(l)])
        distances.append(float(distance))
    return miller_indices, distances


def _cli_polyhedron_args(args) -> tuple[list[list[int]] | None, list[float] | None]:
    if args.facet and (args.miller_indices or args.facet_distances):
        raise ValueError(
            "Use either --facet H K L DIST, or --miller-indices H K L with "
            "--facet-distances DIST; do not mix both forms."
        )
    if args.facet:
        return _parse_facets(args.facet)
    if args.miller_indices or args.facet_distances:
        if not args.miller_indices or not args.facet_distances:
            raise ValueError(
                "--miller-indices and --facet-distances must be provided together."
            )
        if len(args.miller_indices) != len(args.facet_distances):
            raise ValueError(
                "Provide one --facet-distances value for each --miller-indices "
                f"triplet ({len(args.miller_indices)} indices, "
                f"{len(args.facet_distances)} distances)."
            )
        return args.miller_indices, args.facet_distances
    return None, None


def _cmd_build(args) -> None:
    from mckit.io import read_structure, write_structure

    atoms = read_structure(args.input)
    miller_indices, facet_distances = _cli_polyhedron_args(args)

    result = NanoCrystalBuilder().apply(
        structure=atoms,
        shape=args.shape,
        size=args.size,
        center=args.center,
        vacuum=args.vacuum,
        axis=args.axis,
        miller_indices=miller_indices,
        facet_distances=facet_distances,
    )
    output = args.output or f"nanocrystal_{Path(args.input).stem}.extxyz"
    path = write_structure(output, result)
    print(f"Nanocrystal: {len(result)} atoms, shape={args.shape} -> {path}")


def register_cli(subparsers) -> None:
    """Register the ``mckit operate nanocrystal`` command."""
    parser = subparsers.add_parser(
        "nanocrystal",
        help="Cut a finite nanocrystal from a periodic bulk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  Sphere with 20 A diameter:
    mckit operate nanocrystal bulk.cif --shape sphere --size 20

  Cubic-symmetry octahedron bounded by the {111} family at 10 A:
    mckit operate nanocrystal bulk.cif --shape polyhedron \\
      --miller-indices 1 1 1 --facet-distances 10

  Truncated shape using {111} and {100} families:
    mckit operate nanocrystal bulk.cif --shape polyhedron \\
      --miller-indices 1 1 1 --facet-distances 10 \\
      --miller-indices 1 0 0 --facet-distances 12

  Compact equivalent form:
    mckit operate nanocrystal bulk.cif --shape polyhedron \\
      --facet 1 1 1 10 --facet 1 0 0 12

Each Miller index denotes a symmetry family. Equivalent facets are generated
from the detected bulk point symmetry. DIST is the center-to-facet distance
in Angstrom.""",
    )
    parser.add_argument("input", help="Input bulk structure file")
    parser.add_argument(
        "--shape",
        required=True,
        choices=("sphere", "cube", "box", "ellipsoid", "cylinder", "polyhedron"),
    )
    parser.add_argument(
        "--size",
        type=float,
        nargs="+",
        help="Full dimensions in A (not used for polyhedron)",
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("F1", "F2", "F3"),
        help="Cut center in fractional bulk coordinates (default: 0 0 0)",
    )
    parser.add_argument(
        "--axis",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("U", "V", "W"),
        help="Cylinder axis as a lattice direction (default: 0 0 1)",
    )
    parser.add_argument(
        "--facet",
        action="append",
        nargs=4,
        metavar=("H", "K", "L", "DIST"),
        help="Miller family and center distance in A; repeat for more families",
    )
    parser.add_argument(
        "--miller-indices",
        "--miller_indices",
        action="append",
        type=int,
        nargs=3,
        metavar=("H", "K", "L"),
        help="Miller family; repeat together with --facet-distances",
    )
    parser.add_argument(
        "--facet-distances",
        "--facet_distances",
        action="append",
        type=float,
        metavar="DIST",
        help="Center-to-facet distance in A for the corresponding Miller family",
    )
    parser.add_argument("--vacuum", type=float, default=10.0)
    parser.add_argument("--output", "-o")
    parser.set_defaults(handler=_cmd_build)
