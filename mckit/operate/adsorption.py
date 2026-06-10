"""Create surface adsorption structures."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.geometry import find_mic, minkowski_reduce

from mckit.core.conversion import StructureLike, to_ase_atoms
from mckit.core.tool import Operation


def _load_structure(structure: str | StructureLike) -> Atoms:
    if isinstance(structure, str):
        from mckit.io import read_structure

        return read_structure(structure)
    return to_ase_atoms(structure)


def _rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a rotation matrix that maps *source* onto *target*."""
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        trial = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, trial)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    axis = cross / sine
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


def _axis_rotation(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(angle_degrees)
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (
        np.eye(3)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


class AdsorptionBuilder(Operation):
    """Place an adsorbate on a periodic slab.

    The adsorption site is given in fractional coordinates of the slab's
    first two lattice vectors. ``height`` is measured from the outermost slab
    atom along the surface normal.
    """

    def apply(
        self,
        *,
        slab: str | StructureLike,
        adsorbate: str | StructureLike,
        site: Sequence[float] = (0.5, 0.5),
        height: float = 2.0,
        anchor: int = 0,
        orientation_atom: Optional[int] = None,
        azimuth: float = 0.0,
        side: str = "top",
        min_distance: float = 0.7,
        covalent_scale: Optional[float] = None,
        check_periodic_images: bool = True,
    ) -> Atoms:
        """Build and return a slab-adsorbate structure.

        ``anchor`` is the adsorbate atom placed at the requested site.
        When ``orientation_atom`` is supplied, the vector from the anchor to
        that atom is aligned with the outward surface normal before applying
        the azimuthal rotation.
        """
        slab_atoms = _load_structure(slab)
        molecule = _load_structure(adsorbate)
        self._validate(
            slab_atoms, molecule, site, height, anchor, orientation_atom,
            side, min_distance, covalent_scale,
        )

        cell = slab_atoms.cell.array
        normal = np.cross(cell[0], cell[1])
        normal /= np.linalg.norm(normal)
        outward = normal if side == "top" else -normal

        positions = molecule.positions.copy()
        anchor_position = positions[anchor].copy()
        positions -= anchor_position

        if orientation_atom is not None:
            orientation = positions[orientation_atom]
            rotation = _rotation_from_vectors(orientation, outward)
            positions = positions @ rotation.T

        if azimuth:
            rotation = _axis_rotation(outward, azimuth)
            positions = positions @ rotation.T

        if check_periodic_images:
            self._check_surface_cell_fit(
                slab_atoms,
                molecule.numbers,
                positions,
                min_distance=min_distance,
                covalent_scale=covalent_scale,
            )

        surface_coordinates = slab_atoms.positions @ normal
        surface_level = (
            float(surface_coordinates.max())
            if side == "top"
            else float(surface_coordinates.min())
        )
        target_level = surface_level + (height if side == "top" else -height)
        u, v = float(site[0]) % 1.0, float(site[1]) % 1.0
        target = u * cell[0] + v * cell[1] + target_level * normal
        positions += target

        self._check_collisions(
            slab_atoms,
            molecule.numbers,
            positions,
            min_distance=min_distance,
            covalent_scale=covalent_scale,
        )

        placed = molecule.copy()
        placed.positions = positions
        placed.cell = slab_atoms.cell
        placed.pbc = slab_atoms.pbc
        result = slab_atoms + placed
        result.set_array(
            "adsorbate_mask",
            np.r_[np.zeros(len(slab_atoms), dtype=bool),
                  np.ones(len(placed), dtype=bool)],
        )
        return result

    def apply_density(
        self,
        *,
        slab: str | StructureLike,
        adsorbate: str | StructureLike,
        density: float,
        count: int,
        height: float = 2.0,
        anchor: int = 0,
        orientation_atom: Optional[int] = None,
        side: str = "top",
        min_distance: float = 0.7,
        adsorbate_min_distance: float = 2.0,
        covalent_scale: Optional[float] = None,
        random_azimuth: bool = True,
        seed: Optional[int] = None,
        max_repeat: int = 20,
        attempts_per_molecule: int = 24,
    ) -> Atoms:
        """Place multiple adsorbates at a target surface number density.

        ``density`` is the maximum number density in molecules/nm^2. The slab
        is repeated in plane until its area is at least ``count / density`` and
        all molecules can be placed without slab, mutual, or periodic-image
        overlaps. Integer repeats mean the achieved density can be lower.
        """
        slab_atoms = _load_structure(slab)
        molecule = _load_structure(adsorbate)
        self._validate(
            slab_atoms, molecule, (0.5, 0.5), height, anchor,
            orientation_atom, side, min_distance, covalent_scale,
        )
        if density <= 0:
            raise ValueError("density must be positive.")
        if count <= 0:
            raise ValueError("count must be a positive integer.")
        if adsorbate_min_distance < 0:
            raise ValueError("adsorbate_min_distance must be non-negative.")
        if max_repeat <= 0:
            raise ValueError("max_repeat must be positive.")
        if attempts_per_molecule <= 0:
            raise ValueError("attempts_per_molecule must be positive.")

        base_area = float(np.linalg.norm(
            np.cross(slab_atoms.cell[0], slab_atoms.cell[1]),
        ))
        required_area = count * 100.0 / density
        candidates = [
            (nx * ny * base_area, nx, ny)
            for nx in range(1, max_repeat + 1)
            for ny in range(1, max_repeat + 1)
            if nx * ny * base_area >= required_area - 1e-10
        ]
        length_a = float(np.linalg.norm(slab_atoms.cell[0]))
        length_b = float(np.linalg.norm(slab_atoms.cell[1]))
        candidates.sort(key=lambda item: (
            int(np.ceil(
                max(0.0, item[0] / required_area - 1.0) / 0.1 - 1e-12,
            )),
            abs(length_a * item[1] - length_b * item[2]),
            item[0],
        ))
        if not candidates:
            raise ValueError(
                "Requested density and count require a slab larger than "
                f"max_repeat={max_repeat}. Increase max_repeat."
            )

        density_area, density_nx, density_ny = candidates[0]
        rng = np.random.default_rng(seed)
        for candidate_index, (_area, nx, ny) in enumerate(candidates):
            repeated = slab_atoms.repeat((nx, ny, 1))
            result = self._try_density_placement(
                repeated,
                molecule,
                count=count,
                height=height,
                anchor=anchor,
                orientation_atom=orientation_atom,
                side=side,
                min_distance=min_distance,
                adsorbate_min_distance=adsorbate_min_distance,
                covalent_scale=covalent_scale,
                random_azimuth=random_azimuth,
                rng=rng,
                attempts_per_molecule=attempts_per_molecule,
            )
            if result is not None:
                area_nm2 = _area / 100.0
                achieved_density = count / area_nm2
                achieved_min_distance = (
                    self._minimum_interadsorbate_distance(result)
                )
                if candidate_index > 0:
                    selection_limit = "molecule_fit"
                elif nx == 1 and ny == 1 and required_area <= base_area:
                    selection_limit = "base_slab"
                elif abs(_area - required_area) > 1e-8:
                    selection_limit = "integer_repeat"
                else:
                    selection_limit = "density"
                result.info.update({
                    "adsorbate_count": count,
                    "adsorbate_density_per_nm2": achieved_density,
                    "requested_adsorbate_density_per_nm2": density,
                    "slab_repeat": (nx, ny, 1),
                    "density_limited_slab_repeat": (
                        density_nx, density_ny, 1,
                    ),
                    "base_slab_area_ang2": base_area,
                    "required_surface_area_ang2": required_area,
                    "selected_surface_area_ang2": _area,
                    "density_limited_surface_area_ang2": density_area,
                    "density_selection_limit": selection_limit,
                    "density_candidates_tried": candidate_index + 1,
                    "adsorbate_min_distance_ang": adsorbate_min_distance,
                    "minimum_interadsorbate_distance_ang": (
                        achieved_min_distance
                    ),
                })
                return result

        raise ValueError(
            "Could not place all adsorbates without overlap. Reduce density, "
            "increase height or max_repeat, or lower the distance threshold."
        )

    def _try_density_placement(
        self,
        slab: Atoms,
        molecule: Atoms,
        *,
        count: int,
        height: float,
        anchor: int,
        orientation_atom: Optional[int],
        side: str,
        min_distance: float,
        adsorbate_min_distance: float,
        covalent_scale: Optional[float],
        random_azimuth: bool,
        rng: np.random.Generator,
        attempts_per_molecule: int,
    ) -> Optional[Atoms]:
        cell = slab.cell.array
        normal = np.cross(cell[0], cell[1])
        normal /= np.linalg.norm(normal)
        outward = normal if side == "top" else -normal
        surface_coordinates = slab.positions @ normal
        surface_level = (
            float(surface_coordinates.max())
            if side == "top"
            else float(surface_coordinates.min())
        )
        target_level = surface_level + (height if side == "top" else -height)

        base_positions = molecule.positions - molecule.positions[anchor]
        if orientation_atom is not None:
            rotation = _rotation_from_vectors(
                base_positions[orientation_atom], outward,
            )
            base_positions = base_positions @ rotation.T

        sites = self._dispersed_sites(cell, count, rng)
        placed_positions: list[np.ndarray] = []
        placed_numbers: list[np.ndarray] = []
        placed_molecules: list[Atoms] = []
        for molecule_index, site in enumerate(sites):
            placed = None
            for _ in range(attempts_per_molecule):
                positions = base_positions.copy()
                if random_azimuth:
                    rotation = _axis_rotation(
                        outward, float(rng.uniform(0.0, 360.0)),
                    )
                    positions = positions @ rotation.T
                target = (
                    site[0] * cell[0]
                    + site[1] * cell[1]
                    + target_level * normal
                )
                positions += target
                try:
                    self._check_surface_cell_fit(
                        slab,
                        molecule.numbers,
                        positions,
                        min_distance=min_distance,
                        covalent_scale=covalent_scale,
                    )
                    self._check_collisions(
                        slab,
                        molecule.numbers,
                        positions,
                        min_distance=min_distance,
                        covalent_scale=covalent_scale,
                    )
                    if placed_positions:
                        self._check_adsorbate_collisions(
                            molecule.numbers,
                            positions,
                            np.concatenate(placed_numbers),
                            np.concatenate(placed_positions),
                            slab,
                            min_distance=adsorbate_min_distance,
                            covalent_scale=covalent_scale,
                        )
                except ValueError:
                    continue

                placed = molecule.copy()
                placed.positions = positions
                placed.cell = slab.cell
                placed.pbc = slab.pbc
                placed.set_array(
                    "adsorbate_id",
                    np.full(len(placed), molecule_index, dtype=int),
                )
                break

            if placed is None:
                return None
            placed_positions.append(placed.positions.copy())
            placed_numbers.append(placed.numbers.copy())
            placed_molecules.append(placed)

        slab_result = slab.copy()
        slab_result.set_array(
            "adsorbate_id", np.full(len(slab_result), -1, dtype=int),
        )
        result = slab_result
        for placed in placed_molecules:
            result += placed
        result.set_array("adsorbate_mask", result.arrays["adsorbate_id"] >= 0)
        return result

    @staticmethod
    def _dispersed_sites(
        cell: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate periodic in-plane sites with farthest-point sampling."""
        if count == 1:
            return rng.random((1, 2))

        length_a = float(np.linalg.norm(cell[0]))
        length_b = float(np.linalg.norm(cell[1]))
        pool_size = max(64, count * 16)
        grid_a = max(
            2,
            int(np.ceil(np.sqrt(pool_size * length_a / length_b))),
        )
        grid_b = max(2, int(np.ceil(pool_size / grid_a)))
        shift = rng.random(2)
        candidates = np.array([
            ((i + 0.5) / grid_a + shift[0],
             (j + 0.5) / grid_b + shift[1])
            for i in range(grid_a)
            for j in range(grid_b)
        ]) % 1.0

        selected = [int(rng.integers(len(candidates)))]
        min_distances = np.full(len(candidates), np.inf)
        for _ in range(1, count):
            delta = candidates - candidates[selected[-1]]
            delta -= np.round(delta)
            cart_delta = (
                delta[:, 0, None] * cell[0]
                + delta[:, 1, None] * cell[1]
            )
            min_distances = np.minimum(
                min_distances, np.linalg.norm(cart_delta, axis=1),
            )
            min_distances[selected] = -1.0
            selected.append(int(np.argmax(min_distances)))
        return candidates[selected]

    @staticmethod
    def _validate(
        slab: Atoms,
        adsorbate: Atoms,
        site: Sequence[float],
        height: float,
        anchor: int,
        orientation_atom: Optional[int],
        side: str,
        min_distance: float,
        covalent_scale: Optional[float],
    ) -> None:
        if len(slab) == 0 or len(adsorbate) == 0:
            raise ValueError("Slab and adsorbate must both contain atoms.")
        if len(site) != 2:
            raise ValueError("site must contain two fractional coordinates.")
        if np.linalg.norm(np.cross(slab.cell[0], slab.cell[1])) < 1e-12:
            raise ValueError("Slab must have two non-collinear in-plane vectors.")
        if not 0 <= anchor < len(adsorbate):
            raise IndexError(f"Anchor index {anchor} is out of range.")
        if orientation_atom is not None:
            if not 0 <= orientation_atom < len(adsorbate):
                raise IndexError(
                    f"Orientation atom index {orientation_atom} is out of range."
                )
            if orientation_atom == anchor:
                raise ValueError("orientation_atom must differ from anchor.")
            vector = adsorbate.positions[orientation_atom] - adsorbate.positions[anchor]
            if np.linalg.norm(vector) < 1e-12:
                raise ValueError("Anchor and orientation atom occupy the same position.")
        if side not in {"top", "bottom"}:
            raise ValueError("side must be 'top' or 'bottom'.")
        if height < 0:
            raise ValueError("height must be non-negative.")
        if min_distance < 0:
            raise ValueError("min_distance must be non-negative.")
        if covalent_scale is not None and covalent_scale <= 0:
            raise ValueError("covalent_scale must be positive.")

    @staticmethod
    def _check_collisions(
        slab: Atoms,
        adsorbate_numbers: np.ndarray,
        adsorbate_positions: np.ndarray,
        *,
        min_distance: float,
        covalent_scale: Optional[float],
    ) -> None:
        for ads_number, ads_position in zip(
            adsorbate_numbers, adsorbate_positions,
        ):
            deltas = slab.positions - ads_position
            _, distances = find_mic(
                deltas,
                slab.cell,
                pbc=(True, True, False),
            )
            thresholds = np.full(len(slab), min_distance)
            if covalent_scale is not None:
                thresholds = np.maximum(
                    thresholds,
                    covalent_scale
                    * (covalent_radii[slab.numbers] + covalent_radii[ads_number]),
                )
            colliding = np.flatnonzero(distances < thresholds)
            if colliding.size:
                index = int(colliding[np.argmin(distances[colliding])])
                raise ValueError(
                    "Adsorbate overlaps the slab: "
                    f"distance {distances[index]:.3f} A to slab atom {index}. "
                    "Increase height or change the adsorption site."
                )

    @staticmethod
    def _check_surface_cell_fit(
        slab: Atoms,
        adsorbate_numbers: np.ndarray,
        adsorbate_positions: np.ndarray,
        *,
        min_distance: float,
        covalent_scale: Optional[float],
    ) -> None:
        """Reject adsorbates that overlap their in-plane periodic images."""
        reduced_cell, _ = minkowski_reduce(
            slab.cell, pbc=(True, True, False),
        )
        in_plane = np.asarray(reduced_cell[:2])
        coefficients = np.linalg.lstsq(
            in_plane.T, adsorbate_positions.T, rcond=None,
        )[0].T
        fractional_span = np.ptp(coefficients, axis=0)
        if np.any(fractional_span >= 1.0 - 1e-10):
            raise ValueError(
                "Adsorbate is too large for the slab surface cell: its "
                "in-plane footprint spans a full periodic cell. Build a "
                "larger surface supercell or disable periodic-image checks."
            )

        shifts = [
            i * in_plane[0] + j * in_plane[1]
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            if i != 0 or j != 0
        ]
        radii = covalent_radii[adsorbate_numbers]
        for shift in shifts:
            deltas = (
                adsorbate_positions[:, None, :]
                - (adsorbate_positions[None, :, :] + shift)
            )
            distances = np.linalg.norm(deltas, axis=2)
            thresholds = np.full(distances.shape, min_distance)
            if covalent_scale is not None:
                thresholds = np.maximum(
                    thresholds,
                    covalent_scale * (radii[:, None] + radii[None, :]),
                )
            colliding = np.argwhere(distances < thresholds)
            if colliding.size:
                atom, image_atom = map(int, colliding[0])
                raise ValueError(
                    "Adsorbate is too large for the slab surface cell: "
                    f"atom {atom} is {distances[atom, image_atom]:.3f} A "
                    f"from periodic-image atom {image_atom}. Build a larger "
                    "surface supercell or disable periodic-image checks."
                )

    @staticmethod
    def _check_adsorbate_collisions(
        candidate_numbers: np.ndarray,
        candidate_positions: np.ndarray,
        placed_numbers: np.ndarray,
        placed_positions: np.ndarray,
        slab: Atoms,
        *,
        min_distance: float,
        covalent_scale: Optional[float],
    ) -> None:
        for number, position in zip(candidate_numbers, candidate_positions):
            deltas = placed_positions - position
            _, distances = find_mic(
                deltas, slab.cell, pbc=(True, True, False),
            )
            thresholds = np.full(len(placed_positions), min_distance)
            if covalent_scale is not None:
                thresholds = np.maximum(
                    thresholds,
                    covalent_scale
                    * (covalent_radii[placed_numbers] + covalent_radii[number]),
                )
            if np.any(distances < thresholds):
                raise ValueError("Adsorbates overlap each other.")

    @staticmethod
    def _minimum_interadsorbate_distance(result: Atoms) -> Optional[float]:
        mask = result.arrays["adsorbate_mask"]
        positions = result.positions[mask]
        molecule_ids = result.arrays["adsorbate_id"][mask]
        if len(np.unique(molecule_ids)) < 2:
            return None

        minimum = float("inf")
        for index in range(len(positions) - 1):
            different = molecule_ids[index + 1:] != molecule_ids[index]
            if not np.any(different):
                continue
            deltas = positions[index + 1:][different] - positions[index]
            _, distances = find_mic(
                deltas, result.cell, pbc=(True, True, False),
            )
            minimum = min(minimum, float(np.min(distances)))
        return minimum


def _cmd_build(args) -> None:
    from mckit.io import write_structure

    builder = AdsorptionBuilder()
    if args.density is not None:
        if args.count is None:
            raise ValueError("--count is required when --density is used.")
        result = builder.apply_density(
            slab=args.slab,
            adsorbate=args.adsorbate,
            density=args.density,
            count=args.count,
            height=args.height,
            anchor=args.anchor,
            orientation_atom=args.orientation_atom,
            side=args.side,
            min_distance=args.min_distance,
            adsorbate_min_distance=args.adsorbate_min_distance,
            covalent_scale=args.covalent_scale,
            random_azimuth=not args.fixed_azimuth,
            seed=args.seed,
            max_repeat=args.max_repeat,
        )
    else:
        result = builder.apply(
            slab=args.slab,
            adsorbate=args.adsorbate,
            site=args.site,
            height=args.height,
            anchor=args.anchor,
            orientation_atom=args.orientation_atom,
            azimuth=args.azimuth,
            side=args.side,
            min_distance=args.min_distance,
            covalent_scale=args.covalent_scale,
            check_periodic_images=not args.allow_periodic_overlap,
        )
    output = args.output or (
        f"{Path(args.slab).stem}_{Path(args.adsorbate).stem}_adsorption.extxyz"
    )
    path = write_structure(output, result)
    print(
        f"Built {args.side} adsorption structure with {len(result)} atoms -> {path}"
    )
    if args.density is not None:
        info = result.info
        repeat = info["slab_repeat"]
        area = info["selected_surface_area_ang2"]
        requested = info["requested_adsorbate_density_per_nm2"]
        achieved = info["adsorbate_density_per_nm2"]
        limit = info["density_selection_limit"]
        explanations = {
            "base_slab": (
                "the input slab is already the smallest available cell"
            ),
            "integer_repeat": (
                "integer slab repeats cannot match the requested area exactly"
            ),
            "molecule_fit": (
                "earlier density-compatible repeats could not place all "
                "molecules without overlap"
            ),
            "density": "the requested area is matched exactly",
        }
        print(
            f"  Adsorbates: {info['adsorbate_count']}; "
            f"requested maximum density: {requested:.6g} molecules/nm^2"
        )
        print(
            f"  Selected repeat: {repeat[0]} x {repeat[1]}; "
            f"surface area: {area:.3f} A^2; "
            f"achieved density: {achieved:.6g} molecules/nm^2"
        )
        print(
            "  Inter-adsorbate atom distance: "
            f"required >= {info['adsorbate_min_distance_ang']:.3f} A"
        )
        actual_distance = info["minimum_interadsorbate_distance_ang"]
        if actual_distance is not None:
            print(f"  Achieved minimum distance: {actual_distance:.3f} A")
        if limit == "molecule_fit":
            density_repeat = info["density_limited_slab_repeat"]
            print(
                f"  Area-only minimum repeat was "
                f"{density_repeat[0]} x {density_repeat[1]}; "
                f"{info['density_candidates_tried']} candidates were tried."
            )
        print(f"  Selection limit: {explanations[limit]}.")


def register_cli(subparsers) -> None:
    adsorption = subparsers.add_parser(
        "adsorption",
        help="Place an adsorbate on a surface slab",
    )
    adsorption.add_argument("slab", help="Surface slab structure file")
    adsorption.add_argument("adsorbate", help="Adsorbate structure file")
    adsorption.add_argument(
        "--site", type=float, nargs=2, default=(0.5, 0.5),
        metavar=("U", "V"),
        help="Fractional coordinates along the slab a/b vectors",
    )
    adsorption.add_argument(
        "--height", type=float, default=2.0,
        help="Anchor height above the outermost slab atom in A",
    )
    adsorption.add_argument(
        "--anchor", type=int, default=0,
        help="Zero-based adsorbate anchor atom index",
    )
    adsorption.add_argument(
        "--orientation-atom", type=int,
        help="Atom defining the anchor-to-outward orientation vector",
    )
    adsorption.add_argument(
        "--azimuth", type=float, default=0.0,
        help="Rotation around the outward surface normal in degrees",
    )
    adsorption.add_argument(
        "--side", choices=("top", "bottom"), default="top",
        help="Slab side on which to place the adsorbate",
    )
    adsorption.add_argument(
        "--min-distance", type=float, default=0.7,
        help="Minimum allowed adsorbate-slab distance in A",
    )
    adsorption.add_argument(
        "--adsorbate-min-distance", type=float, default=2.0,
        help="Minimum distance between atoms in different adsorbates in A",
    )
    adsorption.add_argument(
        "--covalent-scale", type=float,
        help="Also enforce this fraction of summed covalent radii",
    )
    adsorption.add_argument(
        "--allow-periodic-overlap", action="store_true",
        help="Allow an adsorbate that spans or contacts periodic surface images",
    )
    adsorption.add_argument(
        "--density", type=float,
        help="Maximum adsorbate number density in molecules/nm^2",
    )
    adsorption.add_argument(
        "--count", type=int,
        help="Number of molecules to place in density mode",
    )
    adsorption.add_argument(
        "--seed", type=int,
        help="Random seed for density-mode placement",
    )
    adsorption.add_argument(
        "--fixed-azimuth", action="store_true",
        help="Do not randomize molecular azimuths in density mode",
    )
    adsorption.add_argument(
        "--max-repeat", type=int, default=20,
        help="Maximum automatic repeat along either in-plane direction",
    )
    adsorption.add_argument("-o", "--output", help="Output structure file")
    adsorption.set_defaults(handler=_cmd_build)
