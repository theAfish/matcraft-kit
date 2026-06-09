"""Extract, commensurate, and stack van der Waals layers."""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.neighborlist import natural_cutoffs, neighbor_list

from mckit.core.structure import Structure
from mckit.core.tool import Operation


StructureLike = Union[Atoms, Structure, "PmgStructure"]


@dataclass
class LayerInfo:
    """Metadata for one extracted and normalized layer."""

    source_index: int
    component_index: int
    source_atoms: int
    layer_atoms: int
    normal: np.ndarray
    thickness: float


@dataclass
class StackMatch:
    """Commensurate match selected while adding one layer."""

    layer_index: int
    requested_angle: float
    actual_angle: float
    angle_error: float
    area: float
    max_strain: float
    layer_transform: np.ndarray
    stack_transform: np.ndarray


@dataclass
class VdWStackResult:
    """Stacked structure plus extraction and matching diagnostics."""

    structure: Structure
    layers: List[LayerInfo] = field(default_factory=list)
    matches: List[StackMatch] = field(default_factory=list)


@dataclass
class _MatchCandidate:
    match: object
    angle: float
    strain: float
    layer_cell: np.ndarray
    stack_cell: np.ndarray
    deformation: np.ndarray


@dataclass
class _StackStep:
    candidate: _MatchCandidate
    target: np.ndarray
    layer_deformation: np.ndarray
    stack_deformation: np.ndarray


@dataclass
class _SearchState:
    cell: np.ndarray
    steps: List[_StackStep]
    angle_errors: List[float]
    strains: List[float]

    @property
    def score(self) -> Tuple[float, float, float, float, float]:
        return (
            max(self.angle_errors, default=0.0),
            sum(self.angle_errors),
            max(self.strains, default=0.0),
            sum(self.strains),
            float(abs(np.linalg.det(self.cell))),
        )


def _rotation_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a proper row-vector rotation mapping source onto target."""
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-12:
        if dot > 0:
            return np.eye(3)
        axis = np.cross(source, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-12:
            axis = np.cross(source, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = cross / np.linalg.norm(cross)
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    # Rodrigues matrix is for column vectors; transpose for ASE row vectors.
    return (
        np.eye(3) + skew * np.sin(np.arccos(dot))
        + (skew @ skew) * (1.0 - dot)
    ).T


def _angle_distance(a: float, b: float) -> float:
    """Smallest absolute difference between angles in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _polar_angle_2d(source: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """Return rotation angle and maximum principal strain from source to target."""
    from scipy.linalg import polar

    deformation = np.linalg.solve(source, target)
    rotation, stretch = polar(deformation)
    angle = np.degrees(np.arctan2(rotation[0, 1], rotation[0, 0]))
    strain = float(np.max(np.abs(np.linalg.eigvalsh(stretch) - 1.0)))
    return float(angle), strain


def _signed_permutation_matrices() -> List[np.ndarray]:
    """Return 2D basis reorderings that preserve an equivalent lattice."""
    matrices = []
    for permutation in ((0, 1), (1, 0)):
        for signs in itertools.product((-1.0, 1.0), repeat=2):
            matrix = np.zeros((2, 2))
            matrix[0, permutation[0]] = signs[0]
            matrix[1, permutation[1]] = signs[1]
            matrices.append(matrix)
    return matrices


_BASIS_VARIANTS = _signed_permutation_matrices()


def _best_basis_mapping(
    source: np.ndarray,
    target: np.ndarray,
    requested_angle: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Choose equivalent right-handed bases nearest the requested rotation."""
    best = None
    for source_change in _BASIS_VARIANTS:
        source_basis = source_change @ source
        for target_change in _BASIS_VARIANTS:
            target_basis = target_change @ target
            deformation = np.linalg.solve(source_basis, target_basis)
            if np.linalg.det(deformation) <= 0:
                continue
            angle, strain = _polar_angle_2d(source_basis, target_basis)
            score = (_angle_distance(angle, requested_angle), strain)
            if best is None or score < best[0]:
                best = (
                    score,
                    source_basis,
                    target_basis,
                    deformation,
                    angle,
                    strain,
                )
    if best is None:
        raise ValueError("Could not find a right-handed mapping between ZSL bases.")
    return best[1:]


class VdWStackBuilder(Operation):
    """Build a commensurate stack from one or more layered structures.

    Each input may be a bulk layered crystal or an isolated slab. Covalent
    connectivity is used to extract one representative sheet. The sheet is
    rotated so its fitted normal is along z, then ZSL matching chooses the
    commensurate supercell whose twist is closest to the requested angle.
    """

    def __init__(self) -> None:
        self.last_result: Optional[VdWStackResult] = None

    @staticmethod
    def _to_atoms(obj: StructureLike) -> Atoms:
        if isinstance(obj, Atoms):
            return obj.copy()
        if isinstance(obj, Structure):
            return obj.to_ase_atoms()
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            return AseAtomsAdaptor().get_atoms(obj)
        except (TypeError, AttributeError) as exc:
            raise TypeError(
                "Expected ase.Atoms, mckit Structure, or pymatgen Structure; "
                f"got {type(obj).__name__}"
            ) from exc

    @staticmethod
    def _bond_graph(atoms: Atoms, bond_scale: float):
        cutoffs = natural_cutoffs(atoms, mult=bond_scale)
        i_idx, j_idx, shifts = neighbor_list("ijS", atoms, cutoffs)
        adjacency = [[] for _ in atoms]
        bond_vectors = []
        cell = np.asarray(atoms.cell)
        positions = atoms.positions
        for i, j, shift in zip(i_idx, j_idx, shifts):
            if i == j and not np.any(shift):
                continue
            delta = positions[j] + shift @ cell - positions[i]
            adjacency[int(i)].append((int(j), np.asarray(shift), delta))
            bond_vectors.append(delta)
        return adjacency, bond_vectors

    @staticmethod
    def _components(adjacency) -> List[List[int]]:
        seen = set()
        components = []
        for start in range(len(adjacency)):
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            component = []
            while stack:
                atom = stack.pop()
                component.append(atom)
                for neighbor, _, _ in adjacency[atom]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            components.append(sorted(component))
        return sorted(components, key=lambda value: (-len(value), value[0]))

    @staticmethod
    def _unwrap_component(
        atoms: Atoms, component: Sequence[int], adjacency
    ) -> np.ndarray:
        component_set = set(component)
        unwrapped = {component[0]: atoms.positions[component[0]].copy()}
        pending = [component[0]]
        while pending:
            atom = pending.pop()
            for neighbor, _, delta in adjacency[atom]:
                if neighbor not in component_set or neighbor in unwrapped:
                    continue
                unwrapped[neighbor] = unwrapped[atom] + delta
                pending.append(neighbor)
        return np.array([
            unwrapped.get(index, atoms.positions[index]) for index in component
        ])

    @staticmethod
    def _periodic_plane(
        atoms: Atoms, component: Sequence[int], adjacency
    ) -> Tuple[np.ndarray, int]:
        """Determine component dimensionality from periodic graph cycles.

        Bond directions are not suitable for fitting finite-thickness layers:
        bonds in structures such as S-Mo-S point out of the layer mid-plane.
        Periodic graph cycles instead recover the lattice translations under
        which the bonded component repeats.
        """
        component_set = set(component)
        offsets = {component[0]: np.zeros(3, dtype=int)}
        pending = [component[0]]
        cycles = []
        while pending:
            atom = pending.pop()
            for neighbor, shift, _ in adjacency[atom]:
                if neighbor not in component_set:
                    continue
                proposed = offsets[atom] + shift
                if neighbor not in offsets:
                    offsets[neighbor] = proposed
                    pending.append(neighbor)
                else:
                    cycle = proposed - offsets[neighbor]
                    if np.any(cycle):
                        cycles.append(cycle)

        if not cycles:
            raise ValueError(
                "The selected component has no periodic bonded translations."
            )
        translations = np.asarray(cycles, dtype=float) @ np.asarray(atoms.cell)
        _, singular_values, vh = np.linalg.svd(translations, full_matrices=True)
        tolerance = max(translations.shape) * singular_values[0] * 1e-10
        rank = int(np.sum(singular_values > tolerance))
        if rank != 2:
            description = {1: "one-dimensional", 3: "three-dimensional"}.get(
                rank, f"rank-{rank}",
            )
            raise ValueError(
                "The selected covalent component is "
                f"{description}, not a two-dimensional sheet."
            )
        normal = vh[-1]
        return normal / np.linalg.norm(normal), rank

    @staticmethod
    def _in_plane_cell(cell: np.ndarray, normal: np.ndarray) -> np.ndarray:
        projected = cell - np.outer(cell @ normal, normal)
        candidates = []
        for i in range(3):
            for j in range(i + 1, 3):
                area = np.linalg.norm(np.cross(projected[i], projected[j]))
                if area < 1e-8:
                    continue
                out_fraction = (
                    abs(np.dot(cell[i], normal)) / max(np.linalg.norm(cell[i]), 1e-12)
                    + abs(np.dot(cell[j], normal)) / max(np.linalg.norm(cell[j]), 1e-12)
                )
                candidates.append((out_fraction, -area, i, j))
        if not candidates:
            raise ValueError("Could not identify two independent in-plane cell vectors.")
        _, _, i, j = min(candidates)
        return np.array([projected[i], projected[j]])

    def extract_layer(
        self,
        structure: StructureLike,
        *,
        source_index: int = 0,
        component: int = 0,
        bond_scale: float = 1.15,
        layer_vacuum: float = 10.0,
        max_thickness_ratio: Optional[float] = None,
    ) -> Tuple[Atoms, LayerInfo]:
        """Extract and normalize one covalently connected 2D component."""
        atoms = self._to_atoms(structure)
        if len(atoms) == 0 or atoms.cell.rank < 3:
            raise ValueError("Layer extraction requires a non-empty 3D periodic cell.")
        atoms.pbc = True
        adjacency, _ = self._bond_graph(atoms, bond_scale)
        components = self._components(adjacency)
        if component < 0 or component >= len(components):
            raise ValueError(
                f"Component {component} is out of range; found {len(components)}."
        )
        indices = components[component]
        positions = self._unwrap_component(atoms, indices, adjacency)
        normal, _ = self._periodic_plane(atoms, indices, adjacency)
        in_plane = self._in_plane_cell(np.asarray(atoms.cell), normal)
        rotation = _rotation_from_to(normal, np.array([0.0, 0.0, 1.0]))
        vectors = in_plane @ rotation
        vectors[:, 2] = 0.0
        rotated_positions = positions @ rotation
        z = rotated_positions[:, 2]
        thickness = float(np.ptp(z))
        in_plane_length = min(np.linalg.norm(vectors[0]), np.linalg.norm(vectors[1]))
        if (
            max_thickness_ratio is not None
            and thickness > max_thickness_ratio * in_plane_length
        ):
            raise ValueError(
                "The selected sheet exceeds the requested thickness ratio: "
                f"thickness={thickness:.3f} A, in-plane scale={in_plane_length:.3f} A."
            )
        c_length = max(thickness + 2.0 * layer_vacuum, 1.0)
        layer = Atoms(
            numbers=atoms.numbers[indices],
            positions=rotated_positions,
            cell=[vectors[0], vectors[1], [0.0, 0.0, c_length]],
            pbc=True,
        )
        layer.positions[:, 2] -= layer.positions[:, 2].min()
        layer.positions[:, 2] += layer_vacuum
        layer.wrap()
        info = LayerInfo(
            source_index=source_index,
            component_index=component,
            source_atoms=len(atoms),
            layer_atoms=len(layer),
            normal=normal,
            thickness=thickness,
        )
        return layer, info

    @staticmethod
    def _match_layer_candidates(
        stack_vectors: np.ndarray,
        layer_vectors: np.ndarray,
        requested_angle: float,
        max_area: float,
        max_length_tol: float,
        max_angle_tol: float,
        max_strain: float,
        limit: int,
    ) -> List[_MatchCandidate]:
        from pymatgen.analysis.interfaces.zsl import ZSLGenerator

        generator = ZSLGenerator(
            max_area=max_area,
            max_length_tol=max_length_tol,
            max_angle_tol=max_angle_tol,
            bidirectional=True,
        )
        candidates = []
        for match in generator(layer_vectors, stack_vectors):
            layer_sl, stack_sl, deformation, angle, strain = _best_basis_mapping(
                np.asarray(match.film_sl_vectors)[:, :2],
                np.asarray(match.substrate_sl_vectors)[:, :2],
                requested_angle,
            )
            error = _angle_distance(angle, requested_angle)
            if strain > max_strain:
                continue
            score = (error, strain, float(match.match_area))
            candidates.append((
                score,
                _MatchCandidate(
                    match=match,
                    angle=angle,
                    strain=strain,
                    layer_cell=layer_sl,
                    stack_cell=stack_sl,
                    deformation=deformation,
                ),
            ))
        if not candidates:
            return []

        # ZSL can emit several symmetry-equivalent representations of the
        # same solution. Keep distinct cells so the beam is spent on genuinely
        # different paths.
        unique = []
        seen = set()
        for _, candidate in sorted(candidates, key=lambda item: item[0]):
            key = (
                round(candidate.angle, 8),
                tuple(np.round(candidate.stack_cell, 7).ravel()),
                tuple(np.round(candidate.layer_cell, 7).ravel()),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
            if len(unique) >= limit:
                break
        return unique

    @classmethod
    def _match_layer(
        cls,
        stack: Atoms,
        layer: Atoms,
        requested_angle: float,
        max_area: float,
        max_length_tol: float,
        max_angle_tol: float,
        max_strain: float,
    ):
        candidates = cls._match_layer_candidates(
            np.asarray(stack.cell)[:2],
            np.asarray(layer.cell)[:2],
            requested_angle,
            max_area,
            max_length_tol,
            max_angle_tol,
            max_strain,
            limit=1,
        )
        if not candidates:
            raise ValueError(
                "No commensurate lattice match met the area/strain limits. "
                "Increase --max-area or --max-strain."
            )
        candidate = candidates[0]
        return (
            candidate.match,
            candidate.angle,
            candidate.strain,
            candidate.layer_cell,
            candidate.stack_cell,
            candidate.deformation,
        )

    @staticmethod
    def _partition_strain(
        candidate: _MatchCandidate,
        strain_mode: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from scipy.linalg import polar

        rotation, stretch = polar(candidate.deformation)
        if strain_mode == "both":
            eigenvalues, eigenvectors = np.linalg.eigh(stretch)
            stretch_half = (
                eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
            )
            layer_deformation = rotation @ stretch_half
            stack_deformation = np.linalg.inv(stretch_half)
            target = candidate.layer_cell @ layer_deformation
        elif strain_mode == "stack":
            layer_deformation = np.eye(2)
            stack_deformation = np.linalg.inv(candidate.deformation)
            target = candidate.layer_cell
        else:
            layer_deformation = candidate.deformation
            stack_deformation = np.eye(2)
            target = candidate.stack_cell
        return target, layer_deformation, stack_deformation

    @classmethod
    def _plan_stack(
        cls,
        layers: Sequence[Atoms],
        angles: Sequence[float],
        *,
        max_area: float,
        max_length_tol: float,
        max_angle_tol: float,
        max_strain: float,
        strain_mode: str,
        search_width: int,
        matches_per_step: int,
    ) -> _SearchState:
        """Beam-search complete N-layer commensurate-cell paths."""
        states = [_SearchState(
            cell=np.asarray(layers[0].cell)[:2, :2].copy(),
            steps=[],
            angle_errors=[],
            strains=[],
        )]
        for index, layer in enumerate(layers[1:], start=1):
            expanded = []
            layer_vectors = np.asarray(layer.cell)[:2]
            for state in states:
                candidates = cls._match_layer_candidates(
                    state.cell,
                    layer_vectors,
                    angles[index],
                    max_area,
                    max_length_tol,
                    max_angle_tol,
                    max_strain,
                    limit=matches_per_step,
                )
                for candidate in candidates:
                    target, layer_deformation, stack_deformation = (
                        cls._partition_strain(candidate, strain_mode)
                    )
                    expanded.append(_SearchState(
                        cell=target,
                        steps=state.steps + [_StackStep(
                            candidate=candidate,
                            target=target,
                            layer_deformation=layer_deformation,
                            stack_deformation=stack_deformation,
                        )],
                        angle_errors=state.angle_errors + [
                            _angle_distance(candidate.angle, angles[index])
                        ],
                        strains=state.strains + [candidate.strain],
                    ))
            if not expanded:
                raise ValueError(
                    f"No joint commensurate solution remained while adding layer "
                    f"{index + 1}. Increase --max-area, --max-strain, "
                    "or --search-width."
                )

            distinct = []
            seen = set()
            for state in sorted(expanded, key=lambda value: value.score):
                key = tuple(np.round(state.cell, 7).ravel())
                if key in seen:
                    continue
                seen.add(key)
                distinct.append(state)
                if len(distinct) >= search_width:
                    break
            states = distinct
        return min(states, key=lambda value: value.score)

    @staticmethod
    def _integer_transform(transform) -> np.ndarray:
        rounded = np.rint(np.asarray(transform)).astype(int)
        if not np.allclose(transform, rounded, atol=1e-6):
            raise ValueError(f"ZSL returned a non-integer supercell transform: {transform}")
        result = np.eye(3, dtype=int)
        result[:2, :2] = rounded
        return result

    @staticmethod
    def _map_supercell(
        atoms: Atoms,
        source_2d: np.ndarray,
        target_2d: np.ndarray,
        deformation: np.ndarray,
    ) -> Atoms:
        """Express a supercell in a reduced basis, then deform it physically."""
        result = atoms.copy()
        canonical = np.asarray(result.cell).copy()
        canonical[:2] = 0.0
        canonical[:2, :2] = source_2d
        result.set_cell(canonical, scale_atoms=False)
        result.wrap()

        affine = np.eye(3)
        affine[:2, :2] = deformation
        result.positions[:] = result.positions @ affine
        mapped = canonical @ affine
        mapped[:2] = 0.0
        mapped[:2, :2] = target_2d
        result.set_cell(mapped, scale_atoms=False)
        result.wrap()
        return result

    @staticmethod
    def _join(stack: Atoms, layer: Atoms, gap: float, vacuum: float) -> Atoms:
        stack = stack.copy()
        layer = layer.copy()
        stack.positions[:, 2] -= stack.positions[:, 2].min()
        layer.positions[:, 2] -= layer.positions[:, 2].min()
        layer.positions[:, 2] += stack.positions[:, 2].max() + gap
        positions = np.vstack([stack.positions, layer.positions])
        numbers = np.concatenate([stack.numbers, layer.numbers])
        positions[:, 2] += vacuum
        # Keep headroom above the top layer. A site exactly at c wraps onto
        # the bottom layer and destroys the requested interlayer spacing.
        c_length = float(positions[:, 2].max() + max(vacuum, gap, 1.0))
        cell = np.asarray(stack.cell).copy()
        cell[2] = [0.0, 0.0, max(c_length, 1.0)]
        return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)

    def apply(
        self,
        *,
        structures: Sequence[StructureLike],
        angles: Optional[Sequence[float]] = None,
        components: Optional[Sequence[int]] = None,
        gap: float = 3.35,
        vacuum: float = 15.0,
        max_area: float = 400.0,
        max_length_tol: float = 0.03,
        max_angle_tol: float = 0.01,
        max_strain: float = 0.05,
        bond_scale: float = 1.15,
        strain_mode: str = "both",
        search_width: int = 8,
        matches_per_step: int = 8,
    ) -> Structure:
        """Build an N-layer commensurate van der Waals stack.

        ``angles`` are absolute in-plane rotations relative to the first layer.
        Supply either N values (the first should be zero) or N-1 values for
        layers 2..N.
        """
        structures = list(structures)
        if not structures:
            raise ValueError("At least one input structure is required.")
        if gap <= 0 or vacuum < 0:
            raise ValueError("gap must be positive and vacuum must be non-negative.")
        if strain_mode not in {"both", "stack", "layer"}:
            raise ValueError("strain_mode must be 'both', 'stack', or 'layer'.")
        if search_width < 1 or matches_per_step < 1:
            raise ValueError("search_width and matches_per_step must be positive.")

        if angles is None:
            normalized_angles = [0.0] * len(structures)
        else:
            normalized_angles = [float(value) for value in angles]
            if len(normalized_angles) == len(structures) - 1:
                normalized_angles.insert(0, 0.0)
            if len(normalized_angles) != len(structures):
                raise ValueError("Provide either N angles or N-1 angles.")
        if abs(normalized_angles[0]) > 1e-8:
            raise ValueError("The first layer defines zero rotation.")

        selected_components = list(components or [0] * len(structures))
        if len(selected_components) != len(structures):
            raise ValueError("components must contain one index per structure.")

        layers = []
        layer_info = []
        for index, (structure, component) in enumerate(
            zip(structures, selected_components)
        ):
            layer, info = self.extract_layer(
                structure,
                source_index=index,
                component=component,
                bond_scale=bond_scale,
                layer_vacuum=max(vacuum, 5.0),
            )
            layers.append(layer)
            layer_info.append(info)

        plan = self._plan_stack(
            layers,
            normalized_angles,
            max_area=max_area,
            max_length_tol=max_length_tol,
            max_angle_tol=max_angle_tol,
            max_strain=max_strain,
            strain_mode=strain_mode,
            search_width=search_width,
            matches_per_step=matches_per_step,
        )

        stack = layers[0]
        stack.positions[:, 2] -= stack.positions[:, 2].min()
        matches = []
        for index, (layer, step) in enumerate(
            zip(layers[1:], plan.steps), start=1,
        ):
            candidate = step.candidate
            match = candidate.match
            stack_transform = self._integer_transform(match.substrate_transformation)
            layer_transform = self._integer_transform(match.film_transformation)
            stack_super = make_supercell(stack, stack_transform)
            layer_super = make_supercell(layer, layer_transform)

            stack_super = self._map_supercell(
                stack_super,
                candidate.stack_cell,
                step.target,
                step.stack_deformation,
            )
            layer_super = self._map_supercell(
                layer_super,
                candidate.layer_cell,
                step.target,
                step.layer_deformation,
            )
            stack = self._join(stack_super, layer_super, gap, vacuum=0.0)
            matches.append(StackMatch(
                layer_index=index,
                requested_angle=normalized_angles[index],
                actual_angle=candidate.angle,
                angle_error=_angle_distance(
                    candidate.angle, normalized_angles[index],
                ),
                area=float(abs(np.linalg.det(step.target))),
                max_strain=candidate.strain,
                layer_transform=layer_transform[:2, :2],
                stack_transform=stack_transform[:2, :2],
            ))

        stack.positions[:, 2] -= stack.positions[:, 2].min()
        stack.positions[:, 2] += vacuum
        final_cell = np.asarray(stack.cell).copy()
        final_cell[2] = [0.0, 0.0, stack.positions[:, 2].max() + vacuum]
        stack.set_cell(final_cell, scale_atoms=False)
        stack.wrap()
        structure = Structure.from_ase_atoms(stack)
        self.last_result = VdWStackResult(structure, layer_info, matches)
        return structure


def _cmd_build(args) -> None:
    from mckit.io.reader import read_structure
    from mckit.io.writer import write_structure

    structures = [read_structure(path) for path in args.structures]
    builder = VdWStackBuilder()
    result = builder.apply(
        structures=structures,
        angles=args.angles,
        components=args.components,
        gap=args.gap,
        vacuum=args.vacuum,
        max_area=args.max_area,
        max_length_tol=args.max_length_tol,
        max_angle_tol=args.max_angle_tol,
        max_strain=args.max_strain,
        bond_scale=args.bond_scale,
        strain_mode=args.strain_mode,
        search_width=args.search_width,
        matches_per_step=args.matches_per_step,
    )
    output = args.output
    if not output:
        names = "-".join(Path(path).stem for path in args.structures)
        output = f"{names}_vdw_stack.extxyz"
    path = write_structure(output, result)
    print(f"Built vdW stack -> {path}  ({len(result)} atoms)")
    for match in builder.last_result.matches:
        print(
            f"  layer {match.layer_index + 1}: requested={match.requested_angle:.4f} deg "
            f"actual={match.actual_angle:.4f} deg strain={match.max_strain:.5f} "
            f"area={match.area:.2f} A^2"
        )


def register_cli(subparsers) -> None:
    """Register ``mckit operate vdw-stack``."""
    parser = subparsers.add_parser(
        "vdw-stack",
        help="Extract and commensurately stack 2D van der Waals layers",
    )
    parser.add_argument("structures", nargs="+", help="Layered structure files")
    parser.add_argument(
        "--angles", "--angle", type=float, nargs="*", default=None,
        help="Absolute angles for N layers, or angles for layers 2..N (degrees)",
    )
    parser.add_argument(
        "--components", type=int, nargs="*", default=None,
        help="Connected-component index to extract from each input (default: 0)",
    )
    parser.add_argument("--gap", type=float, default=3.35, help="Interlayer gap (A)")
    parser.add_argument("--vacuum", type=float, default=15.0, help="Outer vacuum (A)")
    parser.add_argument(
        "--max-area", type=float, default=400.0,
        help="Maximum commensurate supercell area (A^2)",
    )
    parser.add_argument("--max-length-tol", type=float, default=0.03)
    parser.add_argument("--max-angle-tol", type=float, default=0.01)
    parser.add_argument("--max-strain", type=float, default=0.05)
    parser.add_argument("--bond-scale", type=float, default=1.15)
    parser.add_argument(
        "--strain-mode", choices=["both", "stack", "layer"], default="both",
        help="Which side absorbs mismatch: both, existing stack, or incoming layer",
    )
    parser.add_argument(
        "--search-width", type=int, default=8,
        help="Number of partial multilayer solutions retained during joint search",
    )
    parser.add_argument(
        "--matches-per-step", type=int, default=8,
        help="Commensurate matches explored per layer and partial solution",
    )
    parser.add_argument("--output", "-o")
    parser.set_defaults(handler=_cmd_build)
