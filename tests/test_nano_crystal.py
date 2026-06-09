import numpy as np
import pytest
from ase import Atoms

from mckit.operate import NanoCrystalBuilder
from mckit.cli import build_parser
from mckit.operate.nano_crystal import _cli_polyhedron_args, _polyhedron_geometry


def simple_cubic(a=2.0):
    return Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * a, pbc=True)


def test_sphere_is_finite_centered_cluster():
    result = NanoCrystalBuilder().apply(
        structure=simple_cubic(), shape="sphere", size=4.1, vacuum=5.0,
    )

    assert isinstance(result, Atoms)
    assert len(result) == 7
    assert not np.any(result.pbc)
    assert np.diag(result.cell.array) == pytest.approx([14.0, 14.0, 14.0])


def test_box_uses_full_dimensions_and_fractional_center():
    result = NanoCrystalBuilder().apply(
        structure=simple_cubic(),
        shape="box",
        size=[4.1, 0.5, 0.5],
        center=[0.5, 0.0, 0.0],
        vacuum=0.0,
    )

    assert len(result) == 2
    assert np.ptp(result.positions, axis=0) == pytest.approx([2.0, 0.0, 0.0])


def test_cylinder_axis_is_a_lattice_direction():
    cell = np.diag([2.0, 3.0, 4.0])
    bulk = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
    result = NanoCrystalBuilder().apply(
        structure=bulk,
        shape="cylinder",
        size=[1.0, 8.1],
        axis=[0, 0, 1],
        vacuum=1.0,
    )

    assert len(result) == 3
    assert np.ptp(result.positions[:, 2]) == pytest.approx(8.0)


def test_miller_faceted_cube():
    result = NanoCrystalBuilder().apply(
        structure=simple_cubic(),
        shape="polyhedron",
        miller_indices=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        facet_distances=[2.1, 2.1, 2.1],
        vacuum=0.0,
    )

    assert len(result) == 27
    assert result.get_chemical_symbols() == ["Cu"] * 27


def test_single_cubic_miller_family_expands_to_bounded_polyhedron():
    result = NanoCrystalBuilder().apply(
        structure=simple_cubic(),
        shape="polyhedron",
        miller_indices=[[1, 1, 1]],
        facet_distances=[2.1],
        vacuum=0.0,
    )

    assert len(result) == 7


def test_miller_normals_are_correct_for_skewed_cells():
    cell = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    normals, _, _ = _polyhedron_geometry(
        cell,
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [1.0, 1.0, 1.0],
        1e-8,
    )
    direct_vectors = cell

    assert normals[0] @ direct_vectors[1] == pytest.approx(0.0)
    assert normals[0] @ direct_vectors[2] == pytest.approx(0.0)


def test_does_not_mutate_input_atoms():
    bulk = simple_cubic()
    result = NanoCrystalBuilder().apply(
        structure=bulk, shape="cube", size=0.5, vacuum=2.0,
    )

    assert len(result) == 1
    assert np.all(bulk.pbc)
    assert result.info["nanocrystal_shape"] == "cube"


def test_cli_accepts_explicit_polyhedron_argument_names():
    args = build_parser().parse_args(
        [
            "operate",
            "nanocrystal",
            "bulk.cif",
            "--shape",
            "polyhedron",
            "--miller_indices",
            "1",
            "1",
            "1",
            "--facet_distances",
            "5",
        ]
    )

    assert _cli_polyhedron_args(args) == ([[1, 1, 1]], [5.0])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"shape": "sphere", "size": -1.0}, "greater than zero"),
        ({"shape": "box", "size": [1.0, 2.0]}, "3 values"),
        ({"shape": "cylinder", "size": [2.0, 3.0], "axis": [0, 0, 0]}, "axis"),
        ({"shape": "polyhedron"}, "requires Miller families"),
    ],
)
def test_invalid_shape_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        NanoCrystalBuilder().apply(structure=simple_cubic(), **kwargs)
