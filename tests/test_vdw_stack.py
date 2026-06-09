import numpy as np
import pytest
from ase import Atoms

from mckit.core.structure import Structure
from mckit.operate import VdWStackBuilder
from mckit.operate.vdw_stack import _MatchCandidate


def graphite():
    a = 2.46
    c = 6.70
    return Atoms(
        "C4",
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [1 / 3, 2 / 3, 0.0],
            [0.0, 0.0, 0.5],
            [2 / 3, 1 / 3, 0.5],
        ],
        cell=[
            [a, 0.0, 0.0],
            [-a / 2, np.sqrt(3) * a / 2, 0.0],
            [0.0, 0.0, c],
        ],
        pbc=True,
    )


def molybdenum_disulfide():
    a = 3.18430383
    c = 19.44629514
    return Atoms(
        symbols=["Mo", "Mo", "Mo", "S", "S", "S", "S", "S", "S"],
        scaled_positions=[
            [2 / 3, 1 / 3, 0.3331083733],
            [1 / 3, 2 / 3, 0.6664417067],
            [0.0, 0.0, 0.99977504],
            [1 / 3, 2 / 3, 0.0795968867],
            [0.0, 0.0, 0.25329574],
            [0.0, 0.0, 0.41293022],
            [2 / 3, 1 / 3, 0.5866290733],
            [2 / 3, 1 / 3, 0.7462635533],
            [1 / 3, 2 / 3, 0.9199624067],
        ],
        cell=[
            [a / 2, -np.sqrt(3) * a / 2, 0.0],
            [a / 2, np.sqrt(3) * a / 2, 0.0],
            [0.0, 0.0, c],
        ],
        pbc=True,
    )


def test_extracts_one_graphene_sheet():
    layer, info = VdWStackBuilder().extract_layer(graphite())

    assert len(layer) == 2
    assert info.source_atoms == 4
    assert info.layer_atoms == 2
    assert info.thickness == pytest.approx(0.0, abs=1e-8)
    assert np.cross(layer.cell[0], layer.cell[1])[2] > 0


def test_extracts_finite_thickness_mos2_sheet():
    layer, info = VdWStackBuilder().extract_layer(molybdenum_disulfide())

    assert len(layer) == 3
    assert info.thickness == pytest.approx(3.1042992)
    assert abs(info.normal[2]) == pytest.approx(1.0)


def test_builds_commensurate_graphene_bilayer_with_requested_gap():
    builder = VdWStackBuilder()
    result = builder.apply(
        structures=[graphite(), graphite()],
        angles=[0.0],
        gap=3.35,
        vacuum=12.0,
        max_area=50.0,
    )

    assert isinstance(result, Structure)
    assert len(result) == 4
    z_planes = np.unique(np.round(result.cart_positions[:, 2], decimals=6))
    assert len(z_planes) == 2
    assert z_planes[1] - z_planes[0] == pytest.approx(3.35)
    assert builder.last_result.matches[0].actual_angle == pytest.approx(0.0)
    assert builder.last_result.matches[0].max_strain == pytest.approx(0.0)


def test_angle_count_is_validated():
    with pytest.raises(ValueError, match="N angles or N-1"):
        VdWStackBuilder().apply(
            structures=[graphite(), graphite(), graphite()],
            angles=[0.0],
        )


def test_selects_nearby_commensurate_twist():
    builder = VdWStackBuilder()
    builder.apply(
        structures=[graphite(), graphite()],
        angles=[21.8],
        max_area=150.0,
    )

    match = builder.last_result.matches[0]
    assert match.actual_angle == pytest.approx(21.7867893)
    assert match.angle_error < 0.02


def test_thirty_degree_match_does_not_collapse_cell():
    builder = VdWStackBuilder()
    result = builder.apply(
        structures=[graphite(), graphite()],
        angles=[30.0],
        max_area=400.0,
    )

    area = np.linalg.norm(np.cross(
        result.lattice.matrix[0], result.lattice.matrix[1],
    ))
    distances = result.atoms.get_all_distances(mic=True)
    minimum_distance = distances[np.triu_indices(len(result), 1)].min()

    assert builder.last_result.matches[0].actual_angle == pytest.approx(30.0)
    assert area > 250.0
    assert minimum_distance > 1.0


def test_joint_search_can_reconsider_the_first_pair(monkeypatch):
    def candidate(cell, angle):
        return _MatchCandidate(
            match=object(),
            angle=angle,
            strain=0.0,
            layer_cell=cell,
            stack_cell=cell,
            deformation=np.eye(2),
        )

    def fake_candidates(
        stack_vectors,
        layer_vectors,
        requested_angle,
        max_area,
        max_length_tol,
        max_angle_tol,
        max_strain,
        limit,
    ):
        area = abs(np.linalg.det(stack_vectors))
        if requested_angle == 10.0:
            values = [
                candidate(np.diag([2.0, 2.0]), 10.0),
                candidate(np.diag([3.0, 3.0]), 11.0),
            ]
        elif area < 5.0:
            values = [candidate(np.diag([2.0, 2.0]), 50.0)]
        else:
            values = [candidate(np.diag([3.0, 3.0]), 20.0)]
        return values[:limit]

    monkeypatch.setattr(
        VdWStackBuilder,
        "_match_layer_candidates",
        staticmethod(fake_candidates),
    )
    layers = [
        Atoms("H", cell=np.diag([1.0, 1.0, 10.0]), pbc=True)
        for _ in range(3)
    ]
    common = dict(
        max_area=100.0,
        max_length_tol=0.03,
        max_angle_tol=0.01,
        max_strain=0.05,
        strain_mode="both",
        matches_per_step=2,
    )
    greedy = VdWStackBuilder._plan_stack(
        layers, [0.0, 10.0, 20.0], search_width=1, **common,
    )
    joint = VdWStackBuilder._plan_stack(
        layers, [0.0, 10.0, 20.0], search_width=2, **common,
    )

    assert greedy.score[0] == pytest.approx(30.0)
    assert joint.score[0] == pytest.approx(1.0)
    assert joint.steps[0].candidate.angle == pytest.approx(11.0)
