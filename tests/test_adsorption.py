import numpy as np
import pytest
from ase import Atoms
from ase.geometry import find_mic

from mckit.operate import AdsorptionBuilder


def slab():
    return Atoms(
        "Cu4",
        scaled_positions=[
            [0.0, 0.0, 0.25],
            [0.5, 0.0, 0.25],
            [0.0, 0.5, 0.25],
            [0.5, 0.5, 0.25],
        ],
        cell=[[4.0, 0.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 12.0]],
        pbc=True,
    )


def test_places_anchor_at_fractional_site_and_height():
    adsorbate = Atoms("OH", positions=[[0, 0, 0], [0, 0, 1]])
    result = AdsorptionBuilder().apply(
        slab=slab(),
        adsorbate=adsorbate,
        site=(0.25, 0.5),
        height=2.0,
    )

    expected_xy = 0.25 * result.cell[0] + 0.5 * result.cell[1]
    assert result.positions[4, :2] == pytest.approx(expected_xy[:2])
    assert result.positions[4, 2] == pytest.approx(5.0)
    assert result.arrays["adsorbate_mask"].tolist() == [
        False, False, False, False, True, True,
    ]


def test_orients_anchor_vector_along_bottom_normal():
    adsorbate = Atoms("CO", positions=[[0, 0, 0], [1, 0, 0]])
    result = AdsorptionBuilder().apply(
        slab=slab(),
        adsorbate=adsorbate,
        side="bottom",
        height=2.0,
        orientation_atom=1,
    )

    direction = result.positions[5] - result.positions[4]
    assert direction == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)
    assert result.positions[4, 2] == pytest.approx(1.0)


def test_azimuth_rotates_around_surface_normal():
    adsorbate = Atoms("HH", positions=[[0, 0, 0], [1, 0, 0]])
    result = AdsorptionBuilder().apply(
        slab=slab(),
        adsorbate=adsorbate,
        azimuth=90.0,
    )

    direction = result.positions[5] - result.positions[4]
    assert direction == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)


def test_rejects_adsorbate_slab_collision():
    with pytest.raises(ValueError, match="overlaps the slab"):
        AdsorptionBuilder().apply(
            slab=slab(),
            adsorbate=Atoms("H", positions=[[0, 0, 0]]),
            site=(0.0, 0.0),
            height=0.1,
        )


def test_rejects_adsorbate_larger_than_surface_cell():
    adsorbate = Atoms("HH", positions=[[0, 0, 0], [4.5, 0, 0]])

    with pytest.raises(ValueError, match="too large for the slab surface cell"):
        AdsorptionBuilder().apply(
            slab=slab(),
            adsorbate=adsorbate,
            height=2.0,
        )


def test_rejects_adsorbate_periodic_image_collision():
    adsorbate = Atoms("HH", positions=[[0, 0, 0], [3.6, 0, 0]])

    with pytest.raises(ValueError, match="periodic-image atom"):
        AdsorptionBuilder().apply(
            slab=slab(),
            adsorbate=adsorbate,
            height=2.0,
            min_distance=0.5,
        )


def test_periodic_image_check_can_be_disabled():
    adsorbate = Atoms("HH", positions=[[0, 0, 0], [4.5, 0, 0]])
    result = AdsorptionBuilder().apply(
        slab=slab(),
        adsorbate=adsorbate,
        height=2.0,
        check_periodic_images=False,
    )

    assert len(result) == 6


def test_density_mode_builds_supercell_and_places_requested_count():
    result = AdsorptionBuilder().apply_density(
        slab=slab(),
        adsorbate=Atoms("H", positions=[[0, 0, 0]]),
        density=10.0,
        count=4,
        height=2.0,
        min_distance=1.0,
        seed=7,
    )

    assert result.info["adsorbate_count"] == 4
    assert result.info["adsorbate_density_per_nm2"] <= 10.0
    assert np.prod(result.info["slab_repeat"][:2]) >= 3
    assert result.arrays["adsorbate_mask"].sum() == 4
    assert sorted(np.unique(result.arrays["adsorbate_id"])) == [-1, 0, 1, 2, 3]


def test_density_metadata_explains_base_slab_density_cap():
    result = AdsorptionBuilder().apply_density(
        slab=slab(),
        adsorbate=Atoms("H", positions=[[0, 0, 0]]),
        density=50.0,
        count=1,
        seed=3,
    )

    assert result.info["slab_repeat"] == (1, 1, 1)
    assert result.info["adsorbate_density_per_nm2"] == pytest.approx(6.25)
    assert result.info["density_selection_limit"] == "base_slab"


def test_density_metadata_explains_integer_repeat_quantization():
    result = AdsorptionBuilder().apply_density(
        slab=slab(),
        adsorbate=Atoms("H", positions=[[0, 0, 0]]),
        density=5.0,
        count=1,
        seed=3,
    )

    assert np.prod(result.info["slab_repeat"][:2]) == 2
    assert result.info["adsorbate_density_per_nm2"] == pytest.approx(3.125)
    assert result.info["density_selection_limit"] == "integer_repeat"
    assert result.info["required_surface_area_ang2"] == pytest.approx(20.0)
    assert result.info["selected_surface_area_ang2"] == pytest.approx(32.0)


def test_density_mode_is_reproducible_and_avoids_mutual_overlap():
    kwargs = dict(
        slab=slab(),
        adsorbate=Atoms("HH", positions=[[0, 0, 0], [0, 0, 0.75]]),
        density=5.0,
        count=3,
        height=2.0,
        min_distance=1.2,
        seed=11,
    )
    first = AdsorptionBuilder().apply_density(**kwargs)
    second = AdsorptionBuilder().apply_density(**kwargs)

    assert first.positions == pytest.approx(second.positions)
    adsorbate_positions = first.positions[first.arrays["adsorbate_mask"]]
    adsorbate_ids = first.arrays["adsorbate_id"][first.arrays["adsorbate_mask"]]
    deltas = (
        adsorbate_positions[:, None, :] - adsorbate_positions[None, :, :]
    )
    _, distances = find_mic(
        deltas.reshape(-1, 3),
        first.cell,
        pbc=(True, True, False),
    )
    distances = distances.reshape(len(adsorbate_positions), -1)
    different_molecules = adsorbate_ids[:, None] != adsorbate_ids[None, :]
    assert np.all(distances[different_molecules] >= 1.2)


def test_density_two_molecules_are_dispersed_without_overlap():
    result = AdsorptionBuilder().apply_density(
        slab=slab(),
        adsorbate=Atoms("HH", positions=[[0, 0, 0], [0, 0, 0.75]]),
        density=5.0,
        count=2,
        seed=1,
    )

    assert result.info["minimum_interadsorbate_distance_ang"] >= 2.0
    assert result.info["density_candidates_tried"] == 1


def test_density_fifteen_molecules_uses_balanced_fast_path():
    result = AdsorptionBuilder().apply_density(
        slab=slab(),
        adsorbate=Atoms("HH", positions=[[0, 0, 0], [0, 0, 0.75]]),
        density=5.0,
        count=15,
        seed=1,
        attempts_per_molecule=1,
    )

    assert sorted(result.info["slab_repeat"][:2]) == [4, 5]
    assert result.info["density_candidates_tried"] == 1
    assert result.info["minimum_interadsorbate_distance_ang"] >= 2.0
    assert len(np.unique(result.arrays["adsorbate_id"])) == 16


def test_density_mode_requires_feasible_repeat_limit():
    with pytest.raises(ValueError, match="larger than max_repeat"):
        AdsorptionBuilder().apply_density(
            slab=slab(),
            adsorbate=Atoms("H", positions=[[0, 0, 0]]),
            density=1.0,
            count=100,
            max_repeat=2,
        )


def test_input_structures_are_not_modified():
    surface = slab()
    adsorbate = Atoms("H", positions=[[0, 0, 0]])
    original_surface = surface.positions.copy()
    original_adsorbate = adsorbate.positions.copy()

    AdsorptionBuilder().apply(slab=surface, adsorbate=adsorbate)

    assert np.array_equal(surface.positions, original_surface)
    assert np.array_equal(adsorbate.positions, original_adsorbate)
