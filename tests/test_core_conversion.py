import numpy as np
import pytest
from ase import Atoms

from mckit.core import (
    to_ase_atoms,
    to_pymatgen_structure,
)
from mckit.operate import BulkBuilder


def test_to_ase_atoms_copies_inputs_by_default():
    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)

    converted = to_ase_atoms(atoms)
    converted.positions[0, 0] = 1.0

    assert atoms.positions[0, 0] == pytest.approx(0.0)


def test_pymatgen_round_trip_preserves_data_and_types():
    atoms = Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 5.0,
        pbc=True,
    )

    pymatgen_structure = to_pymatgen_structure(atoms)
    round_trip = to_ase_atoms(pymatgen_structure)

    assert isinstance(round_trip, Atoms)
    assert round_trip.get_chemical_formula() == "ClNa"
    assert round_trip.cell.array == pytest.approx(atoms.cell.array)


@pytest.mark.parametrize("conventional", [False, True])
def test_bulk_builder_always_returns_atoms(conventional):
    result = BulkBuilder().apply(
        structure_type="fcc",
        element="Cu",
        a=3.61,
        conventional_unit_cell=conventional,
    )

    assert isinstance(result, Atoms)
    assert result.get_chemical_symbols() == ["Cu"] * (4 if conventional else 1)


@pytest.mark.parametrize(
    ("kwargs", "expected_formula", "expected_atoms"),
    [
        (
            dict(structure_type="fluorite", element="ZrO2", a=5.12),
            "O8Zr4",
            12,
        ),
        (
            dict(structure_type="fluorite", elements=["Zr", "O", "O"], a=5.12),
            "O8Zr4",
            12,
        ),
        (
            dict(structure_type="fluorite", elements=["zr", "o2"], a=5.12),
            "O8Zr4",
            12,
        ),
    ],
)
def test_bulk_builder_accepts_stoichiometric_inputs(kwargs, expected_formula, expected_atoms):
    result = BulkBuilder().apply(**kwargs)

    assert isinstance(result, Atoms)
    assert len(result) == expected_atoms
    assert result.get_chemical_formula() == expected_formula
