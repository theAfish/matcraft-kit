import numpy as np
import pytest
from ase import Atoms

from mckit.core import (
    Structure,
    to_ase_atoms,
    to_mckit_structure,
    to_pymatgen_structure,
)
from mckit.operate import BulkBuilder


def test_to_ase_atoms_copies_inputs_by_default():
    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)

    converted = to_ase_atoms(atoms)
    converted.positions[0, 0] = 1.0

    assert atoms.positions[0, 0] == pytest.approx(0.0)


def test_structure_conversions_preserve_data_and_types():
    atoms = Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 5.0,
        pbc=True,
    )

    mckit_structure = to_mckit_structure(atoms)
    pymatgen_structure = to_pymatgen_structure(mckit_structure)
    round_trip = to_mckit_structure(pymatgen_structure)

    assert isinstance(mckit_structure, Structure)
    assert isinstance(round_trip, Structure)
    assert round_trip.composition == {"Na": 1, "Cl": 1}
    assert round_trip.lattice.matrix == pytest.approx(atoms.cell.array)


@pytest.mark.parametrize("conventional", [False, True])
def test_bulk_builder_always_returns_mckit_structure(conventional):
    result = BulkBuilder().apply(
        structure_type="fcc",
        element="Cu",
        a=3.61,
        conventional_unit_cell=conventional,
    )

    assert isinstance(result, Structure)
    assert result.composition == {"Cu": 4 if conventional else 1}
