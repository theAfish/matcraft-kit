from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from mckit.operate.solvation import (
    AVOGADRO_PER_ANGSTROM3,
    SolvationBuilder,
    WINDOWS_DLL_NOT_FOUND,
    _check_packmol_runtime,
    _load_solvent,
)


def test_load_solvent_accepts_lowercase_ase_name():
    water, water_name = _load_solvent("h2o")

    assert water.get_chemical_formula() == "H2O"
    assert water_name == "H2O"


def test_load_solvent_accepts_smiles():
    pytest.importorskip("rdkit")
    propane, propane_name = _load_solvent("CCC")

    assert propane.get_chemical_formula() == "C3H8"
    assert propane_name == "CCC"


def test_concentration_is_converted_to_nearest_molecule_count(monkeypatch):
    system = Atoms("Si", positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
    water = Atoms(
        "OH2",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
    )
    captured = {}

    def fake_pack(**kwargs):
        captured.update(kwargs)
        molecule = kwargs["solvent_atoms"]
        packed = Atoms(
            molecule.get_chemical_symbols() * kwargs["count"],
            positions=np.tile(molecule.positions, (kwargs["count"], 1)),
        )
        packed.cell = kwargs["system_atoms"].cell
        packed.pbc = True
        return packed

    monkeypatch.setattr(SolvationBuilder, "_pack", staticmethod(fake_pack))
    result = SolvationBuilder().apply(
        system=system,
        solvent=water,
        concentration=55.5,
    )

    expected_count = int(np.floor(
        55.5 * system.get_volume() * AVOGADRO_PER_ANGSTROM3 + 0.5
    ))
    assert captured["count"] == expected_count == 33
    assert result.info["solvent_count"] == 33
    assert np.count_nonzero(result.arrays["solvent_mask"]) == 99
    assert np.array_equal(
        np.unique(result.arrays["solvent_id"]),
        np.arange(-1, 33),
    )


def test_exact_count_and_input_immutability(monkeypatch):
    system = Atoms("Na", positions=[[1, 1, 1]], cell=[8, 9, 10], pbc=True)
    original = system.copy()
    solvent = Atoms("He", positions=[[0, 0, 0]])

    def fake_pack(**kwargs):
        packed = Atoms(
            "He2",
            positions=[[2, 2, 2], [4, 4, 4]],
            cell=kwargs["system_atoms"].cell,
            pbc=True,
        )
        return packed

    monkeypatch.setattr(SolvationBuilder, "_pack", staticmethod(fake_pack))
    result = SolvationBuilder().apply(
        system=system,
        solvent=solvent,
        count=2,
    )

    assert len(result) == 3
    assert np.array_equal(system.positions, original.positions)
    assert np.array_equal(system.cell.array, original.cell.array)


def test_pymatgen_packmol_wrapper_builds_and_reads_input(monkeypatch, tmp_path):
    from pymatgen.io.packmol import PackmolSet

    system = Atoms("Na", positions=[[1, 1, 1]], cell=[8, 9, 10], pbc=True)
    solvent = Atoms("He", positions=[[0, 0, 0]])

    def fake_run(input_set, path, timeout=30):
        input_text = (tmp_path / "captured.inp")
        generated = (Path(path) / str(input_set.inputfile)).read_text()
        input_text.write_text(generated)
        packed = Atoms(
            ["Na"] * 27 + ["He"] * 2,
            positions=np.zeros((29, 3)),
        )
        write(Path(path) / str(input_set.outputfile), packed)

    monkeypatch.setattr(
        "mckit.operate.solvation._check_packmol_runtime",
        lambda: None,
    )
    monkeypatch.setattr(PackmolSet, "run", fake_run)
    packed = SolvationBuilder._pack(
        system_atoms=system,
        solvent_atoms=solvent,
        count=2,
        lengths=np.array([8.0, 9.0, 10.0]),
        axes=np.eye(3),
        tolerance=2.0,
        margin=1.0,
        seed=7,
        timeout=30.0,
        solvent_name="He",
    )

    text = (tmp_path / "captured.inp").read_text()
    assert "fixed 0. 0. 0. 0. 0. 0." in text
    assert "inside box 1.0 1.0 1.0 7.0 8.0 9.0" in text
    assert len(packed) == 2
    assert packed.get_chemical_symbols() == ["He", "He"]


def test_rejects_triclinic_cell_before_running_packmol():
    system = Atoms(
        "Si",
        positions=[[0, 0, 0]],
        cell=[[5, 0, 0], [1, 5, 0], [0, 0, 5]],
        pbc=True,
    )
    with pytest.raises(ValueError, match="orthorhombic"):
        SolvationBuilder().apply(
            system=system,
            solvent=Atoms("He"),
            count=1,
        )


def test_requires_exactly_one_amount_mode():
    system = Atoms("Si", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    solvent = Atoms("He")
    builder = SolvationBuilder()

    with pytest.raises(ValueError, match="exactly one"):
        builder.apply(system=system, solvent=solvent)
    with pytest.raises(ValueError, match="exactly one"):
        builder.apply(
            system=system,
            solvent=solvent,
            count=1,
            concentration=1.0,
        )


def test_windows_missing_dll_error_is_actionable(monkeypatch):
    import mckit.operate.solvation as solvation

    class Probe:
        returncode = WINDOWS_DLL_NOT_FOUND

    monkeypatch.setattr(solvation.os, "name", "nt")
    monkeypatch.setattr(
        solvation.subprocess,
        "run",
        lambda *args, **kwargs: Probe(),
    )

    with pytest.raises(
        RuntimeError,
        match="PyPI Windows wheel.*Conda is not required",
    ):
        _check_packmol_runtime()
