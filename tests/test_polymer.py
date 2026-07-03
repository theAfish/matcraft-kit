import csv
import json

import pytest

from mckit.operate import PolymerBuilder
from mckit.operate.polymer import (
    build_oligomer,
    build_sequence_oligomer,
    detect_connection_sites,
    li_contact_summary,
    _write_detect_outputs,
    write_records,
)


def test_build_oligomer_repeats_increase_atom_count():
    one = build_oligomer("[*:1]OCC[*:2]", 1)
    three = build_oligomer("[*:1]OCC[*:2]", 3)

    assert three.GetNumAtoms() > one.GetNumAtoms()
    assert all(atom.GetAtomicNum() != 0 for atom in three.GetAtoms())


def test_rejects_invalid_smiles_and_missing_dummy_atoms():
    with pytest.raises(ValueError, match="could not parse"):
        build_oligomer("not-a-smiles", 2)

    with pytest.raises(ValueError, match="dummy atoms"):
        build_oligomer("CCO", 2)


def test_single_chain_builder_returns_atoms_with_cell():
    records = PolymerBuilder().build_many(
        smiles_list=["[*:1]OCC[*:2]"],
        names=["peo"],
        repeats=[2],
        mode="single_chain",
        seed=11,
    )

    assert len(records) == 1
    assert records[0].stem == "peo_single_chain_r02_conf00"
    assert len(records[0].atoms) > 0
    assert records[0].atoms.cell.volume > 0


def test_rmsd_conformer_records_selection_metadata():
    records = PolymerBuilder().build_many(
        smiles_list=["[*:1]CC(C)(C(=O)OC)[*:2]"],
        names=["pmma"],
        repeats=[2],
        mode="rmsd_conformer",
        rmsd_pool=4,
        seed=12,
    )

    assert len(records) == 1
    metadata = records[0].metadata
    assert metadata["rmsd_pool_size"] >= 2
    assert metadata["selected_cid"] >= 0
    assert metadata["rmsd_to_lowest_energy"] >= 0.0


def test_copolymer_sequence_uses_requested_order():
    mol = build_sequence_oligomer(
        ["[*:1]CC(F)(F)[*:2]", "[*:1]C(F)(C(F)(F)F)C(F)(F)[*:2]"],
        [0, 1, 0],
    )

    assert mol.GetNumAtoms() > 0
    assert all(atom.GetAtomicNum() != 0 for atom in mol.GetAtoms())


def test_single_ion_mode_moves_li_away_from_hard_contacts():
    records = PolymerBuilder().build_many(
        smiles_list=[
            "[*:1]CC([*:2])c1ccc(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)cc1.[Li+]"
        ],
        names=["pstfsi_li"],
        repeats=[1],
        mode="single_ion",
        seed=13,
    )

    summary = li_contact_summary(records[0].atoms)
    assert summary["li_count"] == 1
    assert summary["bad_li_contacts"] == 0


def test_multichain_parallel_combines_requested_chain_count():
    records = PolymerBuilder().build_many(
        smiles_list=["[*:1]OCC[*:2]"],
        names=["peo"],
        repeats=[2],
        mode="multichain_parallel",
        chain_count=3,
        seed=14,
    )

    assert len(records) == 1
    assert records[0].metadata["chain_count"] == 3
    assert records[0].names == ("peo", "peo", "peo")


def test_write_records_creates_portable_manifest(tmp_path):
    records = PolymerBuilder().build_many(
        smiles_list=["[*:1]OCC[*:2]"],
        names=["peo"],
        repeats=[1],
        mode="single_chain",
        seed=15,
    )

    rows = write_records(records, out_dir=tmp_path, formats="extxyz,vasp")

    assert (tmp_path / "manifest.csv").exists()
    assert (tmp_path / rows[0]["extxyz"]).exists()
    assert (tmp_path / rows[0]["vasp"]).exists()
    with (tmp_path / "manifest.csv").open(newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    assert manifest_rows[0]["extxyz"] == rows[0]["extxyz"]
    assert "\\" not in manifest_rows[0]["extxyz"]
    assert json.loads(manifest_rows[0]["metadata_json"])["conformer_id"] >= 0


def test_detect_connection_sites_returns_candidate_pair_smiles():
    result = detect_connection_sites("CCO", max_candidates=3)

    assert result["status"] == "detected"
    assert result["site_count"] == 3
    assert result["pair_candidate_count"] >= 1
    assert "[*:1]" in result["pair_candidates"][0]["candidate_smiles"]
    assert "[*:2]" in result["pair_candidates"][0]["candidate_smiles"]


def test_detect_reports_existing_dummy_sites():
    result = detect_connection_sites("[*:1]OCC[*:2]")

    assert result["status"] == "already_marked"
    assert len(result["existing_dummy_sites"]) == 2
    assert {site["atom_map"] for site in result["existing_dummy_sites"]} == {1, 2}


def test_write_detect_outputs_creates_json_and_csv(tmp_path):
    result = detect_connection_sites("CCO", max_candidates=2)
    _write_detect_outputs(result, tmp_path)

    assert (tmp_path / "detect_summary.json").exists()
    assert (tmp_path / "connection_sites.csv").exists()
    assert (tmp_path / "candidate_pair_smiles.csv").exists()
    rows = list(csv.DictReader((tmp_path / "candidate_pair_smiles.csv").open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["candidate_smiles"]
