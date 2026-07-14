"""Build polymer oligomer structures from repeat-unit SMILES.

The repeat unit must contain two dummy atoms marking polymerization sites,
preferably ``[*:1]`` and ``[*:2]``. The builder removes those dummy atoms,
connects repeat units into short oligomers, generates RDKit 3-D conformers,
and returns ASE ``Atoms`` objects suitable for writing as PDB, extxyz, or VASP.
"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

from mckit.core.tool import Operation
from mckit.io import write_structure


POLYMER_MODES = (
    "single_chain",
    "rmsd_conformer",
    "multichain_parallel",
    "multichain_crossed_mixed",
    "copolymer_sequence",
    "graft_sidechain",
    "single_ion",
)


@dataclass(frozen=True)
class BuildRecord:
    """One generated polymer structure and its metadata."""

    stem: str
    mode: str
    atoms: Atoms
    names: tuple[str, ...]
    smiles: tuple[str, ...]
    repeats: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectionSite:
    """One atom that can be converted into a polymer connection site."""

    atom_index: int
    symbol: str
    total_hydrogens: int
    degree: int
    in_ring: bool
    aromatic: bool
    neighbor_symbols: tuple[str, ...]


def _safe_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return stem.strip("_.-") or "polymer"


def _dummy_indices(mol: Chem.Mol) -> tuple[int, int]:
    mapped = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 0
    }
    if 1 in mapped and 2 in mapped:
        return mapped[1], mapped[2]

    dummies = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != 2:
        raise ValueError(
            "Repeat-unit SMILES must contain exactly two dummy atoms, "
            "preferably [*:1] and [*:2]."
        )
    return dummies[0], dummies[1]


def _one_dummy_neighbor(mol: Chem.Mol, dummy_idx: int) -> int:
    neighbors = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(dummy_idx).GetNeighbors()]
    if len(neighbors) != 1:
        raise ValueError("Each dummy atom must have exactly one neighboring atom.")
    return neighbors[0]


def _repeat_template(smiles: str) -> tuple[Chem.Mol, int, int, set[int]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse repeat-unit SMILES: {smiles!r}")
    left_dummy, right_dummy = _dummy_indices(mol)
    left_neighbor = _one_dummy_neighbor(mol, left_dummy)
    right_neighbor = _one_dummy_neighbor(mol, right_dummy)
    return mol, left_neighbor, right_neighbor, {left_dummy, right_dummy}


def build_oligomer(smiles: str, repeats: int) -> Chem.Mol:
    """Build one homooligomer RDKit molecule from a repeat-unit SMILES."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")

    repeat, left_neighbor, right_neighbor, dummy_set = _repeat_template(smiles)
    rw = Chem.RWMol()
    left_sites: list[int] = []
    right_sites: list[int] = []

    for _ in range(repeats):
        idx_map: dict[int, int] = {}
        for atom in repeat.GetAtoms():
            old_idx = atom.GetIdx()
            if old_idx in dummy_set:
                continue
            new_atom = Chem.Atom(atom)
            new_atom.SetAtomMapNum(0)
            idx_map[old_idx] = rw.AddAtom(new_atom)

        for bond in repeat.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if begin in dummy_set or end in dummy_set:
                continue
            rw.AddBond(idx_map[begin], idx_map[end], bond.GetBondType())

        left_sites.append(idx_map[left_neighbor])
        right_sites.append(idx_map[right_neighbor])

    for i in range(repeats - 1):
        rw.AddBond(right_sites[i], left_sites[i + 1], Chem.BondType.SINGLE)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def build_sequence_oligomer(smiles_list: Sequence[str], sequence: Sequence[int]) -> Chem.Mol:
    """Build a heterogeneous oligomer from repeat-unit SMILES and a sequence."""
    if not sequence:
        raise ValueError("copolymer_sequence requires at least one sequence index.")
    if not smiles_list:
        raise ValueError("At least one repeat-unit SMILES is required.")

    templates = [_repeat_template(smiles) for smiles in smiles_list]
    rw = Chem.RWMol()
    left_sites: list[int] = []
    right_sites: list[int] = []

    for seq_idx in sequence:
        if seq_idx < 0 or seq_idx >= len(templates):
            raise ValueError(
                f"Sequence index {seq_idx} is out of range for {len(templates)} SMILES."
            )
        repeat, left_neighbor, right_neighbor, dummy_set = templates[seq_idx]
        idx_map: dict[int, int] = {}
        for atom in repeat.GetAtoms():
            old_idx = atom.GetIdx()
            if old_idx in dummy_set:
                continue
            new_atom = Chem.Atom(atom)
            new_atom.SetAtomMapNum(0)
            idx_map[old_idx] = rw.AddAtom(new_atom)

        for bond in repeat.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if begin in dummy_set or end in dummy_set:
                continue
            rw.AddBond(idx_map[begin], idx_map[end], bond.GetBondType())

        left_sites.append(idx_map[left_neighbor])
        right_sites.append(idx_map[right_neighbor])

    for i in range(len(sequence) - 1):
        rw.AddBond(right_sites[i], left_sites[i + 1], Chem.BondType.SINGLE)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _existing_dummy_site_rows(mol: Chem.Mol) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        neighbors = list(atom.GetNeighbors())
        neighbor = neighbors[0] if neighbors else None
        rows.append(
            {
                "dummy_atom_index": atom.GetIdx(),
                "atom_map": atom.GetAtomMapNum(),
                "neighbor_atom_index": neighbor.GetIdx() if neighbor else None,
                "neighbor_symbol": neighbor.GetSymbol() if neighbor else "",
            }
        )
    return rows


def detect_connection_sites(
    smiles: str,
    *,
    max_candidates: int = 50,
) -> dict[str, Any]:
    """Detect plausible polymer connection sites for a SMILES molecule.

    This is a lightweight heuristic. It lists heavy atoms with at least one
    implicit or explicit hydrogen and builds candidate repeat-unit SMILES by
    attaching ``[*:1]`` and ``[*:2]`` to pairs of those atoms.
    """
    if max_candidates < 1:
        raise ValueError("max-candidates must be >= 1.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    existing_dummy_sites = _existing_dummy_site_rows(mol)
    canonical_input = Chem.MolToSmiles(mol, canonical=True)
    sites: list[ConnectionSite] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in (0, 1):
            continue
        hydrogens = int(atom.GetTotalNumHs())
        if hydrogens < 1:
            continue
        sites.append(
            ConnectionSite(
                atom_index=atom.GetIdx(),
                symbol=atom.GetSymbol(),
                total_hydrogens=hydrogens,
                degree=atom.GetDegree(),
                in_ring=atom.IsInRing(),
                aromatic=atom.GetIsAromatic(),
                neighbor_symbols=tuple(nbr.GetSymbol() for nbr in atom.GetNeighbors()),
            )
        )

    pair_candidates: list[dict[str, Any]] = []
    seen_smiles: set[str] = set()
    for left, right in combinations(sites, 2):
        rw = Chem.RWMol(mol)
        for map_num, atom_idx in ((1, left.atom_index), (2, right.atom_index)):
            dummy = Chem.Atom(0)
            dummy.SetAtomMapNum(map_num)
            dummy_idx = rw.AddAtom(dummy)
            rw.AddBond(atom_idx, dummy_idx, Chem.BondType.SINGLE)
        candidate = rw.GetMol()
        try:
            Chem.SanitizeMol(candidate)
        except Exception:
            continue
        candidate_smiles = Chem.MolToSmiles(candidate, canonical=True)
        if candidate_smiles in seen_smiles:
            continue
        seen_smiles.add(candidate_smiles)
        pair_candidates.append(
            {
                "candidate_id": len(pair_candidates) + 1,
                "left_atom_index": left.atom_index,
                "right_atom_index": right.atom_index,
                "left_symbol": left.symbol,
                "right_symbol": right.symbol,
                "candidate_smiles": candidate_smiles,
            }
        )
        if len(pair_candidates) >= max_candidates:
            break

    return {
        "input_smiles": smiles,
        "canonical_smiles": canonical_input,
        "status": "already_marked" if existing_dummy_sites else "detected",
        "existing_dummy_sites": existing_dummy_sites,
        "site_count": len(sites),
        "sites": [
            {
                "atom_index": site.atom_index,
                "symbol": site.symbol,
                "total_hydrogens": site.total_hydrogens,
                "degree": site.degree,
                "in_ring": site.in_ring,
                "aromatic": site.aromatic,
                "neighbor_symbols": list(site.neighbor_symbols),
            }
            for site in sites
        ],
        "pair_candidate_count": len(pair_candidates),
        "pair_candidates": pair_candidates,
        "note": (
            "Candidates are heuristic H-substitution sites. They are useful for "
            "agent/user selection before polymer build, not a reaction-specific "
            "polymerization guarantee."
        ),
    }


def _optimize_conformers(mol: Chem.Mol, cids: Sequence[int]) -> str:
    if not cids:
        return "none"
    if AllChem.MMFFHasAllMoleculeParams(mol):
        try:
            AllChem.MMFFOptimizeMoleculeConfs(
                mol,
                numThreads=0,
                mmffVariant="MMFF94",
            )
            return "MMFF94"
        except Exception:
            pass
    try:
        AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=0)
        return "UFF"
    except Exception:
        return "none"


def embed_conformers(
    mol: Chem.Mol,
    *,
    num_confs: int,
    seed: int,
    random_coords: bool = False,
) -> tuple[Chem.Mol, list[int], str]:
    """Embed and pre-optimize conformers for an RDKit molecule."""
    if num_confs < 1:
        raise ValueError("num_confs must be >= 1.")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 0
    params.useRandomCoords = bool(random_coords)
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params))

    if len(cids) < num_confs:
        params.useRandomCoords = True
        cids = list(
            AllChem.EmbedMultipleConfs(
                mol,
                numConfs=max(num_confs * 4, num_confs + 4),
                params=params,
            )
        )
    if not cids:
        raise RuntimeError("RDKit failed to embed any conformer.")

    cids = cids[:num_confs]
    method = _optimize_conformers(mol, cids)
    return mol, cids, method


def atoms_from_conformer(mol: Chem.Mol, cid: int) -> Atoms:
    """Convert one RDKit conformer to ASE Atoms."""
    conf = mol.GetConformer(int(cid))
    positions = np.asarray(conf.GetPositions(), dtype=float)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)


def rotation_matrix(axis: str, degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    raise ValueError(f"Unknown rotation axis: {axis!r}.")


def transform_atoms(
    atoms: Atoms,
    rotations: Sequence[tuple[str, float]] = (),
    shift: Sequence[float] = (0.0, 0.0, 0.0),
) -> Atoms:
    atoms = atoms.copy()
    atoms.positions -= atoms.positions.mean(axis=0)
    total = np.eye(3)
    for axis, degrees in rotations:
        total = rotation_matrix(axis, degrees) @ total
    atoms.positions = atoms.positions @ total.T
    atoms.positions += np.asarray(shift, dtype=float)
    return atoms


def _min_inter_distance(a: Atoms, b: Atoms) -> float:
    diff = a.positions[:, None, :] - b.positions[None, :, :]
    return float(np.sqrt((diff**2).sum(axis=2)).min())


def combine_chains(chains: Sequence[Atoms], min_distance: float = 1.3) -> Atoms:
    """Combine chains, nudging later chains away from hard contacts."""
    if not chains:
        raise ValueError("At least one chain is required.")

    combined: Atoms | None = None
    for chain in chains:
        trial = chain.copy()
        if combined is not None:
            shift = np.zeros(3)
            attempts = 0
            while _min_inter_distance(combined, trial) < min_distance and attempts < 30:
                shift += np.array([0.0, 1.5, 1.0])
                trial = chain.copy()
                trial.positions += shift
                attempts += 1
        combined = trial if combined is None else combined + trial
    assert combined is not None
    return combined


def stabilize_lithium_positions(atoms: Atoms) -> dict[str, Any]:
    """Move disconnected Li ions near heteroatom anchors without hard contacts."""
    symbols = atoms.get_chemical_symbols()
    li_indices = [i for i, symbol in enumerate(symbols) if symbol == "Li"]
    if not li_indices:
        return li_contact_summary(atoms)

    positions = np.asarray(atoms.positions, dtype=float).copy()
    non_li = [i for i in range(len(atoms)) if i not in li_indices]
    if not non_li:
        return li_contact_summary(atoms)

    anchor_indices = [i for i in non_li if symbols[i] in {"O", "N"}]
    if not anchor_indices:
        anchor_indices = [i for i in non_li if symbols[i] in {"F", "S", "P"}]
    if not anchor_indices:
        anchor_indices = non_li

    def unit(vector: np.ndarray) -> np.ndarray | None:
        norm = float(np.linalg.norm(vector))
        if norm < 1.0e-8 or not np.isfinite(norm):
            return None
        return vector / norm

    base_dirs = [
        np.asarray(v, dtype=float) / np.linalg.norm(v)
        for v in (
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
        )
    ]
    min_allowed = {
        "H": 1.20,
        "C": 2.10,
        "O": 1.45,
        "N": 1.45,
        "F": 1.70,
        "S": 2.05,
        "P": 2.05,
        "Si": 2.05,
    }
    anchor_target = {"O": 1.95, "N": 2.10, "F": 2.00, "S": 2.35, "P": 2.35}
    radii = (1.85, 2.00, 2.20, 2.45, 2.75, 3.10)
    center = positions[non_li].mean(axis=0)

    candidates: list[tuple[float, int, np.ndarray]] = []
    for anchor in anchor_indices:
        anchor_pos = positions[anchor]
        local = [
            j
            for j in non_li
            if j != anchor and np.linalg.norm(positions[j] - anchor_pos) < 3.2
        ]
        dirs = list(base_dirs)
        if local:
            outward = unit(anchor_pos - positions[local].mean(axis=0))
            if outward is not None:
                dirs.insert(0, outward)
        radial = unit(anchor_pos - center)
        if radial is not None:
            dirs.insert(0, radial)

        for direction in dirs:
            for radius in radii:
                pos = anchor_pos + radius * direction
                score = abs(radius - anchor_target.get(symbols[anchor], 2.10))
                for j in non_li:
                    if j == anchor:
                        continue
                    distance = float(np.linalg.norm(pos - positions[j]))
                    allowed = min_allowed.get(symbols[j], 1.75)
                    if distance < allowed:
                        score += 100.0 * (allowed - distance) ** 2
                    elif distance < allowed + 0.35:
                        score += 4.0 * (allowed + 0.35 - distance) ** 2
                score += 0.02 * float(np.linalg.norm(pos - center))
                if symbols[anchor] == "O":
                    score -= 0.25
                elif symbols[anchor] == "N":
                    score -= 0.10
                candidates.append((score, anchor, pos))

    if not candidates:
        return li_contact_summary(atoms)
    candidates.sort(key=lambda item: item[0])

    chosen: list[np.ndarray] = []
    used_anchors: dict[int, int] = {}
    for li_idx in li_indices:
        best: tuple[int, np.ndarray] | None = None
        best_score = float("inf")
        for score, anchor, pos in candidates:
            trial_score = score + 0.45 * used_anchors.get(anchor, 0)
            for chosen_pos in chosen:
                d_li = float(np.linalg.norm(pos - chosen_pos))
                if d_li < 2.40:
                    trial_score += 100.0 * (2.40 - d_li) ** 2
            if trial_score < best_score:
                best = (anchor, pos)
                best_score = trial_score
        if best is None:
            continue
        anchor, pos = best
        positions[li_idx] = pos
        chosen.append(pos)
        used_anchors[anchor] = used_anchors.get(anchor, 0) + 1

    atoms.positions = positions
    return li_contact_summary(atoms)


def li_contact_summary(atoms: Atoms) -> dict[str, Any]:
    thresholds = {
        "H": 1.20,
        "C": 2.10,
        "O": 1.45,
        "N": 1.45,
        "F": 1.70,
        "S": 2.05,
        "P": 2.05,
        "Si": 2.05,
    }
    min_by_symbol: dict[str, float] = {}
    bad = 0
    for i, symbol in enumerate(atoms.symbols):
        if symbol != "Li":
            continue
        for j, other in enumerate(atoms.symbols):
            if i == j:
                continue
            distance = float(np.linalg.norm(atoms.positions[i] - atoms.positions[j]))
            key = f"Li-{other}"
            min_by_symbol[key] = min(min_by_symbol.get(key, float("inf")), distance)
            if distance < thresholds.get(other, 1.75):
                bad += 1
    return {
        "li_count": sum(1 for symbol in atoms.symbols if symbol == "Li"),
        "bad_li_contacts": bad,
        "min_li_c": min_by_symbol.get("Li-C"),
        "min_li_o": min_by_symbol.get("Li-O"),
        "min_li_f": min_by_symbol.get("Li-F"),
        "min_li_s": min_by_symbol.get("Li-S"),
    }


def _with_cell(atoms: Atoms, vacuum: float) -> Atoms:
    atoms = atoms.copy()
    atoms.center(vacuum=vacuum)
    atoms.pbc = False
    return atoms


def _parse_csv_ints(value: str | None) -> tuple[int, ...]:
    if value is None or str(value).strip() == "":
        return ()
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def _parse_formats(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        formats = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    else:
        formats = tuple(str(item).strip().lower() for item in value if str(item).strip())
    valid = {"pdb", "extxyz", "vasp"}
    unknown = sorted(set(formats) - valid)
    if unknown:
        raise ValueError(f"Unsupported output format(s): {', '.join(unknown)}")
    return formats or ("pdb", "extxyz", "vasp")


def _normalize_names(smiles_list: Sequence[str], names: Sequence[str] | None) -> tuple[str, ...]:
    if names:
        if len(names) != len(smiles_list):
            raise ValueError("The number of names must match the number of SMILES.")
        return tuple(_safe_stem(name) for name in names)
    return tuple(f"polymer{i + 1}" for i in range(len(smiles_list)))


def _normalize_repeats(repeats: int | Sequence[int], count: int) -> tuple[int, ...]:
    if isinstance(repeats, int):
        values = (repeats,)
    else:
        values = tuple(int(value) for value in repeats)
    if not values:
        values = (4,)
    if any(value < 1 for value in values):
        raise ValueError("All repeats must be >= 1.")
    if len(values) == 1:
        values = values * count
    elif len(values) != count:
        raise ValueError("repeats must be a single value or match the number of SMILES.")
    return values


def _rmsd_selected_conformer(
    mol: Chem.Mol,
    *,
    pool_size: int,
    seed: int,
) -> tuple[Chem.Mol, int, dict[str, Any]]:
    mol3d, cids, method = embed_conformers(
        mol,
        num_confs=max(2, pool_size),
        seed=seed,
        random_coords=True,
    )
    if AllChem.MMFFHasAllMoleculeParams(mol3d):
        props = AllChem.MMFFGetMoleculeProperties(mol3d)
        energies = []
        for cid in cids:
            ff = AllChem.MMFFGetMoleculeForceField(mol3d, props, confId=cid)
            energies.append((float(ff.CalcEnergy()) if ff else float("inf"), cid))
    else:
        energies = [(0.0, cid) for cid in cids]
    energies.sort()
    ref = energies[0][1]
    selected = max(
        cids,
        key=lambda cid: AllChem.GetConformerRMS(
            mol3d,
            ref,
            cid,
            prealigned=False,
        ),
    )
    rmsd = float(AllChem.GetConformerRMS(mol3d, ref, selected, prealigned=False))
    return mol3d, int(selected), {
        "rdkit_opt": method,
        "rmsd_pool_size": len(cids),
        "selected_cid": int(selected),
        "reference_cid": int(ref),
        "rmsd_to_lowest_energy": rmsd,
    }


class PolymerBuilder(Operation):
    """Build polymer oligomer structures from repeat-unit SMILES."""

    def apply(
        self,
        *,
        smiles: str,
        mode: str = "single_chain",
        repeats: int = 4,
        seed: int = 20260703,
        vacuum: float = 8.0,
    ) -> Atoms:
        """Build one polymer structure and return an ASE Atoms object."""
        records = self.build_many(
            smiles_list=[smiles],
            mode=mode,
            repeats=[repeats],
            confs=1,
            seed=seed,
            vacuum=vacuum,
        )
        return records[0].atoms

    def build_many(
        self,
        *,
        smiles_list: Sequence[str],
        mode: str = "single_chain",
        names: Sequence[str] | None = None,
        repeats: int | Sequence[int] = 4,
        confs: int = 1,
        rmsd_pool: int = 32,
        chain_count: int = 3,
        sequence: Sequence[int] | None = None,
        min_distance: float = 1.3,
        seed: int = 20260703,
        vacuum: float = 8.0,
    ) -> list[BuildRecord]:
        if mode not in POLYMER_MODES:
            raise ValueError(f"Unsupported polymer mode: {mode!r}.")
        if not smiles_list:
            raise ValueError("At least one --smiles value is required.")
        if confs < 1:
            raise ValueError("confs must be >= 1.")
        if chain_count < 1:
            raise ValueError("chain-count must be >= 1.")

        names_tuple = _normalize_names(smiles_list, names)
        repeats_tuple = _normalize_repeats(repeats, len(smiles_list))

        if mode in {"single_chain", "graft_sidechain", "single_ion"}:
            return self._build_single_like(
                smiles_list=smiles_list,
                names=names_tuple,
                repeats=repeats_tuple,
                mode=mode,
                confs=confs,
                seed=seed,
                vacuum=vacuum,
            )

        if mode == "rmsd_conformer":
            return self._build_rmsd(
                smiles_list=smiles_list,
                names=names_tuple,
                repeats=repeats_tuple,
                rmsd_pool=rmsd_pool,
                seed=seed,
                vacuum=vacuum,
            )

        if mode in {"multichain_parallel", "multichain_crossed_mixed"}:
            return self._build_multichain(
                smiles_list=smiles_list,
                names=names_tuple,
                repeats=repeats_tuple,
                mode=mode,
                chain_count=chain_count,
                min_distance=min_distance,
                seed=seed,
                vacuum=vacuum,
            )

        if mode == "copolymer_sequence":
            if sequence is None:
                sequence = tuple(range(len(smiles_list)))
            return self._build_copolymer_sequence(
                smiles_list=smiles_list,
                names=names_tuple,
                sequence=tuple(sequence),
                seed=seed,
                vacuum=vacuum,
            )

        raise AssertionError(f"Unhandled mode: {mode}")

    def _build_single_like(
        self,
        *,
        smiles_list: Sequence[str],
        names: Sequence[str],
        repeats: Sequence[int],
        mode: str,
        confs: int,
        seed: int,
        vacuum: float,
    ) -> list[BuildRecord]:
        records: list[BuildRecord] = []
        for index, (smiles, name, repeat_count) in enumerate(zip(smiles_list, names, repeats)):
            mol = build_oligomer(smiles, repeat_count)
            mol3d, cids, opt_method = embed_conformers(
                mol,
                num_confs=confs,
                seed=seed + index * 1000,
            )
            for conf_index, cid in enumerate(cids):
                atoms = atoms_from_conformer(mol3d, cid)
                li_summary = stabilize_lithium_positions(atoms) if mode == "single_ion" else li_contact_summary(atoms)
                atoms = _with_cell(atoms, vacuum)
                stem = f"{name}_{mode}_r{repeat_count:02d}_conf{conf_index:02d}"
                records.append(
                    BuildRecord(
                        stem=_safe_stem(stem),
                        mode=mode,
                        atoms=atoms,
                        names=(name,),
                        smiles=(smiles,),
                        repeats=(repeat_count,),
                        metadata={
                            "rdkit_opt": opt_method,
                            "conformer_id": int(cid),
                            "li_contact_summary": li_summary,
                        },
                    )
                )
        return records

    def _build_rmsd(
        self,
        *,
        smiles_list: Sequence[str],
        names: Sequence[str],
        repeats: Sequence[int],
        rmsd_pool: int,
        seed: int,
        vacuum: float,
    ) -> list[BuildRecord]:
        records: list[BuildRecord] = []
        for index, (smiles, name, repeat_count) in enumerate(zip(smiles_list, names, repeats)):
            mol = build_oligomer(smiles, repeat_count)
            mol3d, cid, info = _rmsd_selected_conformer(
                mol,
                pool_size=rmsd_pool,
                seed=seed + index * 1000,
            )
            atoms = _with_cell(atoms_from_conformer(mol3d, cid), vacuum)
            stem = f"{name}_rmsd_r{repeat_count:02d}"
            records.append(
                BuildRecord(
                    stem=_safe_stem(stem),
                    mode="rmsd_conformer",
                    atoms=atoms,
                    names=(name,),
                    smiles=(smiles,),
                    repeats=(repeat_count,),
                    metadata=info,
                )
            )
        return records

    def _build_multichain(
        self,
        *,
        smiles_list: Sequence[str],
        names: Sequence[str],
        repeats: Sequence[int],
        mode: str,
        chain_count: int,
        min_distance: float,
        seed: int,
        vacuum: float,
    ) -> list[BuildRecord]:
        rotations_parallel = [[("z", 0.0)], [("z", 8.0)], [("z", -8.0)], [("z", 4.0)]]
        rotations_crossed = [
            [("z", 60.0), ("x", 20.0)],
            [("z", -55.0), ("x", -15.0)],
            [("y", 70.0), ("z", 15.0)],
            [("x", 45.0), ("y", -30.0)],
        ]
        rotations = rotations_parallel if mode == "multichain_parallel" else rotations_crossed
        shifts = []
        for i in range(chain_count):
            if mode == "multichain_parallel":
                shifts.append([(i - (chain_count - 1) / 2.0) * 8.0, 0.0, 0.6 * (i % 2)])
            else:
                angle = 2.0 * np.pi * i / max(chain_count, 1)
                shifts.append([5.0 * np.cos(angle), 5.0 * np.sin(angle), 1.2 * (i % 3)])

        chains: list[Atoms] = []
        chain_names: list[str] = []
        chain_repeats: list[int] = []
        chain_smiles: list[str] = []
        for i in range(chain_count):
            source = i % len(smiles_list)
            mol = build_oligomer(smiles_list[source], repeats[source])
            mol3d, cids, _ = embed_conformers(
                mol,
                num_confs=1,
                seed=seed + i * 97,
            )
            atoms = atoms_from_conformer(mol3d, cids[0])
            atoms = transform_atoms(atoms, rotations[i % len(rotations)], shifts[i])
            chains.append(atoms)
            chain_names.append(names[source])
            chain_repeats.append(repeats[source])
            chain_smiles.append(smiles_list[source])

        combined = combine_chains(chains, min_distance=min_distance)
        li_summary = stabilize_lithium_positions(combined)
        combined = _with_cell(combined, vacuum)
        stem = f"{mode}_{'_'.join(chain_names)}"
        return [
            BuildRecord(
                stem=_safe_stem(stem),
                mode=mode,
                atoms=combined,
                names=tuple(chain_names),
                smiles=tuple(chain_smiles),
                repeats=tuple(chain_repeats),
                metadata={
                    "chain_count": chain_count,
                    "min_distance_requested": min_distance,
                    "li_contact_summary": li_summary,
                },
            )
        ]

    def _build_copolymer_sequence(
        self,
        *,
        smiles_list: Sequence[str],
        names: Sequence[str],
        sequence: Sequence[int],
        seed: int,
        vacuum: float,
    ) -> list[BuildRecord]:
        mol = build_sequence_oligomer(smiles_list, sequence)
        mol3d, cids, opt_method = embed_conformers(mol, num_confs=1, seed=seed)
        atoms = _with_cell(atoms_from_conformer(mol3d, cids[0]), vacuum)
        seq_names = tuple(names[i] for i in sequence)
        stem = f"copolymer_{'_'.join(seq_names)}"
        return [
            BuildRecord(
                stem=_safe_stem(stem),
                mode="copolymer_sequence",
                atoms=atoms,
                names=seq_names,
                smiles=tuple(smiles_list),
                repeats=(len(sequence),),
                metadata={
                    "rdkit_opt": opt_method,
                    "sequence": list(sequence),
                    "sequence_names": list(seq_names),
                    "conformer_id": int(cids[0]),
                },
            )
        ]


def write_records(
    records: Sequence[BuildRecord],
    *,
    out_dir: str | Path,
    formats: str | Sequence[str] = ("extxyz",),
) -> list[dict[str, Any]]:
    """Write build records and return manifest rows."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    selected_formats = _parse_formats(formats)
    rows: list[dict[str, Any]] = []

    for record in records:
        paths: dict[str, str] = {}
        if "pdb" in selected_formats:
            path = out_path / f"{record.stem}.pdb"
            write_structure(str(path), record.atoms)
            paths["pdb"] = path.name
        if "extxyz" in selected_formats:
            path = out_path / f"{record.stem}.extxyz"
            write_structure(str(path), record.atoms)
            paths["extxyz"] = path.name
        if "vasp" in selected_formats:
            path = out_path / f"{record.stem}.vasp"
            write_structure(str(path), record.atoms, format="vasp", direct=False)
            paths["vasp"] = path.name

        row = {
            "stem": record.stem,
            "mode": record.mode,
            "names": ",".join(record.names),
            "smiles": "|".join(record.smiles),
            "repeats": ",".join(str(item) for item in record.repeats),
            "num_atoms": len(record.atoms),
            "formula": record.atoms.get_chemical_formula(),
            "pdb": paths.get("pdb", ""),
            "extxyz": paths.get("extxyz", ""),
            "vasp": paths.get("vasp", ""),
            "metadata_json": json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
        }
        rows.append(row)

    if rows:
        with (out_path / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with (out_path / "manifest.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def _cmd_build(args) -> None:
    t0 = time.time()
    names = []
    if args.names:
        names.extend(item.strip() for item in args.names.split(",") if item.strip())
    if args.name:
        names.extend(args.name)
    names_value = names or None

    repeats = _parse_csv_ints(args.repeats)
    sequence = _parse_csv_ints(args.sequence)
    out_dir = args.out_dir or f"polymer_build_{time.strftime('%Y%m%d_%H%M%S')}"

    builder = PolymerBuilder()
    records = builder.build_many(
        smiles_list=args.smiles,
        mode=args.mode,
        names=names_value,
        repeats=repeats or (4,),
        confs=args.confs,
        rmsd_pool=args.rmsd_pool,
        chain_count=args.chain_count,
        sequence=sequence or None,
        min_distance=args.min_distance,
        seed=args.seed,
        vacuum=args.vacuum,
    )
    write_records(records, out_dir=out_dir, formats=args.formats)
    summary = {
        "status": "ok",
        "mode": args.mode,
        "structures": len(records),
        "output": str(out_dir),
        "manifest": "manifest.csv",
        "elapsed_s": round(time.time() - t0, 3),
    }
    Path(out_dir).joinpath("run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    structure_label = "structure" if len(records) == 1 else "structures"
    print(f"Built {len(records)} polymer {structure_label} -> {out_dir}")
    print(f"Manifest: {Path(out_dir) / 'manifest.csv'}")


def _write_detect_outputs(result: dict[str, Any], out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "detect_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    site_rows = result.get("sites", [])
    if site_rows:
        with (out_path / "connection_sites.csv").open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "atom_index",
                "symbol",
                "total_hydrogens",
                "degree",
                "in_ring",
                "aromatic",
                "neighbor_symbols",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in site_rows:
                writer.writerow(
                    {
                        **row,
                        "neighbor_symbols": ",".join(row.get("neighbor_symbols", [])),
                    }
                )

    pair_rows = result.get("pair_candidates", [])
    if pair_rows:
        with (out_path / "candidate_pair_smiles.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows)


def _cmd_detect(args) -> None:
    result = detect_connection_sites(
        args.smiles,
        max_candidates=args.max_candidates,
    )
    if args.out_dir:
        _write_detect_outputs(result, args.out_dir)
        print(
            f"Detected {result['pair_candidate_count']} candidate connection pairs "
            f"from {args.smiles!r} -> {args.out_dir}"
        )
        print(f"Summary: {Path(args.out_dir) / 'detect_summary.json'}")
    else:
        print(
            f"Detected {result['pair_candidate_count']} candidate connection pairs "
            f"from {args.smiles!r}"
        )


def register_cli(subparsers) -> None:
    """Register polymer builder commands with the mckit CLI."""
    polymer = subparsers.add_parser(
        "polymer",
        help="Build polymer oligomers from repeat-unit SMILES",
    )
    polymer_sub = polymer.add_subparsers(dest="action", required=True)

    build = polymer_sub.add_parser(
        "build",
        help="Generate 3-D polymer structures with RDKit",
    )
    build.add_argument(
        "--smiles",
        action="append",
        required=True,
        help="Repeat-unit SMILES with two dummy atoms. Repeat for mixed systems.",
    )
    build.add_argument(
        "--name",
        action="append",
        help="Name for one SMILES. May be repeated.",
    )
    build.add_argument(
        "--names",
        help="Comma-separated names matching --smiles values.",
    )
    build.add_argument(
        "--mode",
        choices=POLYMER_MODES,
        default="single_chain",
        help="Polymer sampling mode.",
    )
    build.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to polymer_build_<timestamp>.",
    )
    build.add_argument(
        "--repeats",
        default="4",
        help="Repeat count, or comma-separated counts matching --smiles.",
    )
    build.add_argument("--confs", type=int, default=1, help="Conformers per single-chain input.")
    build.add_argument("--rmsd-pool", type=int, default=32, help="Conformer pool size for RMSD mode.")
    build.add_argument("--chain-count", type=int, default=3, help="Number of chains for multichain modes.")
    build.add_argument(
        "--sequence",
        default="",
        help="Comma-separated SMILES indices for copolymer_sequence, e.g. 0,1,0.",
    )
    build.add_argument("--min-distance", type=float, default=1.3, help="Minimum interchain distance target in A.")
    build.add_argument("--seed", type=int, default=20260703, help="Random seed.")
    build.add_argument("--vacuum", type=float, default=8.0, help="Vacuum padding in A.")
    build.add_argument(
        "--formats",
        default="extxyz",
        help="Comma-separated output formats (default: extxyz): pdb,extxyz,vasp.",
    )
    build.set_defaults(handler=_cmd_build)

    detect = polymer_sub.add_parser(
        "detect",
        help="Detect candidate polymer connection sites in a SMILES molecule",
    )
    detect.add_argument(
        "--smiles",
        required=True,
        help="Input SMILES without, or with, polymer dummy connection atoms.",
    )
    detect.add_argument(
        "--max-candidates",
        type=int,
        default=50,
        help="Maximum number of candidate two-site repeat-unit SMILES to return.",
    )
    detect.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory for detect_summary.json and CSV tables.",
    )
    detect.set_defaults(handler=_cmd_detect)
