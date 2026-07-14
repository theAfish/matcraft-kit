# Polymers

`mckit operate polymer` has two actions:

- `build` builds short polymer oligomer structures from repeat-unit SMILES.
  Each repeat-unit SMILES must include two polymerization dummy atoms,
  preferably `[*:1]` and `[*:2]`. The command writes an extxyz structure by
  default, plus `manifest.csv`, `manifest.jsonl`, and `run_summary.json`.
  Use `--formats pdb,extxyz,vasp` to request additional structure formats.
- `detect` inspects an ordinary SMILES and proposes possible two-site
  repeat-unit SMILES by replacing H-bearing heavy-atom sites with `[*:1]`
  and `[*:2]`. This is a heuristic helper for agents/users to choose
  polymer connection sites before running `build`.

## Build

```bash
mckit operate polymer build --smiles REPEAT_SMILES [--smiles REPEAT_SMILES ...] [options]
```

Use `--name` repeatedly or `--names name1,name2` for mixed systems. Select a
sampling strategy with `--mode`; repeat counts use `--repeats N` or
comma-separated counts. Other controls include `--confs`, `--rmsd-pool`,
`--chain-count`, `--sequence`, `--min-distance`, `--seed`, `--vacuum`,
`--formats extxyz,pdb,vasp`, and `--out-dir`.

## Detect

```bash
mckit operate polymer detect --smiles SMILES [--max-candidates N] [--out-dir DIR]
```

The detection command does not build a polymer. It proposes candidate
two-site repeat units from an ordinary molecule by replacing H-bearing
heavy-atom sites.

## Examples

```bash
# PEO single chain
mckit operate polymer build --smiles "[*:1]OCC[*:2]" --name peo --mode single_chain --repeats 4 --out-dir smoke_peo

# PMMA RMSD-selected conformer
mckit operate polymer build --smiles "[*:1]CC(C)(C(=O)OC)[*:2]" --name pmma --mode rmsd_conformer --repeats 4 --rmsd-pool 8 --out-dir smoke_pmma_rmsd

# Spatially separated PEO multichain
mckit operate polymer build --smiles "[*:1]OCC[*:2]" --name peo --mode multichain_parallel --chain-count 3 --repeats 3 --out-dir smoke_parallel

# Crossed mixed PAN/PVDF multichain
mckit operate polymer build --smiles "[*:1]CC(C#N)[*:2]" --smiles "[*:1]CC(F)(F)[*:2]" --names pan,pvdf --mode multichain_crossed_mixed --repeats 4,4 --out-dir smoke_mixed

# PVDF-HFP copolymer sequence proxy
mckit operate polymer build --smiles "[*:1]CC(F)(F)[*:2]" --smiles "[*:1]C(F)(C(F)(F)F)C(F)(F)[*:2]" --names vdf,hfp --mode copolymer_sequence --sequence 0,1,0 --out-dir smoke_copolymer

# Oligoether-grafted siloxane proxy
mckit operate polymer build --smiles "[*:1]O[Si](C)(COCCOCCOC)[*:2]" --name pdms_oeo --mode graft_sidechain --repeats 2 --out-dir smoke_graft

# PSTFSI-Li single-ion conductor fragment
mckit operate polymer build --smiles "[*:1]CC([*:2])c1ccc(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)cc1.[Li+]" --name pstfsi_li --mode single_ion --repeats 1 --out-dir smoke_single_ion

# Detect possible polymer connection sites for a normal molecule
mckit operate polymer detect --smiles "CCO" --max-candidates 10 --out-dir detect_ethanol
```