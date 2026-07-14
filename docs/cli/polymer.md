# Polymers

Build 3-D oligomers from repeat-unit SMILES or detect candidate connection sites in an ordinary molecule. Repeat units should contain two dummy atoms, preferably `[*:1]` and `[*:2]`.

## Build

```bash
mckit operate polymer build --smiles REPEAT_SMILES [--smiles REPEAT_SMILES ...] [options]
```

Use `--name` repeatedly or `--names name1,name2` for mixed systems. Modes are selected with `--mode`; repeat counts use `--repeats N` or comma-separated counts. Other controls include `--confs`, `--rmsd-pool`, `--chain-count`, `--sequence`, `--min-distance`, `--seed`, `--vacuum`, `--formats extxyz,pdb,vasp`, and `--out-dir`. Output includes structures plus manifest and run-summary files.

## Detect

```bash
mckit operate polymer detect --smiles SMILES [--max-candidates N] [--out-dir DIR]
```

This heuristic proposes two-site repeat units by replacing H-bearing heavy-atom sites. It does not build a polymer.

## Examples

```bash
mckit operate polymer build --smiles "[*:1]OCC[*:2]" --name peo --repeats 4 --out-dir peo_build
mckit operate polymer build --smiles "[*:1]CC(C#N)[*:2]" --smiles "[*:1]CC(F)(F)[*:2]" --names pan,pvdf --mode copolymer_sequence --sequence 0,1,0 --out-dir copolymer
mckit operate polymer detect --smiles CCO --max-candidates 10 --out-dir ethanol_sites
```
