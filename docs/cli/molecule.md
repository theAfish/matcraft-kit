# Molecules

Build isolated molecules from ASE's built-in database or from a SMILES string with RDKit.

## Commands

```bash
mckit operate molecule from_ase NAME [-o OUTPUT]
mckit operate molecule from_smiles SMILES [--vacuum ANGSTROM] [--no-optimize] [-o OUTPUT]
```

`from_ase` accepts names such as `H2O`, `CO2`, `C6H6`, and `C60`; output defaults to `mol_<name>.extxyz`. `from_smiles` uses `5 Å` vacuum and MMFF optimisation by default; output defaults to `mol_<smiles>.extxyz`.

## Examples

```bash
mckit operate molecule from_ase H2O -o water.extxyz
mckit operate molecule from_smiles CCO --vacuum 8 -o ethanol.extxyz
```
