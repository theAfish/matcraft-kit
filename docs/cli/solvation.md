# Solvation

Fill an orthorhombic system cell with solvent using Packmol.

## Command

```bash
mckit operate solvation SYSTEM SOLVENT (--concentration MOL_PER_L | --count N) [options]
```

`SOLVENT` can be a structure file, ASE molecule name, or SMILES string. Select interpretation with `--source auto|file|name|smiles`. `--tolerance` defaults to `2.0 Å`; `--seed` to `1`; `--timeout` to `120 s`. `--boundary-margin` keeps solvent away from cell faces. Use `-o OUTPUT` to choose the result path.

## Examples

```bash
mckit operate solvation cell.extxyz water --count 100 --source name -o hydrated.extxyz
mckit operate solvation cell.extxyz "O" --concentration 2.0 --source smiles --seed 4 -o aqueous.extxyz
```
