# Bulk crystals

Build a periodic bulk crystal with ASE's crystal builders.

## Command

```bash
mckit operate bulk build --type TYPE (--element ELEMENT | --elements ELEMENT [ELEMENT ...]) --a ANGSTROM [--c ANGSTROM] [-o OUTPUT]
```

- `--type`: supported crystal structure type. See `mckit operate bulk build -h` for the current choices.
- `--element`: one element or formula, for example `Cu` or `ZrO2`.
- `--elements`: separate element symbols, for example `Ga As` or `Zr O O`.
- `--a`: required lattice parameter in Å.
- `--c`: optional second lattice parameter, used by hcp structures.
- `-o`, `--output`: output path; defaults to `bulk_<type>.extxyz`.

## Examples

```bash
mckit operate bulk build --type fcc --element Cu --a 3.61 -o cu.extxyz
mckit operate bulk build --type rocksalt --elements Na Cl --a 5.63 -o nacl.extxyz
```
