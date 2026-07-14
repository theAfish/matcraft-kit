# Basic geometric checks

Run fundamental geometric validation on a structure.

## Command

```bash
mckit observe basic_check INPUT [--min-dist ANGSTROM] [--density-bounds MIN MAX] [--json] [--verbose]
```

- `--min-dist`: minimum allowed interatomic distance (default `0.5 Å`).
- `--density-bounds MIN MAX`: allowed density range (default `0.01 30.0`).
- `--json`: emit machine-readable results.
- `--verbose`/`-v`: show all warnings.

The command exits with a non-zero status when the check fails, which makes it suitable for CI or workflow scripts.

## Example

```bash
mckit observe basic_check generated.extxyz --verbose
```
