# Surface slabs

Discover possible terminations or build a slab from a selected termination. Molecules are preserved by default when detected.

## Commands

```bash
mckit operate surface list INPUT --miller H K L [options]
mckit operate surface build INPUT --miller H K L [--termination VALUE] [options]
```

Common options:

- `--layers N`: number of atomic layers (default `4`).
- `--vacuum ANGSTROM`: vacuum thickness (default `15`).
- `--margin ANGSTROM`: surface-layer margin (default `1.5`).
- `--no-preserve-molecules`: disable molecular integrity preservation.
- `--mol-tol ANGSTROM`, `--mol-min-size N`, `--mol-extent ANGSTROM`: tune molecule detection and edge decisions.
- `--no-strip-inorganic`: retain inorganic atoms beyond molecular extent.

`list` accepts `--json FILE`. `build` accepts `--termination VALUE` (index, label, or `all`) and `--output FILE`. With `all`, outputs are generated automatically.

## Examples

```bash
mckit operate surface list bulk.cif --miller 1 1 1 --json terminations.json
mckit operate surface build bulk.cif --miller 1 1 1 --layers 6 --termination 0 -o slab_111.extxyz
mckit operate surface build bulk.cif --miller 0 0 1 --termination all
```
