# Adsorption

Place one adsorbate structure on a surface slab.

## Command

```bash
mckit operate adsorption SLAB ADSORBATE [options]
```

- `--site U V`: fractional in-plane site (default `0.5 0.5`).
- `--height ANGSTROM`: anchor height above the outermost slab atom (default `2.0`).
- `--anchor INDEX`: zero-based adsorbate anchor atom (default `0`).
- `--orientation-atom INDEX`: atom used to define outward orientation.
- `--azimuth DEGREES`: rotation around the surface normal.
- `--side top|bottom`: slab side (default `top`).
- `--min-distance ANGSTROM`: minimum adsorbate/slab distance (default `0.7`).
- `--adsorbate-min-distance ANGSTROM`: separation between adsorbates in density mode (default `2.0`).
- `--density MOLECULES_PER_NM2`, `--count N`, `--seed N`, `--fixed-azimuth`: density-mode placement controls.
- `--covalent-scale VALUE`, `--allow-periodic-overlap`, `--max-repeat N`: collision and periodicity controls.
- `-o`, `--output`: output structure path.

## Examples

```bash
mckit operate adsorption slab.extxyz co.extxyz --site 0.25 0.50 --height 2.2 -o co_on_slab.extxyz
mckit operate adsorption slab.extxyz water.extxyz --density 2.0 --count 20 --seed 7 -o hydrated.extxyz
```
