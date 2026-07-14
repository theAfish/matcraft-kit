# Nanocrystals

Cut a finite nanocrystal from a periodic bulk structure.

## Command

```bash
mckit operate nanocrystal INPUT --shape sphere|cube|box|ellipsoid|cylinder|polyhedron [options]
```

Use `--size DIMENSION [...]` for non-polyhedral full dimensions, `--center F1 F2 F3` for the fractional cut center, and `--axis U V W` for cylinders. Set `--vacuum` (default `10 Å`) and `-o OUTPUT` as needed.

For polyhedra, either repeat `--facet H K L DIST` or pair repeated `--miller-indices H K L` with `--facet-distances DIST`. `DIST` is the center-to-facet distance in Å.

## Examples

```bash
mckit operate nanocrystal bulk.cif --shape sphere --size 20 -o sphere.extxyz
mckit operate nanocrystal bulk.cif --shape polyhedron --facet 1 1 1 10 --facet 1 0 0 12 -o truncated.extxyz
```
