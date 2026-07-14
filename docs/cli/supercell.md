# Supercells

Replicate a periodic structure using either diagonal repeat counts or a full 3×3 integer-like matrix.

## Command

```bash
mckit operate supercell INPUT (--repeat N1 N2 N3 | --matrix M11 M12 M13 M21 M22 M23 M31 M32 M33) [-o OUTPUT]
```

`--repeat` and `--matrix` are mutually exclusive. The matrix is row-major. Output defaults to `supercell_<input>.extxyz`.

## Examples

```bash
mckit operate supercell primitive.cif --repeat 2 2 2 -o conventional.extxyz
mckit operate supercell layer.extxyz --matrix 2 0 0 0 2 0 0 0 1 -o layer_2x2.extxyz
```
