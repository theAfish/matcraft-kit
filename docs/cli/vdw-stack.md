# van der Waals stacking

Extract connected 2-D layers and find commensurate supercells for a multilayer stack.

## Command

```bash
mckit operate vdw-stack STRUCTURE [STRUCTURE ...] [options]
```

Important options are `--angles DEGREES [...]`, `--components INDEX [...]`, `--gap` (default `3.35 Å`), `--vacuum` (default `15 Å`), `--max-area` (default `400 Å²`), `--max-length-tol` (default `0.03`), `--max-angle-tol` (default `0.01`), `--max-strain` (default `0.05`), `--bond-scale` (default `1.15`), and `--strain-mode both|stack|layer`. Joint-search controls are `--search-width` and `--matches-per-step`. Use `-o OUTPUT` for the final stack.

## Example

```bash
mckit operate vdw-stack graphene.extxyz hbn.extxyz --angles 0 30 --gap 3.35 --vacuum 15 -o heterostructure.extxyz
```
