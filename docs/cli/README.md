# `mckit` command reference

This directory is a quick reference for every command currently registered by the `mckit` CLI. Examples use the default output formats and can be copied directly into a terminal.

## Operations

- [Bulk crystals](bulk.md) — `mckit operate bulk build`
- [Supercells](supercell.md) — `mckit operate supercell`
- [Surfaces](surface.md) — `mckit operate surface`
- [Interfaces](interface.md) — `mckit operate interface`
- [Adsorption](adsorption.md) — `mckit operate adsorption`
- [Defects](defect.md) — `mckit operate defect`
- [Molecules](molecule.md) — `mckit operate molecule`
- [Nanocrystals](nanocrystal.md) — `mckit operate nanocrystal`
- [Solvation](solvation.md) — `mckit operate solvation`
- [Perturbation](perturbation.md) — `mckit operate perturb` and `batch-perturb`
- [van der Waals stacking](vdw-stack.md) — `mckit operate vdw-stack`
- [Polymers](polymer.md) — `mckit operate polymer`

## Observations

- [Structure inspection](inspect.md) — `mckit observe inspect`
- [Basic checks](basic-check.md) — `mckit observe basic_check`

## Recommended workflow

1. Start with a structure file supported by ASE, such as CIF, POSCAR, or extxyz.
2. Run an operation and save the result with `--output`/`-o` where available.
3. Check every generated structure:

```bash
mckit observe inspect result.extxyz
mckit observe basic_check result.extxyz --verbose
```

The authoritative parser help remains available with `mckit <category> <tool> -h`; this reference is intended to make common workflows easier to discover without repeatedly opening help output.
