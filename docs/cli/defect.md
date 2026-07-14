# Defects

The defect tool has two workflows: symmetry-aware enumeration and scalable population generation.

## Enumerate

```bash
mckit operate defect enumerate INPUT --type vacancy|substitution|antisite|interstitial [options]
```

Without `--index`, this lists symmetry-unique defects. Add `--index N` to create one and optionally `-o OUTPUT`. For substitution use `--substitution Host=Dopant` (or `--sub`); for interstitial use `--species Li Na`. `--symprec` defaults to `0.01`, and `--min-dist` defaults to `0.9 Å`.

## Populate

```bash
mckit operate defect populate INPUT --vacancy WEIGHT [--substitution WEIGHT] [--antisite WEIGHT] [--interstitial WEIGHT] [options]
```

At least one positive weight is required. Use `--sub-map Host=Dopant` when substitution is enabled and `--species ELEMENT [...]` when interstitials are enabled. Other controls are `--min-dist`, `--max-trials`, `--defect-count`, `--defect-density`, `--seed`, `--random-index`, and `-o OUTPUT`.

## Examples

```bash
mckit operate defect enumerate bulk.cif --type vacancy
mckit operate defect enumerate bulk.cif --type substitution --substitution Ga=Zn --index 0 -o doped.extxyz
mckit operate defect populate supercell.extxyz --vacancy 0.7 --interstitial 0.3 --species Li --defect-count 10 -o defects.extxyz
```
