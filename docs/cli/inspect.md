# Structure inspection

Print a structure summary, including basic structure data and detected structural features.

## Command

```bash
mckit observe inspect INPUT [--json]
```

The default output is human-readable. `--json` emits machine-readable information suitable for scripts.

## Examples

```bash
mckit observe inspect slab.extxyz
mckit observe inspect slab.extxyz --json > slab-info.json
```
