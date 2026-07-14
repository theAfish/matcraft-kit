# Coherent interfaces

Find termination combinations and build a coherent film/substrate interface with the Zur-ZSL matching algorithm.

## Commands

```bash
mckit operate interface list FILM SUBSTRATE [options]
mckit operate interface build FILM SUBSTRATE [options]
```

Common options include `--miller-film H K L` (default `1 0 0`), `--miller-substrate H K L` (default `1 1 1`), `--max-area A2`, `--max-length-tol VALUE`, `--max-angle-tol VALUE`, `--no-preserve-molecules`, `--mol-tol A`, and `--mol-min-size N`.

`list` also supports `--termination-ftol ANGSTROM` and `--json FILE`.

`build` supports `--termination VALUE`, independent `--termination-film VALUE` and `--termination-substrate VALUE`, `--gap ANGSTROM` (default `2.5`), `--vacuum ANGSTROM`, `--thickness-film N`, `--thickness-substrate N`, `--angstrom-thickness`, and `-o OUTPUT`.

## Examples

```bash
mckit operate interface list film.cif substrate.cif --miller-film 0 0 1 --miller-substrate 1 1 1
mckit operate interface build film.cif substrate.cif --miller-film 0 0 1 --gap 2.8 --vacuum 15 -o interface.extxyz
```

Inspect both slabs and the final interface before using the result in a simulation.
