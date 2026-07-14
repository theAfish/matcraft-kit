# Perturbation

Generate one or many randomized structures by displacing atomic positions and optionally the cell vectors.

## Commands

```bash
mckit operate perturb INPUT [options]
mckit operate batch-perturb INPUT [options]
```

Both commands support `--magnitude`/`-m` (default `0.1 Å`), `--mode`, `--indices`, `--cell-magnitude`, `--cell-mode`, `--seed`/`-s`, and `--output`/`-o`. `perturb` writes one structure; `batch-perturb` additionally accepts `--num`/`-n` (default `10`) and writes a numbered series. In batch mode, the seed for each structure is the base seed plus its index.

## Examples

```bash
mckit operate perturb relaxed.extxyz --magnitude 0.05 --seed 10 -o perturbed.extxyz
mckit operate batch-perturb relaxed.extxyz --num 50 --mode random --seed 10 -o training_set
```
