# mckit — MatCraft Toolkit

A modular Python framework for building and analyzing atomic structures. Backed by [ASE](https://wiki.fysik.dtu.dk/ase/) and [pymatgen](https://pymatgen.org/) for materials modelling.

## Features

- **Operations** — modelling materials
- **Observations** — inspect structures and run sanity checks

---

## 1. Usage

### Installation

You can simply use:
```
pip install matcraft-kit
```

For development, clone the repository and install in editable mode:

```bash
pip install -e ".[dev]"
```


### CLI

```bash
# for observations:
mckit observe -h
# for operations:
mckit operate -h
```

#### Polymer builder

`mckit operate polymer` has two actions:

- `build` builds short polymer oligomer structures from repeat-unit SMILES.
  Each repeat-unit SMILES must include two polymerization dummy atoms,
  preferably `[*:1]` and `[*:2]`. The command writes an extxyz structure by
  default, plus `manifest.csv`, `manifest.jsonl`, and `run_summary.json`.
  Use `--formats pdb,extxyz,vasp` to request additional structure formats.
- `detect` inspects an ordinary SMILES and proposes possible two-site
  repeat-unit SMILES by replacing H-bearing heavy-atom sites with `[*:1]`
  and `[*:2]`. This is a heuristic helper for agents/users to choose
  polymer connection sites before running `build`.

```bash
# PEO single chain
mckit operate polymer build --smiles "[*:1]OCC[*:2]" --name peo --mode single_chain --repeats 4 --out-dir smoke_peo

# PMMA RMSD-selected conformer
mckit operate polymer build --smiles "[*:1]CC(C)(C(=O)OC)[*:2]" --name pmma --mode rmsd_conformer --repeats 4 --rmsd-pool 8 --out-dir smoke_pmma_rmsd

# Spatially separated PEO multichain
mckit operate polymer build --smiles "[*:1]OCC[*:2]" --name peo --mode multichain_parallel --chain-count 3 --repeats 3 --out-dir smoke_parallel

# Crossed mixed PAN/PVDF multichain
mckit operate polymer build --smiles "[*:1]CC(C#N)[*:2]" --smiles "[*:1]CC(F)(F)[*:2]" --names pan,pvdf --mode multichain_crossed_mixed --repeats 4,4 --out-dir smoke_mixed

# PVDF-HFP copolymer sequence proxy
mckit operate polymer build --smiles "[*:1]CC(F)(F)[*:2]" --smiles "[*:1]C(F)(C(F)(F)F)C(F)(F)[*:2]" --names vdf,hfp --mode copolymer_sequence --sequence 0,1,0 --out-dir smoke_copolymer

# Oligoether-grafted siloxane proxy
mckit operate polymer build --smiles "[*:1]O[Si](C)(COCCOCCOC)[*:2]" --name pdms_oeo --mode graft_sidechain --repeats 2 --out-dir smoke_graft

# PSTFSI-Li single-ion conductor fragment
mckit operate polymer build --smiles "[*:1]CC([*:2])c1ccc(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)cc1.[Li+]" --name pstfsi_li --mode single_ion --repeats 1 --out-dir smoke_single_ion

# Detect possible polymer connection sites for a normal molecule
mckit operate polymer detect --smiles "CCO" --max-candidates 10 --out-dir detect_ethanol
```


## 2. Development

<details>
    <summary>Click to expand development instructions</summary>

### Prerequisites

- Python ≥ 3.9
- pip

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd matcraft-kit

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Project Structure

```
mckit/
├── __init__.py              # Package root — exports public API
├── core/
│   ├── lattice.py           # Lattice dataclass (3×3 matrix, ASE Cell-backed)
│   └── tool.py              # Abstract base classes: Operation, Observation
├── operate/
│   ├── bulk.py              # BulkBuilder — standard crystal structures
│   └── surface.py           # SurfaceBuilder, TerminationAnalyzer, MoleculeDetector
├── observe/
│   ├── inspect.py              # StructureInspect — structural summary
│   └── fundamental.py       # FundamentalCheck — geometric validity checks
└── io/
    ├── reader.py            # read_structure() -> ASE Atoms
    └── writer.py            # write_structure() — ASE io.write
```

### Architecture

The framework follows an **Operation / Observation** pattern:

- **`Operation`** (abstract) — tools that **build or modify** structures. Subclasses implement `apply(...)` and return `ase.Atoms`.
- **`Observation`** (abstract) — tools that **inspect** structures without modifying them. Subclasses implement `observe(structure) → Any`.

Both are single-method ABCs defined in `mckit.core.tool`, making it straightforward to add new operations or observations.

**Core data flow:**

```
File (CIF, VASP, ...) ──read_structure()──▶ ase.Atoms ──Operation──▶ ase.Atoms ──write_structure()──▶ File
                                                │
                                                └──Observation──▶ dict / CheckResult / ...
```

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=mckit --cov-report=term-missing
```

### Adding a New Operation

Create a file under `mckit/operate/` and subclass `Operation`:

```python
from ase import Atoms
from mckit.core.tool import Operation

class MyBuilder(Operation):
    """Build or modify a structure."""

    def apply(self, *, some_param: float, **kwargs) -> Atoms:
        # ... your logic here ...
        return new_structure
```

Then re-export it from `mckit/operate/__init__.py`.

### Adding a New Observation

Create a file under `mckit/observe/` and subclass `Observation`:

```python
from ase import Atoms
from mckit.core.tool import Observation

class MyAnalysis(Observation):
    """Inspect a structure and return results."""

    def observe(self, structure: Atoms, **kwargs):
        # ... your logic here ...
        return result_dict
```

Then re-export it from `mckit/observe/__init__.py`.


### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy ≥ 1.21` | Numerical computing |
| `ase ≥ 3.22` | Atomic Simulation Environment — crystal builders, I/O, coordinate transforms |
| `pymatgen ≥ 2022.0.0` | Python Materials Genomics — CIF parsing, symmetry analysis |

</details>
