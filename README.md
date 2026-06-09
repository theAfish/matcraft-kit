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
│   ├── structure.py         # Structure dataclass (lattice + species + positions)
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

- **`Operation`** (abstract) — tools that **build or modify** structures. Subclasses implement `apply(...)` and return a `Structure`.
- **`Observation`** (abstract) — tools that **inspect** structures without modifying them. Subclasses implement `observe(structure) → Any`.

Both are single-method ABCs defined in `mckit.core.tool`, making it straightforward to add new operations or observations.

**Core data flow:**

```
File (CIF, VASP, ...) ──read_structure()──▶ ase.Atoms ──Operation──▶ Structure/ase.Atoms ──write_structure()──▶ File
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
from mckit.core.structure import Structure
from mckit.core.tool import Operation

class MyBuilder(Operation):
    """Build or modify a structure."""

    def apply(self, *, some_param: float, **kwargs) -> Structure:
        # ... your logic here ...
        return new_structure
```

Then re-export it from `mckit/operate/__init__.py`.

### Adding a New Observation

Create a file under `mckit/observe/` and subclass `Observation`:

```python
from mckit.core.structure import Structure
from mckit.core.tool import Observation

class MyAnalysis(Observation):
    """Inspect a structure and return results."""

    def observe(self, structure: Structure, **kwargs):
        # ... your logic here ...
        return result_dict
```

Then re-export it from `mckit/observe/__init__.py`.

### Building the Package

```bash
pip install build
python -m build
```

This produces a wheel and sdist under `dist/`.

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy ≥ 1.21` | Numerical computing |
| `ase ≥ 3.22` | Atomic Simulation Environment — crystal builders, I/O, coordinate transforms |
| `pymatgen ≥ 2022.0.0` | Python Materials Genomics — CIF parsing, symmetry analysis |

</details>
