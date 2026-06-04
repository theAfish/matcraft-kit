# mmkit — Materials Modelling Toolkit

A modular Python framework for building and analyzing periodic atomic crystal structures. Backed by [ASE](https://wiki.fysik.dtu.dk/ase/) and [pymatgen](https://pymatgen.org/) for crystallographic computations.

## Features

- **Core data structures** — `Lattice`, `Structure` plus direct ASE / pymatgen interop
- **Operations** — build bulk crystals (FCC, BCC, HCP, diamond, zincblende, rocksalt) and surface slabs with termination control
- **Observations** — inspect structures (lattice params, density, composition) and run sanity checks (overlaps, out-of-cell atoms, density bounds)
- **I/O** — read/write structure files as ASE `Atoms` (CIF, VASP, extxyz, and all ASE-supported formats)
- **CLI** — command-line surface slab builder with termination analysis

---

## 1. Usage

### Installation

```bash
pip install -e .
```

Or install with dev dependencies (pytest, coverage):

```bash
pip install -e ".[dev]"
```

### Quick Start

```python
import mmkit as mm

# Build FCC copper
from mmkit.operate import BulkBuilder

builder = BulkBuilder()
cu = builder.apply(
    structure_type="fcc",
    element="Cu",
    a=3.61,  # lattice parameter in Angstroms
)
print(cu)
# Structure(Cu1, natoms=1, V=11.79 A^3)
```

### Inspect a Structure

```python
from mmkit.observations import StructureInfo, StructureCheck

# Get structural information
info = StructureInfo()
result = info.observe(cu)
print(f"Density: {result['density_g_cm3']:.2f} g/cm³")
print(f"Composition: {result['composition']}")

# Or pretty-print
info.print_summary(cu)

# Validate a structure
checker = StructureCheck(min_dist=0.5)
check = checker.observe(cu)
print(check)  # CheckResult(PASS, warnings=0, errors=0)
```

### Build Bulk Crystals

```python
from mmkit.operate import BulkBuilder
import mmkit as mm

builder = BulkBuilder()

# Single-element structures
cu = builder.apply(structure_type="fcc", element="Cu", a=3.61)
fe = builder.apply(structure_type="bcc", element="Fe", a=2.87)
ti = builder.apply(structure_type="hcp", element="Ti", a=2.95, c=4.68)
si = builder.apply(structure_type="diamond", element="Si", a=5.43)

# Binary structures
gaas = builder.apply(
    structure_type="zincblende",
    elements=["Ga", "As"],
    a=5.65,
)
nacl = builder.apply(
    structure_type="rocksalt",
    elements=["Na", "Cl"],
    a=5.64,
)
```

### Build Surface Slabs

```python
from mmkit.operate import SurfaceBuilder, TerminationAnalyzer

# Read a bulk structure from file (returns ase.Atoms)
bulk_atoms = mm.read_structure("SrTiO3.cif")

# Discover all terminations for a (001) surface
analyzer = TerminationAnalyzer()
terminations = analyzer.analyze(bulk, miller=(0, 0, 1), layers=6)
for t in terminations:
    print(f"  [{t.label}] top={t.top_label}, bot={t.bot_label}, symmetric={t.symmetric}")

# Build a slab with a specific termination
builder = SurfaceBuilder()
slab = builder.apply(
    bulk,
    miller=(0, 0, 1),
    termination=0,        # index or label
    layers=6,
    vacuum=15.0,          # vacuum thickness in Angstroms
)
mm.write_structure("slab.extxyz", slab)
```

### Read / Write Structures

```python
import mmkit as mm

# Read (auto-detects format; CIF uses pymatgen for better symmetry handling)
atoms = mm.read_structure("input.cif")
atoms = mm.read_structure("POSCAR", format="vasp")

# Write (format auto-detected from extension)
mm.write_structure("output.extxyz", atoms)
mm.write_structure("CONTCAR", atoms, format="vasp")
```

### Work with Core Types Directly

```python
import mmkit as mm
import numpy as np

# Lattice
lat = mm.Lattice.cubic(5.43)           # cubic
lat = mm.Lattice.hexagonal(2.95, 4.68) # hexagonal
lat = mm.Lattice.from_parameters(5.0, 5.0, 7.0, 90, 90, 120) # general

# Structure
struct = mm.Structure(lattice=lat)
struct.add_atom("Si", [0.0, 0.0, 0.0])
struct.add_atom("Si", [0.5, 0.5, 0.5])

# Properties
print(struct.num_atoms)      # 2
print(struct.density)         # density in g/cm³
print(struct.composition)     # {'Si': 2}
print(struct.get_distance(0, 1, mic=True))  # distance with minimum image

# Supercell
big = struct.supercell(2, 2, 2)  # 2x2x2 supercell

# Interop
atoms = struct.to_ase_atoms()        # → ase.Atoms
pmg = struct.to_pymatgen()           # → pymatgen.core.Structure
struct2 = mm.Structure.from_ase_atoms(atoms)
struct3 = mm.Structure.from_pymatgen(pmg)
```

### CLI — Surface Modeler

```bash
# Discover all terminations
python -m mmkit.operate.surface list_terminations \
    --input SrTiO3.cif \
    --miller 0 0 1 \
    --layers 6

# Build a slab (termination by index)
python -m mmkit.operate.surface build_slab \
    --input SrTiO3.cif \
    --miller 0 0 1 \
    --termination 0 \
    --layers 6 \
    --vacuum 15.0 \
    --output slab.extxyz

# Build all terminations at once
python -m mmkit.operate.surface build_slab \
    --input SrTiO3.cif \
    --miller 0 0 1 \
    --termination all \
    --layers 6

# With molecular preservation
python -m mmkit.operate.surface build_slab \
    --input CaCO3.cif \
    --miller 1 0 0 \
    --termination 0 \
    --preserve-molecules \
    --output slab.extxyz
```

**CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Bulk structure file |
| `--miller` | *(required)* | Miller indices (h k l) |
| `--layers` | `4` | Number of atomic layers |
| `--vacuum` | `15.0` | Vacuum thickness (Å) |
| `--margin` | `1.5` | Surface layer margin (Å) |
| `--preserve-molecules` | off | Preserve molecular integrity at surfaces |
| `--mol-tol` | `0.45` | Bond detection tolerance (Å) |
| `--mol-min-size` | `2` | Minimum atoms to count as a molecule |
| `--mol-extent` | `2.5` | Approximate molecular radius (Å) |
| `--no-strip-inorganic` | off | Don't strip inorganic atoms beyond molecular extent |

---

## 2. Development

### Prerequisites

- Python ≥ 3.9
- pip

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd mat-modelling-kit

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Project Structure

```
mmkit/
├── __init__.py              # Package root — exports public API
├── core/
│   ├── lattice.py           # Lattice dataclass (3×3 matrix, ASE Cell-backed)
│   ├── structure.py         # Structure dataclass (lattice + species + positions)
│   └── tool.py              # Abstract base classes: Operation, Observation
├── operate/
│   ├── bulk.py              # BulkBuilder — standard crystal structures
│   └── surface.py           # SurfaceBuilder, TerminationAnalyzer, MoleculeDetector
├── observe/
│   ├── info.py              # StructureInfo — structural summary
│   └── check.py             # StructureCheck — validation checks
└── io/
    ├── reader.py            # read_structure() -> ASE Atoms
    └── writer.py            # write_structure() — ASE io.write
```

### Architecture

The framework follows an **Operation / Observation** pattern:

- **`Operation`** (abstract) — tools that **build or modify** structures. Subclasses implement `apply(...)` and return a `Structure`.
- **`Observation`** (abstract) — tools that **inspect** structures without modifying them. Subclasses implement `observe(structure) → Any`.

Both are single-method ABCs defined in `mmkit.core.tool`, making it straightforward to add new operations or observations.

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
pytest --cov=mmkit --cov-report=term-missing
```

### Adding a New Operation

Create a file under `mmkit/operate/` and subclass `Operation`:

```python
from mmkit.core.structure import Structure
from mmkit.core.tool import Operation

class MyBuilder(Operation):
    """Build or modify a structure."""

    def apply(self, *, some_param: float, **kwargs) -> Structure:
        # ... your logic here ...
        return new_structure
```

Then re-export it from `mmkit/operate/__init__.py`.

### Adding a New Observation

Create a file under `mmkit/observe/` and subclass `Observation`:

```python
from mmkit.core.structure import Structure
from mmkit.core.tool import Observation

class MyAnalysis(Observation):
    """Inspect a structure and return results."""

    def observe(self, structure: Structure, **kwargs):
        # ... your logic here ...
        return result_dict
```

Then re-export it from `mmkit/observe/__init__.py`.

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

### License

<!-- Add your license here -->
