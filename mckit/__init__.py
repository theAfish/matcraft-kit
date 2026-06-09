"""
matmod - Materials Modelling Toolkit

A modular framework for building and analyzing atomic structures.
Backed by ASE and pymatgen for crystallographic computations.

Two main subsystems:
  - operations : tools that BUILD or MODIFY structures (bulk, surface, defect, ...)
  - observations: tools that INSPECT structures (info, checks, properties, ...)
"""

from ase import Atoms

from mckit.core.lattice import Lattice
from mckit.core.tool import Operation, Observation
from mckit.io import read_structure, write_structure

__version__ = "0.2.0"
__all__ = [
    "Lattice",
    "Atoms",
    "Operation",
    "Observation",
    "read_structure",
    "write_structure",
]
