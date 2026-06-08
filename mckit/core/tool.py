"""Abstract base classes for matmod tools.

Two families:

* ``Operation``  — *builds* or *modifies* a ``Structure``. Subclasses define
  ``apply(...)`` with whatever signature makes sense (no fixed contract on
  arguments — the only requirement is that it returns a ``Structure``).
* ``Observation`` — *inspects* a ``Structure`` and returns arbitrary data
  without modifying it. Subclasses implement ``observe(structure, **kwargs)``.

Adding a new tool is a one-class affair — see the examples in
``matmod/operations/bulk.py`` and ``matmod/observations/info.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mckit.core.structure import Structure


class Operation(ABC):
    """Base class for structure-building / modifying tools."""

    @abstractmethod
    def apply(self, *args, **kwargs) -> Structure:
        """Run the operation and return the resulting ``Structure``."""

    def __call__(self, *args, **kwargs) -> Structure:
        return self.apply(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} Operation>"


class Observation(ABC):
    """Base class for structure-inspection tools."""

    @abstractmethod
    def observe(self, structure: Structure, **kwargs) -> Any:
        """Inspect the structure and return arbitrary result data."""

    def __call__(self, structure: Structure, **kwargs) -> Any:
        return self.observe(structure, **kwargs)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} Observation>"
