"""Structure inspection tools."""

__all__ = [
    "BasicInfo",
    "InfoSection",
    "MoleculeInfo",
    "SlabCompositionInfo",
    "StructureInfo",
    "StructureInspect",
    "VacuumInfo",
    "FundamentalCheck",
]


def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies at startup."""
    if name in (
        "BasicInfo",
        "InfoSection",
        "MoleculeInfo",
        "SlabCompositionInfo",
        "StructureInfo",
        "StructureInspect",
        "VacuumInfo",
    ):
        from mckit.observe import inspect
        if name == "StructureInfo":
            return inspect.StructureInspect
        return getattr(inspect, name)
    if name == "FundamentalCheck":
        from mckit.observe.fundamental import FundamentalCheck
        return FundamentalCheck
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
