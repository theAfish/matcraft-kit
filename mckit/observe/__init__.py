"""Structure inspection tools."""

__all__ = [
    "BasicInfo",
    "InfoSection",
    "SlabCompositionInfo",
    "StructureInfo",
    "VacuumInfo",
    "FundamentalCheck",
]


def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies at startup."""
    if name in ("BasicInfo", "InfoSection", "SlabCompositionInfo", "StructureInfo", "VacuumInfo"):
        from mckit.observe import inspect
        return getattr(inspect, name)
    if name == "FundamentalCheck":
        from mckit.observe.fundamental import FundamentalCheck
        return FundamentalCheck
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
