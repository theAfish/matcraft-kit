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
        from mmkit.observe import info
        return getattr(info, name)
    if name == "FundamentalCheck":
        from mmkit.observe.fundamental import FundamentalCheck
        return FundamentalCheck
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
