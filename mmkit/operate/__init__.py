"""Structure-building and modification operations."""

__all__ = [
    "AntiSiteCreator",
    "BatchPerturbationBuilder",
    "BulkBuilder",
    "InterstitialCreator",
    "InterfaceBuilder",
    "InterfaceTermination",
    "PerturbationBuilder",
    "SubstitutionCreator",
    "SupercellBuilder",
    "SurfaceBuilder",
    "TerminationAnalyzer",
    "Termination",
    "VacancyCreator",
    "MoleculeDetector",
    "MoleculeRepair",
]


def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies at startup."""
    if name == "BulkBuilder":
        from mmkit.operate.bulk import BulkBuilder
        return BulkBuilder
    if name in ("AntiSiteCreator", "InterstitialCreator", "SubstitutionCreator", "VacancyCreator"):
        from mmkit.operate import defect_creation
        return getattr(defect_creation, name)
    if name in ("InterfaceBuilder", "InterfaceTermination"):
        from mmkit.operate import interface
        return getattr(interface, name)
    if name in ("BatchPerturbationBuilder", "PerturbationBuilder"):
        from mmkit.operate import perturbation
        return getattr(perturbation, name)
    if name == "SupercellBuilder":
        from mmkit.operate.supercell import SupercellBuilder
        return SupercellBuilder
    if name in ("SurfaceBuilder", "TerminationAnalyzer", "MoleculeDetector", "MoleculeRepair", "Termination"):
        from mmkit.operate import surface
        return getattr(surface, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
