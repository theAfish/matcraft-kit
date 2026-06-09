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
        from mckit.operate.bulk import BulkBuilder
        return BulkBuilder
    if name in ("AntiSiteCreator", "InterstitialCreator", "SubstitutionCreator", "VacancyCreator"):
        from mckit.operate import defect_creation
        return getattr(defect_creation, name)
    if name in ("InterfaceBuilder", "InterfaceTermination"):
        from mckit.operate import interface
        return getattr(interface, name)
    if name in ("BatchPerturbationBuilder", "PerturbationBuilder"):
        from mckit.operate import perturbation
        return getattr(perturbation, name)
    if name == "SupercellBuilder":
        from mckit.operate.supercell import SupercellBuilder
        return SupercellBuilder
    if name in ("SurfaceBuilder", "TerminationAnalyzer", "MoleculeRepair", "Termination"):
        from mckit.operate import surface
        return getattr(surface, name)
    if name == "MoleculeDetector":
        from mckit.operate.molecule_utils import MoleculeDetector
        return MoleculeDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
