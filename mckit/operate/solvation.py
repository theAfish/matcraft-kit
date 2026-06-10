"""Fill an orthorhombic simulation cell with solvent molecules using Packmol."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

from mckit.core.conversion import StructureLike, to_ase_atoms
from mckit.core.tool import Operation


AVOGADRO_PER_ANGSTROM3 = 6.02214076e-4
WINDOWS_DLL_NOT_FOUND = 0xC0000135


def _load_structure(structure: str | Path | StructureLike) -> Atoms:
    if isinstance(structure, (str, Path)):
        from mckit.io import read_structure

        return read_structure(str(structure))
    return to_ase_atoms(structure)


def _load_solvent(
    solvent: str | Path | Atoms,
    source: str = "auto",
) -> tuple[Atoms, str]:
    """Resolve a molecule from a file, ASE name, or SMILES string."""
    if isinstance(solvent, Atoms):
        return solvent.copy(), "atoms"

    value = str(solvent)
    path = Path(value)
    if source not in {"auto", "file", "name", "smiles"}:
        raise ValueError("source must be 'auto', 'file', 'name', or 'smiles'.")

    if source == "file" or (source == "auto" and path.is_file()):
        return _load_structure(path), path.stem

    if source in {"auto", "name"}:
        from mckit.operate.molecule_creation import ASEMoleculeBuilder

        candidates = [value]
        if value.upper() != value:
            candidates.append(value.upper())
        for candidate in candidates:
            try:
                return ASEMoleculeBuilder().apply(name=candidate), candidate
            except (KeyError, NotImplementedError):
                pass
        if source == "name":
            raise ValueError(
                f"ASE has no predefined molecule named {value!r}."
            )

    if source in {"auto", "smiles"}:
        from mckit.operate.molecule_creation import SMILESMoleculeBuilder

        molecule = SMILESMoleculeBuilder().apply(
            smiles=value,
            vacuum=0.0,
        )
        return molecule, value

    raise ValueError(f"Could not resolve solvent molecule {value!r}.")


def _check_packmol_runtime() -> None:
    """Detect a broken bundled Windows executable before pymatgen runs it."""
    if os.name != "nt":
        return
    try:
        from packmol.cli import get_binary_path
    except (ImportError, ModuleNotFoundError):
        return

    binary = get_binary_path()
    probe = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        check=False,
    )
    if probe.returncode == WINDOWS_DLL_NOT_FOUND:
        raise RuntimeError(
            "The bundled Packmol executable cannot start because a required "
            "Windows DLL is missing (status 0xC0000135). This is an "
            "installation/runtime problem, not a packing failure. The "
            "Packmol PyPI Windows wheel contains an executable whose compiler "
            "runtime DLLs are not available. Installing GCC alone does not "
            "repair that pre-built executable; Packmol is Fortran and needs "
            "the matching Fortran runtime. Use a Packmol Windows build that "
            "bundles its runtime DLLs or build Packmol with `gfortran` and "
            "place the resulting `packmol.exe` and runtime DLLs on PATH ahead "
            "of the pip launcher. WSL and conda-forge are alternatives, but "
            "Conda is not required. Do not download individual DLLs from "
            "untrusted sites."
        )


class SolvationBuilder(Operation):
    """Pack solvent into the empty space of an orthorhombic system cell.

    Packmol performs the molecular packing. ``concentration`` is the nominal
    molar concentration calculated from the full simulation-cell volume.
    Use ``count`` when an exact number of molecules is required.
    """

    def apply(
        self,
        *,
        system: str | Path | StructureLike,
        solvent: str | Path | Atoms,
        concentration: Optional[float] = None,
        count: Optional[int] = None,
        source: str = "auto",
        tolerance: float = 2.0,
        seed: int = 1,
        timeout: float = 120.0,
        boundary_margin: Optional[float] = None,
    ) -> Atoms:
        system_atoms = _load_structure(system)
        solvent_atoms, solvent_name = _load_solvent(solvent, source)
        lengths, axes = self._validate_cell(system_atoms)

        if (concentration is None) == (count is None):
            raise ValueError(
                "Provide exactly one of concentration (mol/L) or count."
            )
        if concentration is not None and concentration <= 0:
            raise ValueError("concentration must be positive.")
        if count is not None and count <= 0:
            raise ValueError("count must be a positive integer.")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if timeout <= 0:
            raise ValueError("timeout must be positive.")
        if len(system_atoms) == 0:
            raise ValueError("system must contain at least one atom.")
        if len(solvent_atoms) == 0:
            raise ValueError("solvent must contain at least one atom.")

        volume = float(system_atoms.get_volume())
        requested_concentration = concentration
        if count is None:
            count = int(np.floor(
                concentration * volume * AVOGADRO_PER_ANGSTROM3 + 0.5
            ))
            if count < 1:
                raise ValueError(
                    "The requested concentration gives fewer than one "
                    "molecule in this cell. Increase the cell or use count=1."
                )

        margin = tolerance / 2.0 if boundary_margin is None else boundary_margin
        if margin < 0:
            raise ValueError("boundary_margin must be non-negative.")
        if np.any(lengths <= 2.0 * margin):
            raise ValueError(
                "The cell is too small for the requested boundary margin."
            )

        packed_solvent = self._pack(
            system_atoms=system_atoms,
            solvent_atoms=solvent_atoms,
            count=count,
            lengths=lengths,
            axes=axes,
            tolerance=tolerance,
            margin=margin,
            seed=seed,
            timeout=timeout,
            solvent_name=solvent_name,
        )

        result_system = system_atoms.copy()
        result_system.set_array(
            "solvent_id", np.full(len(result_system), -1, dtype=int),
        )
        packed_solvent.set_array(
            "solvent_id",
            np.repeat(np.arange(count, dtype=int), len(solvent_atoms)),
        )
        result = result_system + packed_solvent
        result.set_array("solvent_mask", result.arrays["solvent_id"] >= 0)

        achieved = count / (volume * AVOGADRO_PER_ANGSTROM3)
        result.info.update({
            "solvent_name": solvent_name,
            "solvent_count": count,
            "solvent_atoms_per_molecule": len(solvent_atoms),
            "solvent_concentration_mol_l": achieved,
            "requested_solvent_concentration_mol_l": requested_concentration,
            "packmol_tolerance_ang": tolerance,
            "packmol_seed": seed,
        })
        return result

    @staticmethod
    def _validate_cell(system: Atoms) -> tuple[np.ndarray, np.ndarray]:
        cell = np.asarray(system.cell.array, dtype=float)
        lengths = np.linalg.norm(cell, axis=1)
        if np.any(lengths < 1e-10) or abs(np.linalg.det(cell)) < 1e-10:
            raise ValueError("system must have a non-singular 3D cell.")
        axes = cell / lengths[:, None]
        if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-7):
            raise ValueError(
                "Packmol solvation currently requires an orthorhombic cell."
            )
        return lengths, axes

    @staticmethod
    def _pack(
        *,
        system_atoms: Atoms,
        solvent_atoms: Atoms,
        count: int,
        lengths: np.ndarray,
        axes: np.ndarray,
        tolerance: float,
        margin: float,
        seed: int,
        timeout: float,
        solvent_name: str,
    ) -> Atoms:
        from pymatgen.core import Molecule
        from pymatgen.io.packmol import PackmolBoxGen

        wrapped = system_atoms.copy()
        wrapped.wrap()
        local_positions = wrapped.positions @ axes.T

        image_positions = []
        image_symbols = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    shift = np.array([i, j, k], dtype=float) * lengths
                    image_positions.extend(local_positions + shift)
                    image_symbols.extend(wrapped.get_chemical_symbols())

        fixed = Molecule(image_symbols, image_positions)
        centered_solvent = solvent_atoms.positions - np.mean(
            solvent_atoms.positions, axis=0,
        )
        solvent_molecule = Molecule(
            solvent_atoms.get_chemical_symbols(),
            centered_solvent,
        )
        box = [
            margin, margin, margin,
            lengths[0] - margin,
            lengths[1] - margin,
            lengths[2] - margin,
        ]
        safe_name = "solvent_" + "".join(
            char if char.isalnum() else "_" for char in solvent_name
        )[:40] or "solvent"

        generator = PackmolBoxGen(
            tolerance=tolerance,
            seed=seed,
            outputfile="packed.xyz",
            stdoutfile="packmol.stdout",
        )
        input_set = generator.get_input_set([
            {
                "name": "fixed_system_images",
                "number": 1,
                "coords": fixed,
                "constraints": ["fixed 0. 0. 0. 0. 0. 0."],
            },
            {
                "name": safe_name,
                "number": count,
                "coords": solvent_molecule,
                "constraints": [
                    "inside box " + " ".join(str(value) for value in box),
                ],
            },
        ])

        with TemporaryDirectory(prefix="mckit_packmol_") as directory:
            input_set.write_input(directory)
            try:
                _check_packmol_runtime()
                input_set.run(directory, timeout=timeout)
            except RuntimeError as exc:
                if "required Windows DLL" in str(exc):
                    raise
                raise RuntimeError(
                    "Packmol is required for solvation. Install it with "
                    "`pip install packmol` (or conda-forge on Windows) and "
                    "ensure `packmol` is on PATH."
                ) from exc
            except ValueError as exc:
                raise RuntimeError(
                    "Packmol failed while packing the solvent. "
                    f"Backend message: {exc}"
                ) from exc
            packed = ase_read(Path(directory) / "packed.xyz")

        fixed_count = len(fixed)
        expected = fixed_count + count * len(solvent_atoms)
        if len(packed) != expected:
            raise RuntimeError(
                f"Packmol returned {len(packed)} atoms; expected {expected}."
            )
        packed_solvent = packed[fixed_count:]
        packed_solvent.positions = packed_solvent.positions @ axes
        packed_solvent.cell = system_atoms.cell
        packed_solvent.pbc = system_atoms.pbc
        return packed_solvent


def _cmd_solvate(args) -> None:
    from mckit.io import write_structure

    result = SolvationBuilder().apply(
        system=args.system,
        solvent=args.solvent,
        concentration=args.concentration,
        count=args.count,
        source=args.source,
        tolerance=args.tolerance,
        seed=args.seed,
        timeout=args.timeout,
        boundary_margin=args.boundary_margin,
    )
    output = args.output or f"{Path(args.system).stem}_solvated.extxyz"
    path = write_structure(output, result)
    print(
        f"Added {result.info['solvent_count']} solvent molecules "
        f"({result.info['solvent_concentration_mol_l']:.6g} mol/L) -> {path}"
    )


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "solvation",
        help="Fill an orthorhombic system cell with solvent using Packmol",
    )
    parser.add_argument("system", help="Input system structure file")
    parser.add_argument(
        "solvent",
        help="Solvent file, ASE molecule name, or SMILES string",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--concentration", type=float,
        help="Target nominal concentration in mol/L",
    )
    group.add_argument(
        "--count", type=int,
        help="Exact number of solvent molecules",
    )
    parser.add_argument(
        "--source", choices=("auto", "file", "name", "smiles"),
        default="auto",
        help="How to interpret the solvent argument (default: auto)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=2.0,
        help="Packmol minimum atom distance in A (default: 2.0)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Packmol seed")
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="Packmol timeout in seconds",
    )
    parser.add_argument(
        "--boundary-margin", type=float,
        help="Keep solvent atoms this far from each cell face in A",
    )
    parser.add_argument("-o", "--output", help="Output structure file")
    parser.set_defaults(handler=_cmd_solvate)
