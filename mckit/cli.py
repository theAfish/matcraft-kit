"""mckit CLI — auto-dispatching central entry point.

Any module under ``mckit.operate`` or ``mckit.observe`` can hook into the CLI
by defining a ``register_cli(subparsers)`` function.  This file discovers and
calls them automatically, so adding a new tool never requires touching this file.

"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path


def _discover_modules():
    """Yield modules with ``register_cli`` from mckit.operate and mckit.observe."""
    import mckit.operate
    import mckit.observe

    for pkg in (mckit.operate, mckit.observe):
        pkg_path = str(Path(pkg.__file__).parent)
        for info in pkgutil.iter_modules([pkg_path]):
            if info.name.startswith("_"):
                continue
            mod = importlib.import_module(f"{pkg.__name__}.{info.name}")
            if hasattr(mod, "register_cli"):
                yield mod


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="mckit",
        description="MatCraft Kit",
    )
    top_sub = parser.add_subparsers(dest="category")

    operate = top_sub.add_parser("operate", help="Build / modify structures")
    observe = top_sub.add_parser("observe", help="Inspect structures")
    # defect = top_sub.add_parser("defect", help="Quick defect workflows")

    op_sub = operate.add_subparsers(dest="tool")
    ob_sub = observe.add_subparsers(dest="tool")
    # defect_sub = defect.add_subparsers(dest="tool")

    for mod in _discover_modules():
        package = mod.__name__.split(".")[-2]  # 'operate' or 'observe'
        sub = op_sub if package == "operate" else ob_sub
        mod.register_cli(sub)

    # # Optional top-level shortcuts (currently defect workflows).
    # try:
    #     from mmkit.operate import defect_creation as defect_mod

    #     if hasattr(defect_mod, "register_cli_root"):
    #         defect_mod.register_cli_root(defect_sub)
    # except Exception:
    #     # Keep CLI resilient if optional shortcut wiring fails.
    #     pass

    return parser


def main():
    """Console entry point."""
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "category", None):
        parser.print_help()
    elif hasattr(args, "handler"):
        try:
            args.handler(args)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
