"""mmkit CLI — auto-dispatching central entry point.

Any module under ``mmkit.operate`` or ``mmkit.observe`` can hook into the CLI
by defining a ``register_cli(subparsers)`` function.  This file discovers and
calls them automatically, so adding a new tool never requires touching this file.

Usage::

    mmkit operate surface list cu.cif --miller 1 1 1
    mmkit operate bulk build --type fcc --element Cu --a 3.61
    mmkit observe info structure.cif
    mmkit observe check structure.cif --verbose
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path


def _discover_modules():
    """Yield modules with ``register_cli`` from mmkit.operate and mmkit.observe."""
    import mmkit.operate
    import mmkit.observe

    for pkg in (mmkit.operate, mmkit.observe):
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
        prog="mmkit",
        description="Materials Modelling Kit",
    )
    top_sub = parser.add_subparsers(dest="category")

    operate = top_sub.add_parser("operate", help="Build / modify structures")
    observe = top_sub.add_parser("observe", help="Inspect structures")

    op_sub = operate.add_subparsers(dest="tool")
    ob_sub = observe.add_subparsers(dest="tool")

    for mod in _discover_modules():
        package = mod.__name__.split(".")[-2]  # 'operate' or 'observe'
        sub = op_sub if package == "operate" else ob_sub
        mod.register_cli(sub)

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
