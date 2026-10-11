"""Command-line entry point. WP00 stub: only prints usage; WP09 adds the real commands."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the requested action. Returns the exit code."""
    parser = argparse.ArgumentParser(
        prog="thermosim",
        description="ThermoSim: thermal simulation of a heated vessel.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.parse_args(argv)
    parser.print_help()
    return 0
