"""Desktop GUI (WP08): PySide6 + pyqtgraph shell.

The GUI talks to the simulation only through the `Runner` protocol (decision D7). Until
WP11 swaps in the real runner, `runner_factory.make_runner` returns the `FakeRunner`.
"""

from __future__ import annotations
