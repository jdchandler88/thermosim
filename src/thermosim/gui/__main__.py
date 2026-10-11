"""`python -m thermosim.gui` launches the GUI."""

from __future__ import annotations

import sys

from thermosim.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
