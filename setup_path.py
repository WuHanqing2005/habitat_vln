"""
setup_path.py — Habitat-Sim & Habitat-Lab Python Path Configuration
====================================================================

Must be imported BEFORE any habitat-sim or habitat-lab imports.
Adds the source repositories to sys.path so Python can find the
habitat_sim and habitat modules.

Uses pathlib for OS-agnostic path handling (Windows / WSL / Linux).
"""

import sys
import os
from pathlib import Path


def setup_environment():
    """
    Configure sys.path to include habitat-sim and habitat-lab source directories.

    Returns:
        True if paths were configured successfully, False otherwise.
    """
    # Get the project root directory (where this script lives)
    base_dir = Path(__file__).parent.resolve()

    # Paths to the library source directories
    habitat_sim_path = base_dir / "libs" / "habitat-sim-main" / "src_python"
    habitat_lab_path = base_dir / "libs" / "habitat-lab-main"

    # Verify paths exist
    paths_ok = True

    if not habitat_sim_path.exists():
        print(f"[WARN] habitat-sim path not found: {habitat_sim_path}")
        paths_ok = False
    else:
        if str(habitat_sim_path) not in sys.path:
            sys.path.append(str(habitat_sim_path))

    if not habitat_lab_path.exists():
        print(f"[WARN] habitat-lab path not found: {habitat_lab_path}")
        paths_ok = False
    else:
        if str(habitat_lab_path) not in sys.path:
            sys.path.append(str(habitat_lab_path))

    if paths_ok:
        print("Environment paths configured successfully.")
    else:
        print("[WARN] Some library paths are missing. Habitat imports may fail.")

    return paths_ok


# Run setup on import
_setup_ok = setup_environment()
