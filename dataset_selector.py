"""
dataset_selector.py — Dynamic Dataset & Scene Discovery for Habitat-VLN
=======================================================================

Provides CLI-based interactive selection of:
  1. Dataset folder (e.g., hm3d, habitat_test_scenes)
  2. Scene (.glb file) within the chosen dataset

Also resolves companion assets:
  - .navmesh  (navigation mesh)
  - .semantic.glb  (semantic mesh)
  - .semantic.txt  (semantic class labels)

All paths use pathlib / os.path for OS-agnostic handling.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SceneConfig:
    """Holds all resolved paths for a selected scene."""
    scene_path: Path                # Path to the .glb file
    render_asset_path: Path         # Asset actually loaded by habitat-sim
    navmesh_path: Optional[Path]    # Path to the .navmesh file (may be None)
    semantic_glb_path: Optional[Path]   # Path to .semantic.glb (may be None)
    semantic_txt_path: Optional[Path]   # Path to .semantic.txt (may be None)
    dataset_config_path: Optional[Path] = None  # Habitat scene dataset config
    uses_basis_glb: bool = False
    dataset_name: str = ""          # Name of the parent dataset folder
    scene_name: str = ""            # Short name (stem of the .glb file)

    def __post_init__(self):
        if not self.scene_name:
            self.scene_name = self.scene_path.stem
        if not self.dataset_name:
            self.dataset_name = self.scene_path.parent.parent.name


def find_matching_hm3d_config(
    dataset_root: Path, scene_folder_name: str
) -> Optional[Path]:
    """
    Find an HM3D scene-dataset config that explicitly references this scene.

    The downloaded mini-splits often ship a small subset of scenes plus one or
    more config JSONs at the dataset root. We only opt in when the config
    actually mentions the selected scene folder.
    """
    for config_path in sorted(
        dataset_root.glob("hm3d*_minival_basis.scene_dataset_config.json")
    ):
        try:
            config_text = config_path.read_text(encoding="utf-8")
        except OSError:
            continue

        if f"{scene_folder_name}/*.basis.glb" in config_text:
            return config_path

    return None


# ============================================================================
# Dataset Discovery
# ============================================================================

def discover_datasets(data_root: Optional[Path] = None) -> List[Path]:
    """
    Scan the scene_datasets directory and return a list of valid dataset
    folder paths (excluding files like .tar, .json, etc.).
    """
    if data_root is None:
        # Default: relative to this script's location
        base_dir = Path(__file__).parent.resolve()
        data_root = base_dir / "data" / "scene_datasets"

    if not data_root.exists():
        print(f"[ERROR] Data directory not found: {data_root}")
        return []

    datasets: List[Path] = []
    for entry in sorted(data_root.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            datasets.append(entry)

    return datasets


def print_datasets_menu(datasets: List[Path]) -> None:
    """Print a numbered menu of available datasets."""
    print("\n" + "=" * 60)
    print("  Available Datasets")
    print("=" * 60)
    for idx, ds_path in enumerate(datasets, start=1):
        # Count .glb files inside (non-basis, non-semantic)
        glb_count = count_main_scenes(ds_path)
        print(f"  [{idx}] {ds_path.name}  ({glb_count} scenes)")
    print("-" * 60)


def select_dataset(datasets: List[Path]) -> Optional[Path]:
    """
    Present an interactive menu for dataset selection.
    Returns the chosen dataset Path, or None if the user cancels.
    """
    if not datasets:
        print("[ERROR] No datasets found in data/scene_datasets/")
        return None

    print_datasets_menu(datasets)

    while True:
        try:
            choice = input("\nSelect a dataset by number (or 'q' to quit): ").strip()
            if choice.lower() in ("q", "quit", "exit"):
                print("Exiting.")
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                selected = datasets[idx]
                print(f"  >> Selected: {selected.name}")
                return selected
            else:
                print(f"  [ERROR] Please enter a number between 1 and {len(datasets)}.")
        except ValueError:
            print("  [ERROR] Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting.")
            return None


# ============================================================================
# Scene Discovery
# ============================================================================

def count_main_scenes(dataset_path: Path) -> int:
    """Count the number of 'main' .glb files (non-basis, non-semantic)."""
    return len(list(dataset_path.rglob("[!.]*.glb"))) - count_basis_scenes(dataset_path) - count_semantic_scenes(dataset_path)


def count_basis_scenes(dataset_path: Path) -> int:
    """Count .basis.glb files."""
    return len(list(dataset_path.rglob("*.basis.glb")))


def count_semantic_scenes(dataset_path: Path) -> int:
    """Count .semantic.glb files."""
    return len(list(dataset_path.rglob("*.semantic.glb")))


def discover_scenes(dataset_path: Path) -> List[Path]:
    """
    Find all 'main' .glb scene files in the dataset directory.
    
    Rules:
      - Includes files ending with .glb (case-insensitive)
      - Excludes *.basis.glb (these are compressed variants)
      - Excludes *.semantic.glb (these are semantic annotation meshes)
      - Scans recursively (HM3D uses subdirectories per scene)
    """
    all_glb_files: List[Path] = []
    
    for glb_path in sorted(dataset_path.rglob("*.glb")):
        name = glb_path.name.lower()
        # Skip basis and semantic variants
        if name.endswith(".basis.glb"):
            continue
        if name.endswith(".semantic.glb"):
            continue
        all_glb_files.append(glb_path)

    return all_glb_files


def print_scenes_menu(scenes: List[Path]) -> None:
    """Print a numbered menu of available scenes."""
    print("\n" + "=" * 60)
    print("  Available Scenes")
    print("=" * 60)
    for idx, scene_path in enumerate(scenes, start=1):
        # Show relative path from dataset root for clarity
        try:
            # Try to show a compact relative path
            parent = scene_path.parent
            if parent.name != scene_path.parent.parent.name:
                # HM3D style: 00800-TEEsavR23oF/TEEsavR23oF.glb
                rel = f"{parent.name}/{scene_path.name}"
            else:
                rel = scene_path.name
        except Exception:
            rel = scene_path.name
        print(f"  [{idx:2d}] {rel}")
    print("-" * 60)


def select_scene(scenes: List[Path]) -> Optional[Path]:
    """
    Present an interactive menu for scene selection.
    Returns the chosen .glb Path, or None if the user cancels.
    """
    if not scenes:
        print("[ERROR] No scene (.glb) files found in this dataset.")
        return None

    print_scenes_menu(scenes)

    while True:
        try:
            choice = input("\nSelect a scene by number (or 'q' to quit, 'a' for all): ").strip()
            if choice.lower() in ("q", "quit", "exit"):
                print("Exiting.")
                return None
            if choice.lower() == "a":
                print("  >> Using first scene as default.")
                return scenes[0]

            idx = int(choice) - 1
            if 0 <= idx < len(scenes):
                selected = scenes[idx]
                print(f"  >> Selected: {selected.name}")
                return selected
            else:
                print(f"  [ERROR] Please enter a number between 1 and {len(scenes)}.")
        except ValueError:
            print("  [ERROR] Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting.")
            return None


# ============================================================================
# Asset Resolution
# ============================================================================

def resolve_assets(scene_glb_path: Path) -> SceneConfig:
    """
    Given a path to a .glb scene file, find the corresponding:
      - .navmesh file
      - .semantic.glb file
      - .semantic.txt file

    Resolution strategy:
      1. Look in the same directory as the .glb file.
      2. Try exact stem match first (e.g., TEEsavR23oF.glb → TEEsavR23oF.navmesh).
      3. If not found, try .basis.navmesh (HM3D convention).
      4. For semantic files, try .semantic.glb and .semantic.txt.
    """
    scene_dir = scene_glb_path.parent
    dataset_root = scene_dir.parent
    stem = scene_glb_path.stem  # e.g., "TEEsavR23oF"

    basis_glb_path = scene_dir / f"{stem}.basis.glb"
    render_asset_path = scene_glb_path if scene_glb_path.exists() else basis_glb_path
    uses_basis_glb = render_asset_path == basis_glb_path

    # --- Navmesh resolution ---
    navmesh_path: Optional[Path] = None

    # Try 1: Exact stem match (e.g., TEEsavR23oF.navmesh)
    candidate = scene_dir / f"{stem}.navmesh"
    if candidate.exists():
        navmesh_path = candidate

    # Try 2: .basis.navmesh (HM3D convention)
    if navmesh_path is None:
        candidate = scene_dir / f"{stem}.basis.navmesh"
        if candidate.exists():
            navmesh_path = candidate

    # Try 3: Any .navmesh file in the same directory
    if navmesh_path is None:
        navmeshes = list(scene_dir.glob("*.navmesh"))
        if navmeshes:
            navmesh_path = navmeshes[0]

    # --- Semantic .glb resolution ---
    semantic_glb_path: Optional[Path] = None
    candidate = scene_dir / f"{stem}.semantic.glb"
    if candidate.exists():
        semantic_glb_path = candidate

    # --- Semantic .txt resolution ---
    semantic_txt_path: Optional[Path] = None
    candidate = scene_dir / f"{stem}.semantic.txt"
    if candidate.exists():
        semantic_txt_path = candidate

    # --- Dataset config resolution (mainly for HM3D semantics) ---
    dataset_config_path: Optional[Path] = None
    if dataset_root.name.lower() == "hm3d" and uses_basis_glb:
        dataset_config_path = find_matching_hm3d_config(dataset_root, scene_dir.name)

    # Build the config
    config = SceneConfig(
        scene_path=scene_glb_path,
        render_asset_path=render_asset_path,
        navmesh_path=navmesh_path,
        semantic_glb_path=semantic_glb_path,
        semantic_txt_path=semantic_txt_path,
        dataset_config_path=dataset_config_path,
        uses_basis_glb=uses_basis_glb,
        scene_name=stem,
        dataset_name=scene_dir.parent.name,
    )

    return config


def print_scene_config(config: SceneConfig) -> None:
    """Display the resolved scene configuration."""
    print("\n" + "-" * 60)
    print("  Scene Configuration")
    print("-" * 60)
    print(f"  Dataset:      {config.dataset_name}")
    print(f"  Scene:        {config.scene_name}")
    print(f"  GLB:          {config.scene_path}")
    print(f"  Render Asset: {config.render_asset_path}")
    print(f"  Navmesh:      {config.navmesh_path if config.navmesh_path else '[NOT FOUND]'}")
    print(f"  Semantic GLB: {config.semantic_glb_path if config.semantic_glb_path else '[N/A]'}")
    print(f"  Semantic TXT: {config.semantic_txt_path if config.semantic_txt_path else '[N/A]'}")
    print(f"  Dataset CFG:  {config.dataset_config_path if config.dataset_config_path else '[N/A]'}")
    print("-" * 60)


# ============================================================================
# Semantic Parsing
# ============================================================================

@dataclass
class SemanticClass:
    """Represents a single semantic class entry from a .semantic.txt file."""
    id: int
    color_hex: str       # e.g., "97C517"
    name: str            # e.g., "ceiling"
    floor: int           # e.g., 1

    @property
    def color_rgb(self) -> Tuple[int, int, int]:
        """Convert hex color string to (R, G, B) tuple."""
        h = self.color_hex.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def parse_semantic_txt(txt_path: Path) -> List[SemanticClass]:
    """
    Parse a HM3D .semantic.txt file into a list of SemanticClass entries.
    
    Format:
      Line 1: Header (ignored)
      Subsequent lines: id,hex_color,"class_name",floor
      Example: 1,97C517,"ceiling",1
    """
    classes: List[SemanticClass] = []
    
    if not txt_path or not txt_path.exists():
        return classes

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  [WARN] Failed to read semantic file {txt_path}: {e}")
        return classes

    for line in lines:
        line = line.strip()
        if not line or line.startswith("HM3D") or line.startswith("#"):
            continue  # Skip header / comments

        try:
            # Parse: id,hex_color,"class_name",floor
            parts = line.split(",")
            if len(parts) < 4:
                continue

            sem_id = int(parts[0])
            color_hex = parts[1].strip()
            # Class name may contain commas if quoted, but HM3D format
            # uses simple names without commas, so parts[2] is sufficient
            class_name = parts[2].strip().strip('"')
            floor = int(parts[3])

            classes.append(SemanticClass(
                id=sem_id,
                color_hex=color_hex,
                name=class_name,
                floor=floor,
            ))
        except (ValueError, IndexError) as e:
            # Skip malformed lines
            continue

    return classes


# ============================================================================
# Quick Test (when run directly)
# ============================================================================

def main():
    """Interactive dataset and scene selection."""
    print("=" * 60)
    print("  Habitat-VLN Dataset Selector")
    print("=" * 60)

    # Step 1: Discover datasets
    datasets = discover_datasets()
    if not datasets:
        print("[ERROR] No datasets found. Check data/scene_datasets/")
        sys.exit(1)

    # Step 2: Select dataset
    ds_path = select_dataset(datasets)
    if ds_path is None:
        sys.exit(0)

    # Step 3: Discover scenes
    scenes = discover_scenes(ds_path)
    if not scenes:
        print(f"[ERROR] No .glb scene files found in {ds_path}")
        sys.exit(1)

    # Step 4: Select scene
    scene_path = select_scene(scenes)
    if scene_path is None:
        sys.exit(0)

    # Step 5: Resolve assets
    config = resolve_assets(scene_path)
    print_scene_config(config)

    # Step 6: Parse semantic data if available
    if config.semantic_txt_path:
        classes = parse_semantic_txt(config.semantic_txt_path)
        print(f"\n  Parsed {len(classes)} semantic classes from {config.semantic_txt_path.name}")
        # Show a few examples
        for c in classes[:5]:
            print(f"    ID={c.id:3d}  color=#{c.color_hex}  name={c.name:20s}  floor={c.floor}")

    return config


if __name__ == "__main__":
    main()
