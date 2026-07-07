"""Quick test to verify asset resolution for HM3D scenes."""
import sys
sys.path.insert(0, 'f:/habitat_vln')

from pathlib import Path
from dataset_selector import discover_datasets, discover_scenes, resolve_assets

# Test HM3D dataset
hm3d = Path('f:/habitat_vln/data/scene_datasets/hm3d')
scenes = discover_scenes(hm3d)
print(f"Found {len(scenes)} scenes in hm3d")

for s in scenes:
    config = resolve_assets(s)
    nav_ok = "OK" if config.navmesh_path else "MISSING"
    sem_glb_ok = "OK" if config.semantic_glb_path else "N/A"
    sem_txt_ok = "OK" if config.semantic_txt_path else "N/A"
    print(f"  {s.parent.name}/{s.name}")
    print(f"    Navmesh: {nav_ok} -> {config.navmesh_path}")
    print(f"    Sem.glb: {sem_glb_ok}")
    print(f"    Sem.txt: {sem_txt_ok}")
