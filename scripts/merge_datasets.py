"""
NautiCAI Dataset Merger
Merges 6 Roboflow datasets into one unified YOLO dataset
with consistent class mapping
Datasets:
    1. Underwater Pipelines (1,131 images)
    2. Subsea (7,105 images)
    3. Biofouling Detection (4,259 images)
    4. Marine Growth 70 images
    5. Marine Growth 781 images
    6. Hull Scan (1,187 images)
Total: ~15,845 images | 19 classes
"""

import os
import shutil
import glob
from pathlib import Path

# ── Final class mapping ───────────────────────────────────────────────────────
FINAL_CLASSES = {
    # Subsea Infrastructure
    "pipeline":         0,
    "leakage":          1,
    "pipe_coupling":    2,
    "concrete":         3,
    "anode":            4,
    "flange":           5,
    "buoy":             6,
    "bend_restrictor":  7,
    "biofouling":       8,
    "anomaly":          9,
    # Hull Classes
    "bilge_keel":       10,
    "draft_mark":       11,
    "hull":             12,
    "propeller":        13,
    "ropeguard":        14,
    "rudder":           15,
    "sea_chest":        16,
    "thruster_blades":  17,
    "thruster_grating": 18,
}

# ── Per-dataset class remapping ───────────────────────────────────────────────
DATASET_MAPS = {
    "dataset1_pipelines": {
        0: "concrete",        # Concrete Mat
        1: "concrete",        # Concrete weight
        2: "leakage",         # Leakage
        3: "pipeline",        # Pipe
        4: "pipe_coupling",   # Pipe coupling
    },
    "dataset2_subsea": {
        0: None,              # 1-stripe-marker (skip)
        1: None,              # 2-stripe-marker (skip)
        2: None,              # 3-stripe-marker (skip)
        3: "anode",           # anode
        4: "bend_restrictor", # bend-restrictor
        5: "buoy",            # buoy
        6: "flange",          # flange
        7: None,              # lifting-strop (skip)
    },
    "dataset3_biofouling": {
        0: "biofouling",      # biofouling
    },
    "dataset4_marinegrowth_70": {
        0: "biofouling",      # MarineGrowth → biofouling
    },
    "dataset5_marinegrowth_781": {
        0: "anomaly",         # anomaly
        1: "biofouling",      # marine-growth → biofouling
    },
    "dataset6_hull_yolo": {
        0: "bilge_keel",
        1: "draft_mark",
        2: "hull",
        3: "propeller",
        4: "ropeguard",
        5: "rudder",
        6: "sea_chest",
        7: "thruster_blades",
        8: "thruster_grating",
    },
}

ROOT      = Path("data")
MERGED    = ROOT / "merged"
SPLITS    = ["train", "valid", "test"]
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}

def remap_label_file(src_path, dst_path, class_map):
    lines_out = []
    with open(src_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts    = line.split()
            orig_id  = int(parts[0])
            new_name = class_map.get(orig_id)
            if new_name is None:
                continue
            new_id   = FINAL_CLASSES[new_name]
            parts[0] = str(new_id)
            lines_out.append(" ".join(parts))

    if lines_out:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(dst_path, "w") as f:
            f.write("\n".join(lines_out) + "\n")
        return True
    return False

def merge():
    # Create output dirs
    for split in ["train", "val", "test"]:
        os.makedirs(MERGED / "images" / split, exist_ok=True)
        os.makedirs(MERGED / "labels" / split, exist_ok=True)

    total_images = 0

    for ds_name, class_map in DATASET_MAPS.items():
        ds_path = ROOT / ds_name
        if not ds_path.exists():
            print(f"⚠️  Skipping {ds_name} — folder not found")
            continue

        print(f"\n📂 Processing {ds_name}...")
        ds_count = 0

        for split in SPLITS:
            img_dir   = ds_path / split / "images"
            lbl_dir   = ds_path / split / "labels"
            out_split = SPLIT_MAP[split]

            if not img_dir.exists():
                continue

            for img_path in glob.glob(str(img_dir / "*.*")):
                img_path = Path(img_path)
                stem     = img_path.stem
                ext      = img_path.suffix
                lbl_path = lbl_dir / (stem + ".txt")

                if not lbl_path.exists():
                    continue

                new_name = f"{ds_name}_{stem}"
                dst_img  = MERGED / "images" / out_split / (new_name + ext)
                dst_lbl  = MERGED / "labels" / out_split / (new_name + ".txt")

                ok = remap_label_file(str(lbl_path), str(dst_lbl), class_map)
                if ok:
                    os.makedirs(dst_img.parent, exist_ok=True)
                    shutil.copy2(img_path, dst_img)
                    ds_count += 1

        print(f"   ✅ {ds_count} images copied")
        total_images += ds_count

    print(f"\n🎉 Merge complete! Total images: {total_images}")

    # Write final data.yaml
    yaml_content = f"""train: data/merged/images/train
val:   data/merged/images/val
test:  data/merged/images/test

nc: {len(FINAL_CLASSES)}
names: {list(FINAL_CLASSES.keys())}
"""
    with open("data/merged/data.yaml", "w") as f:
        f.write(yaml_content)

    print("✅ data/merged/data.yaml written!")
    print(f"\nFinal {len(FINAL_CLASSES)} classes:")
    for name, idx in FINAL_CLASSES.items():
        print(f"  {idx}: {name}")

if __name__ == "__main__":
    merge()