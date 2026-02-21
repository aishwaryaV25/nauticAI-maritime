import os, shutil, glob
from pathlib import Path

HULL_CLASSES = {
    "Bilge Keel":       0,
    "Draft Mark":       1,
    "Hull":             2,
    "Propeller":        3,
    "Ropeguard":        4,
    "Rudder":           5,
    "Sea Chest":        6,
    "Thruster Blades":  7,
    "Thruster Grating": 8,
}

src  = Path("data/dataset6_hull")
dst  = Path("data/dataset6_hull_yolo")

for split in ["train", "valid", "test"]:
    (dst / split / "images").mkdir(parents=True, exist_ok=True)
    (dst / split / "labels").mkdir(parents=True, exist_ok=True)
    
    for cls_name, cls_id in HULL_CLASSES.items():
        cls_dir = src / split / cls_name
        if not cls_dir.exists():
            continue
        for img_path in cls_dir.glob("*.jpg"):
            # Copy image
            dst_img = dst / split / "images" / img_path.name
            shutil.copy2(img_path, dst_img)
            # Create YOLO label (whole image bounding box)
            dst_lbl = dst / split / "labels" / (img_path.stem + ".txt")
            with open(dst_lbl, "w") as f:
                f.write(f"{cls_id} 0.5 0.5 1.0 1.0\n")

print("✅ Hull dataset converted!")
for split in ["train", "valid", "test"]:
    n = len(list((dst/split/"images").glob("*")))
    print(f"{split}: {n} images")