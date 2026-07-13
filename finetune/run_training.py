"""
Quick remap + train — MEMORY-SAFE version for 16GB M4.
batch=4, workers=0, yolov8n, 15 epochs.
"""

import os
import shutil
from pathlib import Path

SRC = Path("/Users/priyanshu/Downloads/geoquery 2/datasets/VisDrone")
DST = Path("/Users/priyanshu/Downloads/geoquery 2/finetune/datasets/visdrone_remapped")

REMAP = {3: 0, 4: 0, 5: 0, 8: 0}

# Dataset already remapped from previous runs
if not (DST / "train" / "labels").exists():
    def remap_split(split):
        src_img = SRC / "images" / split
        src_lbl = SRC / "labels" / split
        dst_img = DST / split / "images"
        dst_lbl = DST / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        kept = 0
        for lbl_file in sorted(src_lbl.glob("*.txt")):
            new_lines = []
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) in REMAP:
                        new_lines.append(f"{REMAP[int(parts[0])]} {' '.join(parts[1:])}\n")
            if new_lines:
                for ext in [".jpg", ".png"]:
                    img_src = src_img / (lbl_file.stem + ext)
                    if img_src.exists():
                        dst_p = dst_img / img_src.name
                        if not dst_p.exists():
                            try: os.symlink(img_src, dst_p)
                            except: pass
                        break
                with open(dst_lbl / lbl_file.name, "w") as f:
                    f.writelines(new_lines)
                kept += 1
        print(f"  {split}: {kept} images")
    print("Remapping...")
    for s in ["train", "val"]:
        remap_split(s)

import yaml
data = {"path": str(DST), "train": "train/images", "val": "val/images", "nc": 1, "names": {0: "vehicle"}}
with open(DST / "data.yaml", "w") as f:
    yaml.dump(data, f)

print("\n🚀 YOLOv8n training — MEMORY SAFE config")
print("   batch=4, workers=0, 15 epochs")
print("   Should run stable at ~1.0-1.2 it/s\n")

from ultralytics import YOLO
model = YOLO("yolov8n.pt")

results = model.train(
    data=str(DST / "data.yaml"),
    epochs=15,
    imgsz=640,
    batch=4,           # small batch = stable memory
    patience=5,
    device="mps",
    workers=0,          # no multiprocessing = no extra memory
    project=str(DST.parent / "runs"),
    name="geoquery_vehicle",
    exist_ok=True,
    cache=False,        # don't cache images in RAM
    # Aerial augmentation
    degrees=180.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    scale=0.5,
    # Optimization
    optimizer="AdamW",
    lr0=0.002,
    lrf=0.01,
    warmup_epochs=2,
    verbose=True,
    plots=True,
)

best_src = DST.parent / "runs" / "geoquery_vehicle" / "weights" / "best.pt"
best_dst = Path("/Users/priyanshu/Downloads/geoquery 2/finetune/best.pt")
if best_src.exists():
    shutil.copy2(best_src, best_dst)
    print(f"\n✅ Done! Weights → {best_dst}")
    print("Set USE_YOLO = True in config.py")
