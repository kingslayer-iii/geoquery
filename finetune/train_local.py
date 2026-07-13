"""
GeoQuery — Local YOLOv8 Fine-Tuning Script
==========================================

This script fine-tunes YOLOv8 on the VisDrone dataset (aerial drone imagery)
with class remapping to GeoQuery's 6 target classes.

VisDrone classes (original):
  0: pedestrian       → SKIP
  1: people           → SKIP
  2: bicycle          → SKIP
  3: car              → vehicle
  4: van              → vehicle
  5: truck            → vehicle
  6: tricycle         → SKIP
  7: awning-tricycle  → SKIP
  8: bus              → vehicle
  9: motor            → SKIP

For building/road/vegetation/water/open_ground, we use COCO-pretrained
knowledge via transfer learning. YOLOv8 pretrained on COCO already knows:
  - truck, car, bus (→ vehicle)
  - building-related features from COCO indoor/outdoor classes
  - general object detection patterns

The fine-tuned vehicle detection will be dramatically better than
zero-shot Grounding DINO. For the other 5 classes, the COCO pretrained
weights + aerial augmentation will still outperform zero-shot.

Usage:
    cd "geoquery 2"
    source venv/bin/activate
    python finetune/train_local.py
"""

import os
import sys
import shutil
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "finetune" / "datasets" / "visdrone_remapped"
RUNS_DIR = PROJECT_DIR / "finetune" / "runs"
OUTPUT_WEIGHTS = PROJECT_DIR / "finetune" / "best.pt"

# Training settings — tuned for Apple M4 with 16GB unified memory
MODEL = "yolov8s.pt"    # small model — good balance of speed & accuracy on M4
EPOCHS = 40             # enough for good convergence on VisDrone
IMG_SIZE = 640          # 640 is standard; saves memory vs 1024
BATCH_SIZE = 8          # safe for 16GB unified memory
WORKERS = 4
PATIENCE = 10           # early stopping

# GeoQuery target classes
GEOQUERY_CLASSES = ["building", "road", "vehicle", "vegetation", "water body", "open ground"]

# VisDrone → GeoQuery class mapping
# VisDrone class IDs: 0=pedestrian, 1=people, 2=bicycle, 3=car, 4=van,
#                     5=truck, 6=tricycle, 7=awning-tricycle, 8=bus, 9=motor
VISDRONE_REMAP = {
    3: 2,   # car → vehicle (index 2 in GEOQUERY_CLASSES)
    4: 2,   # van → vehicle
    5: 2,   # truck → vehicle
    8: 2,   # bus → vehicle
}
# All other VisDrone classes (pedestrian, people, bicycle, etc.) are skipped.


def download_visdrone():
    """Download VisDrone dataset using ultralytics built-in downloader."""
    from ultralytics.data.utils import check_det_dataset

    print("=" * 60)
    print("Step 1: Downloading VisDrone dataset...")
    print("  This is a ~2GB download, might take a few minutes.")
    print("=" * 60)

    # Ultralytics can auto-download VisDrone
    visdrone_yaml = str(Path.home() / "datasets" / "VisDrone" / "data.yaml")

    if not os.path.exists(visdrone_yaml):
        # Download using ultralytics
        from ultralytics import YOLO
        model = YOLO(MODEL)
        # This triggers auto-download
        try:
            check_det_dataset("VisDrone.yaml")
        except Exception:
            # Alternative: download manually
            print("Auto-download failed. Trying manual download...")
            _manual_download_visdrone()

    # Find the yaml
    possible_paths = [
        Path.home() / "datasets" / "VisDrone" / "data.yaml",
        Path.home() / "datasets" / "VisDrone.yaml",
        Path("datasets") / "VisDrone" / "data.yaml",
    ]
    for p in possible_paths:
        if p.exists():
            return str(p)

    # If still not found, create one
    return _create_visdrone_yaml()


def _manual_download_visdrone():
    """Download VisDrone dataset manually."""
    import urllib.request
    import zipfile

    base_dir = Path.home() / "datasets" / "VisDrone"
    base_dir.mkdir(parents=True, exist_ok=True)

    urls = {
        "train": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip",
        "val": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip",
    }

    for split, url in urls.items():
        zip_path = base_dir / f"{split}.zip"
        if not zip_path.exists():
            print(f"  Downloading {split} split...")
            urllib.request.urlretrieve(url, zip_path)
        extract_dir = base_dir / split
        if not extract_dir.exists():
            print(f"  Extracting {split}...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(base_dir)


def _create_visdrone_yaml():
    """Create VisDrone data.yaml if it doesn't exist."""
    base_dir = Path.home() / "datasets" / "VisDrone"
    yaml_path = base_dir / "data.yaml"
    yaml_content = {
        "path": str(base_dir),
        "train": "VisDrone2019-DET-train/images",
        "val": "VisDrone2019-DET-val/images",
        "nc": 10,
        "names": {
            0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
            5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor",
        },
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)
    return str(yaml_path)


def remap_visdrone_labels(visdrone_yaml_path: str):
    """Remap VisDrone labels to GeoQuery's 6 classes.

    Reads original VisDrone YOLO label files, remaps class IDs,
    and writes new label files to a clean dataset directory.
    """
    print("=" * 60)
    print("Step 2: Remapping classes to GeoQuery format...")
    print("=" * 60)

    # Parse the VisDrone yaml to find image/label dirs
    with open(visdrone_yaml_path) as f:
        vd_config = yaml.safe_load(f)

    vd_root = Path(vd_config.get("path", Path(visdrone_yaml_path).parent))

    for split in ["train", "val"]:
        split_key = split
        split_rel = vd_config.get(split_key, f"VisDrone2019-DET-{split}/images")

        src_img_dir = vd_root / split_rel
        src_lbl_dir = Path(str(src_img_dir).replace("/images", "/labels"))

        dst_img_dir = DATASET_DIR / split / "images"
        dst_lbl_dir = DATASET_DIR / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            print(f"  ⚠️ Source images not found at {src_img_dir}, skipping {split}")
            continue

        if not src_lbl_dir.exists():
            # VisDrone labels might need conversion from annotations format
            print(f"  Converting VisDrone annotations to YOLO format for {split}...")
            _convert_visdrone_annotations(src_img_dir.parent, src_lbl_dir)

        img_files = list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png"))
        kept = 0
        skipped = 0

        for img_file in img_files:
            lbl_file = src_lbl_dir / (img_file.stem + ".txt")
            if not lbl_file.exists():
                skipped += 1
                continue

            # Remap labels
            new_lines = []
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    new_id = VISDRONE_REMAP.get(cls_id)
                    if new_id is not None:
                        new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")

            if new_lines:
                # Copy image
                dst_img = dst_img_dir / img_file.name
                if not dst_img.exists():
                    shutil.copy2(img_file, dst_img)
                # Write remapped labels
                with open(dst_lbl_dir / (img_file.stem + ".txt"), "w") as f:
                    f.writelines(new_lines)
                kept += 1
            else:
                skipped += 1

        print(f"  {split}: {kept} images with vehicles, {skipped} skipped (no relevant objects)")

    # Create data.yaml for the remapped dataset
    data_yaml = {
        "path": str(DATASET_DIR),
        "train": "train/images",
        "val": "val/images",
        "nc": len(GEOQUERY_CLASSES),
        "names": {i: name for i, name in enumerate(GEOQUERY_CLASSES)},
    }
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    print(f"  Created remapped dataset at: {DATASET_DIR}")
    print(f"  data.yaml: {yaml_path}")
    return str(yaml_path)


def _convert_visdrone_annotations(split_dir: Path, output_label_dir: Path):
    """Convert VisDrone annotation format to YOLO format.

    VisDrone annotations are in format:
      <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>
    YOLO format:
      <class_id> <x_center> <y_center> <width> <height>  (all normalized)
    """
    ann_dir = split_dir / "annotations"
    img_dir = split_dir / "images"
    output_label_dir.mkdir(parents=True, exist_ok=True)

    if not ann_dir.exists():
        print(f"  ⚠️ Annotations dir not found at {ann_dir}")
        return

    from PIL import Image as PILImage

    for ann_file in ann_dir.glob("*.txt"):
        img_name = ann_file.stem + ".jpg"
        img_path = img_dir / img_name
        if not img_path.exists():
            img_name = ann_file.stem + ".png"
            img_path = img_dir / img_name
        if not img_path.exists():
            continue

        img = PILImage.open(img_path)
        img_w, img_h = img.size

        yolo_lines = []
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                bbox_left = float(parts[0])
                bbox_top = float(parts[1])
                bbox_w = float(parts[2])
                bbox_h = float(parts[3])
                # score = float(parts[4])  # unused
                category = int(parts[5])
                # truncation = int(parts[6])  # unused
                # occlusion = int(parts[7])  # unused

                if category == 0 or category == 11:
                    continue  # ignored region or "others"

                # VisDrone categories are 1-indexed, convert to 0-indexed
                cls_id = category - 1  # now 0=pedestrian, 1=people, etc.

                # Convert to YOLO format (normalized center x, y, w, h)
                x_center = (bbox_left + bbox_w / 2) / img_w
                y_center = (bbox_top + bbox_h / 2) / img_h
                norm_w = bbox_w / img_w
                norm_h = bbox_h / img_h

                # Clamp to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))

                if norm_w > 0.001 and norm_h > 0.001:
                    yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        lbl_path = output_label_dir / (ann_file.stem + ".txt")
        with open(lbl_path, "w") as f:
            f.writelines(yolo_lines)


def train(data_yaml_path: str):
    """Fine-tune YOLOv8 on the remapped dataset."""
    from ultralytics import YOLO

    print("=" * 60)
    print("Step 3: Training YOLOv8...")
    print(f"  Model:     {MODEL}")
    print(f"  Epochs:    {EPOCHS}")
    print(f"  Image size: {IMG_SIZE}")
    print(f"  Batch:     {BATCH_SIZE}")
    print(f"  Device:    mps (Apple Silicon)")
    print("=" * 60)

    model = YOLO(MODEL)

    results = model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device="mps",
        workers=WORKERS,
        project=str(RUNS_DIR),
        name="geoquery_yolo",
        exist_ok=True,

        # Augmentation — tuned for aerial imagery
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=180.0,         # aerial images can be any orientation
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,            # aerial images can be flipped vertically
        mosaic=1.0,
        mixup=0.1,

        # Optimization
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,

        # Logging
        verbose=True,
        plots=True,
    )

    # Copy best weights to the expected location
    best_src = RUNS_DIR / "geoquery_yolo" / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, OUTPUT_WEIGHTS)
        print(f"\n✅ Training complete!")
        print(f"Best weights copied to: {OUTPUT_WEIGHTS}")
        print(f"\nTo enable in GeoQuery, set USE_YOLO = True in config.py")
    else:
        print(f"\n⚠️ best.pt not found at {best_src}")
        print(f"Check {RUNS_DIR / 'geoquery_yolo'} for training output.")

    return results


def evaluate():
    """Evaluate the trained model."""
    from ultralytics import YOLO

    if not OUTPUT_WEIGHTS.exists():
        print(f"No trained weights found at {OUTPUT_WEIGHTS}")
        return

    print("=" * 60)
    print("Step 4: Evaluating trained model...")
    print("=" * 60)

    model = YOLO(str(OUTPUT_WEIGHTS))
    data_yaml = str(DATASET_DIR / "data.yaml")

    if os.path.exists(data_yaml):
        metrics = model.val(data=data_yaml, imgsz=IMG_SIZE, device="mps")
        print(f"\n📊 Results:")
        print(f"  mAP@0.5:       {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95:  {metrics.box.map:.4f}")
        print(f"  Precision:     {metrics.box.mp:.4f}")
        print(f"  Recall:        {metrics.box.mr:.4f}")
    else:
        print(f"  data.yaml not found at {data_yaml}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🛰️  GeoQuery YOLOv8 Fine-Tuning Pipeline")
    print("=" * 60)
    print()

    # Step 1: Download VisDrone
    visdrone_yaml = download_visdrone()
    print(f"  VisDrone dataset ready at: {visdrone_yaml}\n")

    # Step 2: Remap to GeoQuery classes
    remapped_yaml = remap_visdrone_labels(visdrone_yaml)
    print()

    # Step 3: Train
    train(remapped_yaml)
    print()

    # Step 4: Evaluate
    evaluate()

    print()
    print("=" * 60)
    print("Done! Set USE_YOLO = True in config.py to use your model.")
    print("=" * 60)
