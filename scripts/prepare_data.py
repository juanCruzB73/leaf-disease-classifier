"""Build a classification dataset from the LDD (Leaf Disease Detection) COCO annotations.

The uploaded dataset (10573036.zip) is a COCO-style instance segmentation dataset for
grapevine diseases, not a folder-per-class classification dataset. This script:

  1. Extracts the nested zips into data/raw/ (idempotent).
  2. Reads annotations/train.json and annotations/validation.json.
  3. Drops categories that describe the grape bunch/fruit (FRUIT_CATEGORIES) and
     disease categories with too few annotated instances to train on reliably —
     this classifier only targets leaves.
  4. Crops each kept annotation's bounding box (with padding) out of its source image,
     resizes it, and writes it under data/processed/<split>/<class_name>/.
  5. train.json becomes the training pool. validation.json is split (grouped by image,
     so crops from the same photo never straddle both sides) into val/test.
  6. Writes models/label_map.json mapping class name -> index, shared by train.py and the API.
"""
import argparse
import json
import random
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

# Categories that describe the grape bunch/fruit rather than the leaf. The
# classifier only targets leaves, so these are dropped before the min-samples
# filter, regardless of how many instances they have.
FRUIT_CATEGORIES = {
    "vg_black_rot",
    "vg_downy_mildew",
    "vg_grey_mould",
    "vg_powdery_mildew",
    "vines_grape",
    "carie_bianca_grappolo",
}


def extract_if_needed(zip_path: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as outer:
        for name in outer.namelist():
            inner_dest = dest / name
            if not inner_dest.exists():
                outer.extract(name, dest)
            if name.endswith(".zip"):
                marker = dest / (Path(name).stem + ".extracted")
                if not marker.exists():
                    with zipfile.ZipFile(inner_dest) as inner:
                        inner.extractall(dest)
                    marker.touch()


def load_annotations(raw_dir: Path):
    with open(raw_dir / "annotations" / "train.json", encoding="utf-8") as f:
        train = json.load(f)
    with open(raw_dir / "annotations" / "validation.json", encoding="utf-8") as f:
        val = json.load(f)
    return train, val


def category_counts(coco):
    counts = {}
    for ann in coco["annotations"]:
        counts[ann["category_id"]] = counts.get(ann["category_id"], 0) + 1
    return counts


def kept_categories(train, val, min_samples):
    cats = {c["id"]: c["name"] for c in train["categories"]}
    train_counts = category_counts(train)
    val_counts = category_counts(val)
    kept, dropped = {}, {}
    for cid, name in cats.items():
        if name in FRUIT_CATEGORIES:
            continue
        total = train_counts.get(cid, 0) + val_counts.get(cid, 0)
        if total >= min_samples:
            kept[cid] = name
        else:
            dropped[name] = total
    return kept, dropped


def crop_and_save(image_path: Path, bbox, padding, size, out_path: Path):
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    pad_x, pad_y = w * padding, h * padding
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(img.width, x + w + pad_x)
        bottom = min(img.height, y + h + pad_y)
        if right <= left or bottom <= top:
            return False
        crop = img.crop((left, top, right, bottom)).resize((size, size), Image.BILINEAR)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path, quality=95)
        return True


def export_split(coco, image_ids, kept_categories, images_dir, out_dir, split_name, padding, size):
    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in kept_categories and ann["image_id"] in image_ids:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

    written, skipped = 0, 0
    for image_id, anns in anns_by_image.items():
        image_info = images_by_id[image_id]
        image_path = images_dir / image_info["file_name"]
        if not image_path.exists():
            skipped += len(anns)
            continue
        for ann in anns:
            class_name = kept_categories[ann["category_id"]]
            out_path = out_dir / split_name / class_name / f"{image_id}_{ann['id']}.jpg"
            if crop_and_save(image_path, ann["bbox"], padding, size, out_path):
                written += 1
            else:
                skipped += 1
    if skipped:
        print(f"  ({skipped} annotations skipped: missing image or degenerate bbox)")
    return written


def split_image_ids(coco, kept_category_ids, val_fraction, seed):
    """Group by image so crops from one photo don't leak across val/test."""
    from sklearn.model_selection import train_test_split

    dominant_class = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in kept_category_ids:
            continue
        dominant_class.setdefault(ann["image_id"], ann["category_id"])

    image_ids = list(dominant_class.keys())
    labels = [dominant_class[i] for i in image_ids]

    # collapse classes with a single image so stratify doesn't error out
    from collections import Counter

    counts = Counter(labels)
    strat = [l if counts[l] > 1 else -1 for l in labels]

    val_ids, test_ids = train_test_split(
        image_ids, test_size=val_fraction, random_state=seed, stratify=strat
    )
    return set(val_ids), set(test_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default=str(ROOT / "10573036.zip"))
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--min-samples", type=int, default=150,
                         help="Drop disease classes with fewer than this many combined instances.")
    parser.add_argument("--padding", type=float, default=0.15,
                         help="Fractional bbox padding before crop.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--test-fraction", type=float, default=0.5,
                         help="Fraction of validation.json images held out as the final test set "
                              "(rest becomes the val set used for model selection).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    zip_path = Path(args.zip)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    annotations_ready = (
        (raw_dir / "annotations" / "train.json").exists()
        and (raw_dir / "annotations" / "validation.json").exists()
        and (raw_dir / "images").is_dir()
    )
    if zip_path.exists():
        print(f"Extracting {zip_path} -> {raw_dir}")
        extract_if_needed(zip_path, raw_dir)
    elif annotations_ready:
        print(f"{zip_path} not found; using the already extracted dataset in {raw_dir}")
    else:
        raise FileNotFoundError(
            f"Neither {zip_path} nor a complete extracted dataset in {raw_dir} was found."
        )
    images_dir = raw_dir / "images"

    train_coco, val_coco = load_annotations(raw_dir)

    kept, dropped = kept_categories(train_coco, val_coco, args.min_samples)
    print(f"\nKept {len(kept)} classes (>= {args.min_samples} combined instances):")
    for name in sorted(kept.values()):
        print(f"  - {name}")
    print(f"\nDropped {len(dropped)} classes (too few instances to train on reliably):")
    for name, total in sorted(dropped.items(), key=lambda kv: kv[1]):
        print(f"  - {name}: {total}")

    kept_ids = set(kept.keys())

    print("\nExporting train split (from train.json)...")
    all_train_ids = {im["id"] for im in train_coco["images"]}
    n_train = export_split(train_coco, all_train_ids, kept, images_dir, out_dir,
                            "train", args.padding, args.image_size)
    print(f"  wrote {n_train} crops")

    print("\nSplitting validation.json into val/test (grouped by image)...")
    val_ids, test_ids = split_image_ids(val_coco, kept_ids, args.test_fraction, args.seed)
    print(f"  {len(val_ids)} images -> val, {len(test_ids)} images -> test")

    n_val = export_split(val_coco, val_ids, kept, images_dir, out_dir,
                          "val", args.padding, args.image_size)
    n_test = export_split(val_coco, test_ids, kept, images_dir, out_dir,
                           "test", args.padding, args.image_size)
    print(f"  wrote {n_val} crops to val, {n_test} crops to test")

    label_map = {name: idx for idx, name in enumerate(sorted(kept.values()))}
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"\nWrote label map ({len(label_map)} classes) -> {models_dir / 'label_map.json'}")


if __name__ == "__main__":
    main()
