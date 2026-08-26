"""Build a diagnosis dataset without the ambiguous vines_leaf/vines_grape labels.

It combines the seven disease classes produced from the original COCO dataset
with the additional grape-leaf dataset. Exact duplicates in the additional
dataset are removed before splitting, preventing the same image from leaking
across train/validation/test.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
EXTRA_CLASS_MAP = {
    "Healthy Leaves": "healthy_leaf",
    "Bacterial Rot": "vl_bacterial_spot",
    "Downey Mildew": "vl_downy_mildew",
    "Powdery Mildew": "vl_powdery_mildew",
}


def link_or_copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def unique_images(directory: Path):
    seen = set()
    result = []
    for path in sorted(p for p in directory.iterdir() if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).digest()
        if digest not in seen:
            seen.add(digest)
            result.append(path)
    return result


def split_images(images, val_fraction: float, test_fraction: float, seed: int):
    images = list(images)
    random.Random(seed).shuffle(images)
    n = len(images)
    n_test = round(n * test_fraction)
    n_val = round(n * val_fraction)
    return {
        "test": images[:n_test],
        "val": images[n_test:n_test + n_val],
        "train": images[n_test + n_val:],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--extra-data", default=str(ROOT / "Grapes Disease Dataset"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "diagnosis_processed"))
    parser.add_argument("--label-map", default=str(ROOT / "models" / "diagnosis_label_map.json"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_data = Path(args.base_data)
    extra_data = Path(args.extra_data)
    out_dir = Path(args.out_dir)
    label_path = Path(args.label_map)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; remove or choose another --out-dir")

    base_classes = sorted(
        p.name for p in (base_data / "train").iterdir()
        if p.is_dir() and p.name not in {"vines_leaf", "vines_grape"}
    )
    classes = sorted(set(base_classes) | set(EXTRA_CLASS_MAP.values()))

    # Preserve the original image-grouped splits for all COCO-derived crops.
    for split in SPLITS:
        for class_name in base_classes:
            source_dir = base_data / split / class_name
            for source in source_dir.iterdir():
                if source.is_file():
                    link_or_copy(source, out_dir / split / class_name / f"coco_{source.name}")

    # Deduplicate each external class before its deterministic 80/10/10 split.
    extra_stats = {}
    for offset, (source_name, class_name) in enumerate(EXTRA_CLASS_MAP.items()):
        source_dir = extra_data / source_name
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing additional class directory: {source_dir}")
        all_files = [p for p in source_dir.iterdir() if p.is_file()]
        unique = unique_images(source_dir)
        split_files = split_images(unique, args.val_fraction, args.test_fraction, args.seed + offset)
        extra_stats[class_name] = {
            "source": len(all_files),
            "unique": len(unique),
            "duplicates_removed": len(all_files) - len(unique),
        }
        for split, files in split_files.items():
            for index, source in enumerate(files):
                suffix = source.suffix.lower() or ".jpg"
                destination = out_dir / split / class_name / f"extra_{source_name.lower().replace(' ', '_')}_{index:05d}{suffix}"
                link_or_copy(source, destination)

    # ImageFolder requires every class directory to exist in every split.
    for split in SPLITS:
        for class_name in classes:
            (out_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    label_map = {name: index for index, name in enumerate(classes)}
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(label_map, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote diagnosis dataset with {len(classes)} classes to {out_dir}")
    for split in SPLITS:
        total = sum(1 for p in (out_dir / split).glob("*/*") if p.is_file())
        print(f"  {split}: {total} images")
    print("Additional dataset deduplication:")
    for name, stats in extra_stats.items():
        print(f"  {name}: {stats}")
    print(f"Wrote label map to {label_path}")


if __name__ == "__main__":
    main()
