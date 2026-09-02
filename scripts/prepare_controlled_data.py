"""Prepare classification data using only the controlled PlantVillage dataset."""

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_HANDLE = "zienabesam/grape-plant-from-plant-village-dataset"
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ALIASES = {
    "black_rot": "vl_black_rot",
    "black_measles": "vl_black_measles",
    "esca": "vl_black_measles",
    "leaf_blight": "vl_leaf_blight",
    "isariopsis_leaf_spot": "vl_leaf_blight",
    "healthy": "healthy_leaf",
}


def link_or_copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def normalized_name(name: str) -> str:
    return "_".join(name.lower().replace("___", "_").replace("-", "_").split())


def canonical_class(directory_name: str) -> str:
    normalized = normalized_name(directory_name)
    for alias, class_name in CLASS_ALIASES.items():
        if alias in normalized:
            return class_name
    raise ValueError(f"Unknown PlantVillage grape class directory: {directory_name!r}")


def contains_images(directory: Path) -> bool:
    return any(
        p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        for p in directory.iterdir()
    )


def find_class_dirs(dataset_dir: Path) -> dict:
    """Discover PlantVillage's four class folders, regardless of wrapper folders."""
    result = {}
    unknown = []
    for directory in sorted(p for p in dataset_dir.rglob("*") if p.is_dir()):
        if not contains_images(directory):
            continue
        try:
            class_name = canonical_class(directory.name)
        except ValueError:
            unknown.append(directory)
            continue
        if class_name in result:
            raise ValueError(
                f"More than one source directory maps to {class_name}: "
                f"{result[class_name]} and {directory}"
            )
        result[class_name] = directory
    if len(result) != 4:
        discovered = ", ".join(p.name for p in unknown) or "none"
        raise FileNotFoundError(
            f"Expected 4 PlantVillage grape classes below {dataset_dir}, found "
            f"{sorted(result)}. Unrecognized image folders: {discovered}"
        )
    return result


def unique_images(directory: Path):
    seen = set()
    result = []
    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).digest()
        if digest not in seen:
            seen.add(digest)
            result.append(path)
    return files, result


def split_images(images, val_fraction: float, test_fraction: float, seed: int):
    images = list(images)
    random.Random(seed).shuffle(images)
    n_test = round(len(images) * test_fraction)
    n_val = round(len(images) * val_fraction)
    return {
        "test": images[:n_test],
        "val": images[n_test:n_test + n_val],
        "train": images[n_test + n_val:],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-data", default="",
        help="Dataset already downloaded. If omitted, kagglehub downloads it.",
    )
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "controlled_processed"))
    parser.add_argument(
        "--label-map", default=str(ROOT / "models" / "controlled_label_map.json")
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.val_fraction < 0 or args.test_fraction < 0:
        parser.error("split fractions cannot be negative")
    if args.val_fraction + args.test_fraction >= 1:
        parser.error("val-fraction + test-fraction must be less than 1")

    if args.source_data:
        dataset_dir = Path(args.source_data).expanduser().resolve()
    else:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError("Install dependencies: pip install -r requirements.txt") from exc
        dataset_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))

    class_dirs = find_class_dirs(dataset_dir)
    out_dir = Path(args.out_dir)
    label_path = Path(args.label_map)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; remove it or choose another --out-dir")

    stats = {}
    for offset, (class_name, source_dir) in enumerate(sorted(class_dirs.items())):
        all_files, unique = unique_images(source_dir)
        if not unique:
            raise FileNotFoundError(f"No supported images in {source_dir}")
        files_by_split = split_images(
            unique, args.val_fraction, args.test_fraction, args.seed + offset
        )
        stats[class_name] = {
            "source": len(all_files), "unique": len(unique),
            "duplicates_removed": len(all_files) - len(unique),
        }
        for split, files in files_by_split.items():
            for index, source in enumerate(files):
                target = out_dir / split / class_name / f"{class_name}_{index:05d}{source.suffix.lower()}"
                link_or_copy(source, target)

    classes = sorted(class_dirs)
    for split in SPLITS:
        for class_name in classes:
            (out_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    label_map = {name: index for index, name in enumerate(classes)}
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(label_map, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Kaggle source: {DATASET_HANDLE}")
    print(f"Downloaded dataset: {dataset_dir}")
    print(f"Wrote controlled dataset with {len(classes)} classes to {out_dir}")
    for split in SPLITS:
        total = sum(1 for p in (out_dir / split).glob("*/*") if p.is_file())
        print(f"  {split}: {total} images")
    for name, class_stats in stats.items():
        print(f"  {name}: {class_stats}")
    print(f"Wrote label map to {label_path}")


if __name__ == "__main__":
    main()
