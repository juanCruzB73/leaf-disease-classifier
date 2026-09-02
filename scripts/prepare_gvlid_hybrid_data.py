"""Build a four-class PlantVillage + GVLiD comparable hybrid experiment.

The controlled PlantVillage validation and test splits remain byte-identical to
the baseline. Deduplicated GVLiD full-leaf images are split 70/15/15; only their
training split is added to hybrid training. Natural val/test remain isolated.
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
GVLID_MAP = {
    "healthy": "healthy_leaf",
    "Black rot": "vl_black_rot",
    "esca": "vl_black_measles",
    "leaf blight": "vl_leaf_blight",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def link_or_copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def image_files(directory: Path):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing GVLiD class directory: {directory}")
    files = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No supported images in {directory}")
    return files


def unique_images(files):
    seen = set()
    unique = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).digest()
        if digest not in seen:
            seen.add(digest)
            unique.append(path)
    return unique


def split_images(images, val_fraction, test_fraction, seed):
    images = list(images)
    random.Random(seed).shuffle(images)
    n_test = round(len(images) * test_fraction)
    n_val = round(len(images) * val_fraction)
    return {
        "test": images[:n_test],
        "val": images[n_test:n_test + n_val],
        "train": images[n_test + n_val:],
    }


def copy_files(files, destination, prefix):
    for index, source in enumerate(files):
        target = destination / f"{prefix}_{index:06d}{source.suffix.lower()}"
        link_or_copy(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controlled-data", default=str(ROOT / "data" / "controlled_processed")
    )
    parser.add_argument("--gvlid-data", default=str(ROOT / "dataset hibrido"))
    parser.add_argument(
        "--hybrid-out", default=str(ROOT / "data" / "gvlid_hybrid_processed")
    )
    parser.add_argument(
        "--natural-eval-out", default=str(ROOT / "data" / "gvlid_natural_eval")
    )
    parser.add_argument(
        "--source-label-map", default=str(ROOT / "models" / "controlled_label_map.json")
    )
    parser.add_argument(
        "--label-map", default=str(ROOT / "models" / "gvlid_hybrid_label_map.json")
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.val_fraction < 0 or args.test_fraction < 0:
        parser.error("split fractions cannot be negative")
    if args.val_fraction + args.test_fraction >= 1:
        parser.error("val-fraction + test-fraction must be less than 1")

    controlled_data = Path(args.controlled_data)
    gvlid_data = Path(args.gvlid_data)
    hybrid_out = Path(args.hybrid_out)
    natural_eval_out = Path(args.natural_eval_out)
    existing = [str(path) for path in (hybrid_out, natural_eval_out) if path.exists()]
    if existing:
        raise FileExistsError("Output already exists: " + ", ".join(existing))

    label_map = json.loads(Path(args.source_label_map).read_text(encoding="utf-8"))
    if set(GVLID_MAP.values()) != set(label_map):
        raise ValueError(
            "GVLiD and controlled label maps must contain the same four classes; "
            f"controlled={sorted(label_map)}, GVLiD={sorted(GVLID_MAP.values())}"
        )

    # Reproduce the controlled baseline exactly in all three splits.
    controlled_stats = {}
    for split in SPLITS:
        for class_name in label_map:
            files = image_files(controlled_data / split / class_name)
            copy_files(
                files, hybrid_out / split / class_name,
                f"controlled_{class_name}",
            )
            controlled_stats[f"{split}/{class_name}"] = len(files)

    # Audit all classes together before splitting. Identical content carrying
    # different labels is excluded from every class instead of choosing one.
    natural_sources = []
    hash_targets = {}
    for source_class, target_class in sorted(GVLID_MAP.items()):
        files = image_files(gvlid_data / source_class)
        unique = unique_images(files)
        natural_sources.append((source_class, target_class, files, unique))
        for path in unique:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hash_targets.setdefault(digest, set()).add(target_class)
    conflicting_hashes = {
        digest for digest, targets in hash_targets.items() if len(targets) > 1
    }

    natural_stats = {}
    for offset, (source_class, target_class, files, exact_unique) in enumerate(natural_sources):
        unique = [
            path for path in exact_unique
            if hashlib.sha256(path.read_bytes()).hexdigest() not in conflicting_hashes
        ]
        split_files = split_images(
            unique, args.val_fraction, args.test_fraction, args.seed + offset
        )
        natural_stats[target_class] = {
            "source_class": source_class,
            "raw": len(files),
            "exact_unique": len(exact_unique),
            "exact_duplicates_removed": len(files) - len(exact_unique),
            "cross_class_conflicts_removed": len(exact_unique) - len(unique),
            "usable_unique": len(unique),
            **{split: len(values) for split, values in split_files.items()},
        }

        copy_files(
            split_files["train"], hybrid_out / "train" / target_class,
            f"gvlid_{target_class}",
        )
        for split in ("val", "test"):
            copy_files(
                split_files[split], natural_eval_out / split / target_class,
                f"gvlid_{target_class}",
            )

    for split in ("val", "test"):
        for class_name in label_map:
            (natural_eval_out / split / class_name).mkdir(parents=True, exist_ok=True)

    label_path = Path(args.label_map)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    manifest = {
        "source": {
            "name": "GVLiD: GrapeVine Leaf identification of the Diseases",
            "doi": "10.17632/wkymf8bhcg.5",
            "license": "CC BY 4.0",
        },
        "label_map": label_map,
        "gvlid_class_map": GVLID_MAP,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "controlled_stats": controlled_stats,
        "natural_stats": natural_stats,
        "cross_class_conflicting_hashes_removed": len(conflicting_hashes),
        "constraints": {
            "controlled_val_identical_to_baseline": True,
            "controlled_test_identical_to_baseline": True,
            "exact_duplicates_removed_before_natural_split": True,
            "natural_val_and_test_isolated_from_training": True,
        },
    }
    for output in (hybrid_out, natural_eval_out):
        output.mkdir(parents=True, exist_ok=True)
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"Wrote four-class hybrid dataset: {hybrid_out}")
    print(f"Wrote isolated GVLiD val/test: {natural_eval_out}")
    print(f"Wrote label map: {label_path}")
    for class_name, values in natural_stats.items():
        print(f"  {class_name}: {values}")


if __name__ == "__main__":
    main()
