"""Prepare controlled and balanced-hybrid binary black-rot experiments.

Outputs share the same controlled validation/test splits. The hybrid training
set additionally contains natural black-rot positives and natural foliar-disease
negatives. A separate natural binary test is built for domain-gap evaluation.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
BINARY_LABEL_MAP = {"not_black_rot": 0, "black_rot": 1}
CONTROLLED_MAP = {
    "vl_black_rot": "black_rot",
    "healthy_leaf": "not_black_rot",
    "vl_black_measles": "not_black_rot",
    "vl_leaf_blight": "not_black_rot",
}
NATURAL_MAP = {
    "vl_black_rot": "black_rot",
    "vl_downy_mildew": "not_black_rot",
    "vl_powdery_mildew": "not_black_rot",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(directory: Path):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing source class directory: {directory}")
    files = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No supported images in {directory}")
    return files


def link_or_copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def add_images(files, destination, prefix, seen_hashes):
    copied = duplicates = 0
    for source in files:
        digest = hashlib.sha256(source.read_bytes()).digest()
        if digest in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(digest)
        target = destination / f"{prefix}_{copied:06d}{source.suffix.lower()}"
        link_or_copy(source, target)
        copied += 1
    return {"source": len(files), "copied": copied, "duplicates_skipped": duplicates}


def ensure_empty_outputs(paths):
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Output paths already exist; remove them or choose alternatives: "
            + ", ".join(existing)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controlled-data", default=str(ROOT / "data" / "controlled_processed")
    )
    parser.add_argument("--natural-data", default=str(ROOT / "data" / "processed"))
    parser.add_argument(
        "--controlled-out", default=str(ROOT / "data" / "binary_controlled")
    )
    parser.add_argument("--hybrid-out", default=str(ROOT / "data" / "binary_hybrid"))
    parser.add_argument(
        "--natural-test-out", default=str(ROOT / "data" / "binary_natural_test")
    )
    parser.add_argument(
        "--label-map", default=str(ROOT / "models" / "binary_label_map.json")
    )
    args = parser.parse_args()

    controlled_data = Path(args.controlled_data)
    natural_data = Path(args.natural_data)
    controlled_out = Path(args.controlled_out)
    hybrid_out = Path(args.hybrid_out)
    natural_test_out = Path(args.natural_test_out)
    ensure_empty_outputs((controlled_out, hybrid_out, natural_test_out))

    stats = {"controlled": {}, "hybrid_natural_train": {}, "natural_test": {}}

    # Both experiments receive byte-identical controlled splits. Separate hash
    # sets prevent duplicate images within each binary target and split.
    for split in SPLITS:
        controlled_seen = {name: set() for name in BINARY_LABEL_MAP}
        hybrid_seen = {name: set() for name in BINARY_LABEL_MAP}
        for source_class, target_class in CONTROLLED_MAP.items():
            files = image_files(controlled_data / split / source_class)
            key = f"{split}/{source_class}->{target_class}"
            stats["controlled"][key] = add_images(
                files,
                controlled_out / split / target_class,
                f"controlled_{source_class}",
                controlled_seen[target_class],
            )
            add_images(
                files,
                hybrid_out / split / target_class,
                f"controlled_{source_class}",
                hybrid_seen[target_class],
            )

        # Natural images belong only in hybrid training. Validation and test
        # therefore remain exactly equivalent to the controlled baseline.
        if split == "train":
            for source_class, target_class in NATURAL_MAP.items():
                files = image_files(natural_data / "train" / source_class)
                key = f"{source_class}->{target_class}"
                stats["hybrid_natural_train"][key] = add_images(
                    files,
                    hybrid_out / "train" / target_class,
                    f"natural_{source_class}",
                    hybrid_seen[target_class],
                )

    natural_seen = {name: set() for name in BINARY_LABEL_MAP}
    for source_class, target_class in NATURAL_MAP.items():
        files = image_files(natural_data / "test" / source_class)
        key = f"{source_class}->{target_class}"
        stats["natural_test"][key] = add_images(
            files,
            natural_test_out / "test" / target_class,
            f"natural_{source_class}",
            natural_seen[target_class],
        )

    label_path = Path(args.label_map)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(BINARY_LABEL_MAP, indent=2), encoding="utf-8")

    manifest = {
        "task": "black_rot_vs_not_black_rot",
        "label_map": BINARY_LABEL_MAP,
        "controlled_class_map": CONTROLLED_MAP,
        "natural_class_map": NATURAL_MAP,
        "constraints": {
            "controlled_val_identical": True,
            "controlled_test_identical": True,
            "natural_images_in_hybrid_train_only": True,
            "natural_test_isolated": True,
        },
        "stats": stats,
    }
    for output in (controlled_out, hybrid_out, natural_test_out):
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    print(f"Wrote controlled binary dataset: {controlled_out}")
    print(f"Wrote balanced hybrid binary dataset: {hybrid_out}")
    print(f"Wrote shared natural binary test: {natural_test_out}")
    print(f"Wrote label map: {label_path}")
    for name, output in (
        ("controlled train", controlled_out / "train"),
        ("hybrid train", hybrid_out / "train"),
        ("controlled test", controlled_out / "test"),
        ("natural test", natural_test_out / "test"),
    ):
        counts = {
            target: len(image_files(output / target)) for target in BINARY_LABEL_MAP
        }
        print(f"  {name}: {counts}")


if __name__ == "__main__":
    main()
