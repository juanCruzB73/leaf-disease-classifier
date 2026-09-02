"""Build a hybrid experiment comparable to the PlantVillage baseline.

The four-class PlantVillage validation and test splits are copied unchanged.
Natural images are added only to the training split when their disease label is
an explicit semantic match. A separate natural-domain test is also produced.

With the datasets currently available, only ``vl_black_rot`` is a safe match.
Other diseases are deliberately rejected instead of being relabelled.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
DEFAULT_NATURAL_MAP = {"vl_black_rot": "vl_black_rot"}


def link_or_copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def image_files(directory: Path):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def copy_unique(files, destination: Path, prefix: str, seen_hashes: set):
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
    return copied, duplicates


def parse_class_map(values):
    if not values:
        return DEFAULT_NATURAL_MAP
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --natural-class {value!r}; use SOURCE=TARGET")
        source, target = value.split("=", 1)
        result[source] = target
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controlled-data", default=str(ROOT / "data" / "controlled_processed")
    )
    parser.add_argument("--natural-data", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "hybrid_comparable"))
    parser.add_argument(
        "--natural-test-dir", default=str(ROOT / "data" / "natural_comparable_test")
    )
    parser.add_argument(
        "--source-label-map", default=str(ROOT / "models" / "controlled_label_map.json")
    )
    parser.add_argument(
        "--label-map", default=str(ROOT / "models" / "hybrid_comparable_label_map.json")
    )
    parser.add_argument(
        "--natural-class", action="append", default=[], metavar="SOURCE=TARGET",
        help="Explicit equivalent class mapping; repeat for additional verified classes.",
    )
    args = parser.parse_args()

    controlled = Path(args.controlled_data)
    natural = Path(args.natural_data)
    out_dir = Path(args.out_dir)
    natural_test_dir = Path(args.natural_test_dir)
    if out_dir.exists() or natural_test_dir.exists():
        raise FileExistsError(
            "Output already exists; remove it or choose different --out-dir and "
            "--natural-test-dir values"
        )

    label_map = json.loads(Path(args.source_label_map).read_text(encoding="utf-8"))
    expected_indices = set(range(len(label_map)))
    if set(label_map.values()) != expected_indices:
        raise ValueError("The controlled label map must contain consecutive indices")

    natural_map = parse_class_map(args.natural_class)
    unknown_targets = set(natural_map.values()) - set(label_map)
    if unknown_targets:
        raise ValueError(f"Natural mappings target unknown controlled classes: {unknown_targets}")

    stats = {"controlled": {}, "natural_train": {}, "natural_test": {}}
    seen_train = {class_name: set() for class_name in label_map}

    # Keep validation and test identical to the controlled baseline. This is the
    # key constraint that makes both runs directly comparable.
    for split in SPLITS:
        for class_name in label_map:
            source_dir = controlled / split / class_name
            if not source_dir.is_dir():
                raise FileNotFoundError(f"Missing controlled class directory: {source_dir}")
            files = image_files(source_dir)
            seen = seen_train[class_name] if split == "train" else set()
            copied, duplicates = copy_unique(
                files, out_dir / split / class_name, f"controlled_{class_name}", seen
            )
            stats["controlled"][f"{split}/{class_name}"] = {
                "copied": copied, "duplicates_skipped": duplicates
            }

    # Add only verified natural-domain equivalents to hybrid training. Natural
    # test images remain isolated and can evaluate both baseline and hybrid runs.
    for source_class, target_class in natural_map.items():
        train_source = natural / "train" / source_class
        test_source = natural / "test" / source_class
        if not train_source.is_dir() or not test_source.is_dir():
            raise FileNotFoundError(
                f"Natural class {source_class!r} must exist in train and test below {natural}"
            )
        copied, duplicates = copy_unique(
            image_files(train_source), out_dir / "train" / target_class,
            f"natural_{source_class}", seen_train[target_class],
        )
        stats["natural_train"][source_class] = {
            "target": target_class, "copied": copied, "duplicates_skipped": duplicates
        }
        copied, duplicates = copy_unique(
            image_files(test_source), natural_test_dir / "test" / target_class,
            f"natural_{source_class}", set(),
        )
        stats["natural_test"][source_class] = {
            "target": target_class, "copied": copied, "duplicates_skipped": duplicates
        }

    label_path = Path(args.label_map)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    manifest = {
        "label_map": label_map,
        "natural_class_map": natural_map,
        "warning": (
            "Natural-domain coverage is currently partial. Only compare per-class "
            "natural metrics for classes represented in natural_test."
        ),
        "stats": stats,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote comparable hybrid training data to {out_dir}")
    print(f"Wrote isolated natural test data to {natural_test_dir}")
    print(f"Natural mappings: {natural_map}")
    print("WARNING: natural test coverage is partial; inspect manifest.json")


if __name__ == "__main__":
    main()
