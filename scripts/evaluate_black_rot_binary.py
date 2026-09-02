"""Evaluate black-rot recognition on positive and negative natural images.

This diagnostic detects domain shortcuts: every prediction other than
``vl_black_rot`` is treated as a negative prediction, regardless of which of the
other PlantVillage classes the model selected.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

from dataset import eval_transforms
from train import build_model


ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class BinaryNaturalDataset(Dataset):
    def __init__(self, test_dir, positive_class, negative_classes, image_size):
        self.transform = eval_transforms(image_size)
        self.samples = []
        requested = [(positive_class, 1), *((name, 0) for name in negative_classes)]
        for source_class, target in requested:
            class_dir = test_dir / source_class
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing natural test class: {class_dir}")
            files = sorted(
                path for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not files:
                raise FileNotFoundError(f"No supported images in {class_dir}")
            self.samples.extend((path, target, source_class) for path in files)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target, source_class = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, target, source_class


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "processed" / "test"))
    parser.add_argument("--positive-class", default="vl_black_rot")
    parser.add_argument(
        "--negative-classes", nargs="+",
        default=["vl_downy_mildew", "vl_powdery_mildew"],
    )
    parser.add_argument("--model", choices=["resnet50", "mobilenet_v3"], default="resnet50")
    parser.add_argument("--output", default="")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--limit-images", type=int, default=0,
        help="Use only the first N samples for a smoke test; 0 evaluates all images.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    label_path = Path(args.label_map)
    label_map = json.loads(label_path.read_text(encoding="utf-8"))
    if args.positive_class not in label_map:
        raise ValueError(f"{args.positive_class!r} is absent from {label_path}")

    dataset = BinaryNaturalDataset(
        Path(args.data_dir), args.positive_class, args.negative_classes, args.image_size
    )
    if args.limit_images:
        dataset.samples = dataset.samples[:args.limit_images]
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = build_model(args.model, len(label_map), pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    positive_index = label_map[args.positive_class]

    targets, predictions, source_classes = [], [], []
    with torch.no_grad():
        for images, batch_targets, batch_sources in loader:
            predicted_classes = model(images.to(device)).argmax(1).cpu().tolist()
            predictions.extend(int(index == positive_index) for index in predicted_classes)
            targets.extend(batch_targets.tolist())
            source_classes.extend(batch_sources)

    if set(targets) != {0, 1}:
        raise RuntimeError(
            "Evaluation needs both positive and negative samples; check --data-dir "
            "and --negative-classes"
        )
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    per_source = {}
    for source_class in sorted(set(source_classes)):
        indices = [i for i, value in enumerate(source_classes) if value == source_class]
        predicted_positive = sum(predictions[i] for i in indices)
        per_source[source_class] = {
            "support": len(indices),
            "predicted_black_rot": predicted_positive,
            "predicted_black_rot_rate": safe_ratio(predicted_positive, len(indices)),
        }

    output = {
        "model": args.model,
        "checkpoint": str(checkpoint),
        "positive_class": args.positive_class,
        "negative_classes": args.negative_classes,
        "support": len(targets),
        "positive_support": Counter(targets)[1],
        "negative_support": Counter(targets)[0],
        "accuracy": accuracy_score(targets, predictions),
        "balanced_accuracy": balanced_accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall_sensitivity": recall_score(targets, predictions, zero_division=0),
        "specificity": safe_ratio(tn, tn + fp),
        "f1": f1_score(targets, predictions, zero_division=0),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_source_class": per_source,
    }

    print(f"device: {device}")
    print(f"samples: {len(targets)} ({Counter(targets)[1]} positive, {Counter(targets)[0]} negative)")
    print(f"accuracy:          {output['accuracy']:.4f}")
    print(f"balanced_accuracy: {output['balanced_accuracy']:.4f}")
    print(f"precision:         {output['precision']:.4f}")
    print(f"recall/sensitivity:{output['recall_sensitivity']:9.4f}")
    print(f"specificity:       {output['specificity']:.4f}")
    print(f"f1:                {output['f1']:.4f}")
    print(f"confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    for name, values in per_source.items():
        print(
            f"  {name}: {values['predicted_black_rot']}/{values['support']} "
            f"predicted as black rot ({values['predicted_black_rot_rate']:.2%})"
        )

    output_path = Path(args.output) if args.output else checkpoint.parent / "natural_binary_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
