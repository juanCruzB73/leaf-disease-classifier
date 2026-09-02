"""Evaluate a trained model's best_model.pt checkpoint on the held-out test split.

Writes models/<name>/test_metrics.json with accuracy, per-class precision/recall/F1,
macro/weighted F1 and the confusion matrix.

Example:
    python scripts/evaluate.py --model resnet50
    python scripts/evaluate.py --model mobilenet_v3
"""
import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import load_label_map, make_dataset
from train import build_model

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["resnet50", "mobilenet_v3"], required=True)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "controlled_processed"))
    parser.add_argument("--models-dir", default=str(ROOT / "models"))
    parser.add_argument("--label-map", default="",
                        help="Class map JSON (default: <models-dir>/controlled_label_map.json).")
    parser.add_argument("--run-name", default="",
                        help="Checkpoint subdirectory (default: the architecture name).")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(args.models_dir)
    label_map_path = Path(args.label_map) if args.label_map else models_dir / "controlled_label_map.json"
    with open(label_map_path, encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_class = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    test_ds = make_dataset(Path(args.data_dir), "test", label_map, args.image_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model, _, _ = build_model(args.model, num_classes, pretrained=False)
    run_name = args.run_name or args.model
    ckpt_path = models_dir / run_name / "best_model.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    class_names = [idx_to_class[i] for i in range(num_classes)]
    report = classification_report(all_labels, all_preds, labels=list(range(num_classes)),
                                    target_names=class_names,
                                    output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    print(classification_report(all_labels, all_preds, labels=list(range(num_classes)),
                                target_names=class_names, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(" " * 20 + " ".join(f"{n[:6]:>6s}" for n in class_names))
    for name, row in zip(class_names, cm):
        print(f"{name:20s} " + " ".join(f"{v:6d}" for v in row))

    out = {
        "model": args.model,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {name: report[name] for name in class_names},
        "confusion_matrix": cm.tolist(),
        "class_order": class_names,
    }
    out_path = models_dir / run_name / "test_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
