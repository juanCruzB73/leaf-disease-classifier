"""Train and validate a single classification model (ResNet50 or MobileNetV3-Large)
on the processed crop dataset. Run once per architecture, then compare their
models/<name>/test_metrics.json (produced by evaluate.py) to pick a winner.

Example:
    python scripts/train.py --model resnet50 --epochs 8 --finetune-epochs 6
    python scripts/train.py --model mobilenet_v3 --epochs 8 --finetune-epochs 6
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import f1_score

from dataset import load_label_map, make_dataset

ROOT = Path(__file__).resolve().parent.parent


def build_model(name: str, num_classes: int, pretrained: bool = True):
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        backbone_params = [p for n, p in net.named_parameters() if not n.startswith("fc.")]
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head_params = list(net.fc.parameters())
    elif name == "mobilenet_v3":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)
        backbone_params = [p for n, p in net.named_parameters() if not n.startswith("classifier.3")]
        in_features = net.classifier[3].in_features
        net.classifier[3] = nn.Linear(in_features, num_classes)
        head_params = list(net.classifier[3].parameters())
    else:
        raise ValueError(f"unknown model {name!r}")
    return net, backbone_params, head_params


def class_weights(dataset, num_classes, device):
    counts = Counter(dataset.targets)
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    predictions, targets = [], []
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(1)
            correct += (predicted == labels).sum().item()
            n += images.size(0)
            predictions.extend(predicted.detach().cpu().tolist())
            targets.extend(labels.detach().cpu().tolist())
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    return total_loss / n, correct / n, macro_f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["resnet50", "mobilenet_v3"], required=True)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--models-dir", default=str(ROOT / "models"))
    parser.add_argument("--label-map", default="",
                        help="Class map JSON (default: <models-dir>/label_map.json).")
    parser.add_argument("--run-name", default="",
                        help="Artifact subdirectory (default: the architecture name).")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not download/use ImageNet weights (mainly for offline smoke tests).")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5, help="Head-only warmup epochs (backbone frozen).")
    parser.add_argument("--finetune-epochs", type=int, default=5, help="Full fine-tune epochs after warmup.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-batches", type=int, default=0,
                         help="If >0, cap batches/epoch (smoke test / quick iteration).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    label_map_path = Path(args.label_map) if args.label_map else models_dir / "label_map.json"
    with open(label_map_path, encoding="utf-8") as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    train_ds = make_dataset(data_dir, "train", label_map, args.image_size)
    val_ds = make_dataset(data_dir, "val", label_map, args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    if args.limit_batches:
        from itertools import islice

        class _Capped:
            def __init__(self, loader, n):
                self.loader, self.n = loader, n

            def __iter__(self):
                return islice(self.loader, self.n)

        train_loader = _Capped(train_loader, args.limit_batches)
        val_loader = _Capped(val_loader, max(1, args.limit_batches // 4))

    model, backbone_params, head_params = build_model(
        args.model, num_classes, pretrained=not args.no_pretrained
    )
    model.to(device)

    weights = class_weights(train_ds, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    out_dir = models_dir / (args.run_name or args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil_label_map = out_dir / "label_map.json"
    with open(shutil_label_map, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    history = []
    best_val_acc = -1.0
    best_val_macro_f1 = -1.0

    def maybe_save_best(val_acc, val_macro_f1, stage, epoch):
        nonlocal best_val_acc, best_val_macro_f1
        if val_macro_f1 > best_val_macro_f1:
            best_val_acc = val_acc
            best_val_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  -> new best ({stage} epoch {epoch}, val_macro_f1={val_macro_f1:.4f}, "
                  f"val_acc={val_acc:.4f}), saved best_model.pt")

    # Stage 1: warmup — backbone frozen, only the new head trains.
    for p in backbone_params:
        p.requires_grad = False
    optimizer = torch.optim.Adam(head_params, lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_macro_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_macro_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        dt = time.time() - t0
        print(f"[warmup {epoch}/{args.epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"train_macro_f1={train_macro_f1:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f} ({dt:.1f}s)")
        history.append({"stage": "warmup", "epoch": epoch, "train_loss": train_loss,
                         "train_acc": train_acc, "train_macro_f1": train_macro_f1,
                         "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_macro_f1})
        maybe_save_best(val_acc, val_macro_f1, "warmup", epoch)

    # Stage 2: fine-tune — unfreeze the backbone, train everything at a lower LR.
    for p in backbone_params:
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
    for epoch in range(1, args.finetune_epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_macro_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_macro_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        dt = time.time() - t0
        print(f"[finetune {epoch}/{args.finetune_epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"train_macro_f1={train_macro_f1:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f} ({dt:.1f}s)")
        history.append({"stage": "finetune", "epoch": epoch, "train_loss": train_loss,
                         "train_acc": train_acc, "train_macro_f1": train_macro_f1,
                         "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_macro_f1})
        maybe_save_best(val_acc, val_macro_f1, "finetune", epoch)

    torch.save(model.state_dict(), out_dir / "last_model.pt")
    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "best_val_acc": best_val_acc,
                   "best_val_macro_f1": best_val_macro_f1, "history": history}, f, indent=2)
    print(f"\nDone. best_val_macro_f1={best_val_macro_f1:.4f}, "
          f"best_val_acc={best_val_acc:.4f}. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
