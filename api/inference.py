"""Loads the winning trained model once and exposes a predict() function.

Which model/checkpoint to serve is controlled by two env vars so the API doesn't
need code changes once train.py + evaluate.py pick a winner:

    API_MODEL_NAME  - "resnet50" or "mobilenet_v3" (default: mobilenet_v3)
    API_MODEL_PATH  - path to the .pt checkpoint (default: models/<API_MODEL_NAME>/best_model.pt)
"""
import io
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from train import build_model  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

MODEL_NAME = os.environ.get("API_MODEL_NAME", "mobilenet_v3")
MODEL_PATH = Path(os.environ.get("API_MODEL_PATH", ROOT / "models" / MODEL_NAME / "best_model.pt"))
LABEL_MAP_PATH = ROOT / "models" / "label_map.json"

_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_model = None
_idx_to_class = None


def load_model():
    global _model, _idx_to_class
    if _model is not None:
        return

    if not LABEL_MAP_PATH.exists():
        raise RuntimeError(
            f"{LABEL_MAP_PATH} not found. Run scripts/prepare_data.py first."
        )
    with open(LABEL_MAP_PATH, encoding="utf-8") as f:
        label_map = json.load(f)
    _idx_to_class = {v: k for k, v in label_map.items()}

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"{MODEL_PATH} not found. Train it first with "
            f"`python scripts/train.py --model {MODEL_NAME}` "
            f"(or set API_MODEL_NAME / API_MODEL_PATH to a checkpoint that exists)."
        )

    model, _, _ = build_model(MODEL_NAME, len(label_map))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    _model = model


def predict(image_bytes: bytes) -> dict:
    load_model()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    probabilidades = {_idx_to_class[i]: round(p.item(), 4) for i, p in enumerate(probs)}
    top_idx = int(probs.argmax())

    return {
        "clase_predicha": _idx_to_class[top_idx],
        "confianza": round(float(probs[top_idx]), 4),
        "probabilidades": probabilidades,
        "modelo": MODEL_NAME,
    }
