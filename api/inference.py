"""Loads the winning trained model once and exposes a predict() function.

Which model/checkpoint to serve is controlled by two env vars so the API doesn't
need code changes once train.py + evaluate.py pick a winner:

    API_MODEL_NAME  - "resnet50" or "mobilenet_v3" (default: resnet50)
    API_MODEL_PATH  - path to the .pt checkpoint (default: models/<API_MODEL_NAME>/best_model.pt)
    API_LABEL_MAP_PATH - matching class map (default: label_map.json next to the checkpoint,
                         with models/label_map.json as a legacy fallback)
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

MODEL_NAME = os.environ.get("API_MODEL_NAME", "resnet50")
MODEL_PATH = Path(os.environ.get("API_MODEL_PATH", ROOT / "models" / MODEL_NAME / "best_model.pt"))
_checkpoint_label_map = MODEL_PATH.parent / "label_map.json"
_default_label_map = (
    _checkpoint_label_map
    if _checkpoint_label_map.exists()
    else ROOT / "models" / "label_map.json"
)
LABEL_MAP_PATH = Path(os.environ.get("API_LABEL_MAP_PATH", _default_label_map))

# Softmax temperature for probability calibration (>1 softens overconfident logits;
# fit on a held-out split, e.g. by minimizing NLL/ECE, then set via env var).
TEMPERATURE = float(os.environ.get("API_TEMPERATURE", "1.0"))
# Below this top-class probability the prediction is flagged as unreliable instead
# of being reported as a confident diagnosis.
CONFIDENCE_THRESHOLD = float(os.environ.get("API_CONFIDENCE_THRESHOLD", "0.5"))

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
    expected_indices = set(range(len(label_map)))
    if set(label_map.values()) != expected_indices:
        raise RuntimeError(
            f"{LABEL_MAP_PATH} must map classes to every index from 0 to "
            f"{len(label_map) - 1}."
        )
    _idx_to_class = {v: k for k, v in label_map.items()}

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"{MODEL_PATH} not found. Train it first with "
            f"`python scripts/train.py --model {MODEL_NAME}` "
            f"(or set API_MODEL_NAME / API_MODEL_PATH to a checkpoint that exists)."
        )

    # The checkpoint already contains every weight; avoid downloading ImageNet
    # weights when the API starts in an offline/production environment.
    model, _, _ = build_model(MODEL_NAME, len(label_map), pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    _model = model


def predict(image_bytes: bytes) -> dict:
    load_model()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits / TEMPERATURE, dim=1)[0]

    probabilidades = {_idx_to_class[i]: round(p.item(), 4) for i, p in enumerate(probs)}
    top_idx = int(probs.argmax())
    confianza = float(probs[top_idx])

    return {
        "clase_predicha": _idx_to_class[top_idx],
        "confianza": round(confianza, 4),
        "es_incierto": confianza < CONFIDENCE_THRESHOLD,
        "probabilidades": probabilidades,
        "modelo": MODEL_NAME,
    }
