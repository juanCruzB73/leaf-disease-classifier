from fastapi import FastAPI, File, HTTPException, UploadFile

from .inference import load_model, predict
from .schemas import PredictionResponse

app = FastAPI(title="Leaf Disease Classifier API")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


@app.on_event("startup")
def _startup():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"unsupported content type: {file.content_type}")

    image_bytes = await file.read()
    try:
        result = predict(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="could not read image")

    return result
