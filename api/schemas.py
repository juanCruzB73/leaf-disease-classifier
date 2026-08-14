from pydantic import BaseModel


class PredictionResponse(BaseModel):
    clase_predicha: str
    confianza: float
    probabilidades: dict[str, float]
    modelo: str
