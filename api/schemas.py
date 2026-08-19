from pydantic import BaseModel


class PredictionResponse(BaseModel):
    clase_predicha: str
    confianza: float
    es_incierto: bool
    probabilidades: dict[str, float]
    modelo: str
