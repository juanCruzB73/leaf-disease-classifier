# Leaf Disease Classifier

Clasificador de enfermedades de vid con PyTorch y una API HTTP construida con FastAPI.

## Ejecutar la API

Las dependencias están instaladas en `.venv` y el modelo predeterminado es ResNet50.

```bash
source .venv/bin/activate
uvicorn api.main:app --reload
```

La documentación interactiva queda disponible en <http://127.0.0.1:8000/docs>.
Para clasificar una imagen:

```bash
curl -X POST -F "file=@hoja.jpg;type=image/jpeg" http://127.0.0.1:8000/predict
```

Se puede servir MobileNetV3 sin cambiar el código:

```bash
API_MODEL_NAME=mobilenet_v3 uvicorn api.main:app
```

## Interfaz gráfica con Streamlit

La interfaz funciona directamente con el modelo local; no es necesario iniciar la API.

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Abrí <http://localhost:8501>, seleccioná una imagen JPEG o PNG y la aplicación
mostrará la predicción, su confianza y un gráfico con las nueve probabilidades.

## Datos, entrenamiento y evaluación

Los datos procesados están separados por clase en `data/processed/{train,val,test}`.
Para volver a generarlos desde `data/raw`:

```bash
python scripts/prepare_data.py
```

Para entrenar y evaluar una arquitectura:

```bash
python scripts/train.py --model resnet50 --epochs 5 --finetune-epochs 5
python scripts/evaluate.py --model resnet50
```

El mejor checkpoint se guarda en `models/<modelo>/best_model.pt` y las métricas de
prueba en `models/<modelo>/test_metrics.json`.

### Dataset de diagnóstico corregido

El flujo corregido elimina `vines_leaf` y `vines_grape` (describen el órgano, no
su estado sanitario), incorpora las hojas sanas y la mancha bacteriana del dataset
adicional, y elimina duplicados antes de dividir:

```bash
python scripts/prepare_diagnosis_data.py
```

Entrenamiento y evaluación independientes del modelo anterior:

```bash
python scripts/train.py \
  --model resnet50 \
  --data-dir data/diagnosis_processed \
  --label-map models/diagnosis_label_map.json \
  --run-name diagnosis_resnet50 \
  --epochs 8 --finetune-epochs 15

python scripts/evaluate.py \
  --model resnet50 \
  --data-dir data/diagnosis_processed \
  --label-map models/diagnosis_label_map.json \
  --run-name diagnosis_resnet50
```

Para servir el nuevo modelo después del entrenamiento:

```bash
API_MODEL_PATH=models/diagnosis_resnet50/best_model.pt \
API_LABEL_MAP_PATH=models/diagnosis_resnet50/label_map.json \
streamlit run streamlit_app.py
```

## Endpoints

- `GET /health`: estado del servicio.
- `POST /predict`: acepta archivos JPEG o PNG y devuelve clase, confianza,
  probabilidades para las nueve clases y modelo utilizado.
