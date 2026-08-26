# Leaf Disease Classifier

Clasificador de enfermedades de vid con PyTorch y una API HTTP construida con FastAPI.

## Setup en una máquina nueva

Requiere Python 3.10+ instalado. Clonar el repo y correr el script de setup para el
sistema operativo correspondiente — crea `.venv` e instala `requirements.txt`:

**Linux / macOS**

```bash
git clone <url-del-repo>
cd leaf-disease-classifier
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
git clone <url-del-repo>
cd leaf-disease-classifier
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

Para tests de UI con Playwright (opcional, no hace falta para usar la app), agregar
`--dev` (`./setup.sh --dev`) o `-Dev` (`.\setup.ps1 -Dev`) — instala Playwright y
descarga un Chromium headless.

`10573036.zip` (dataset COCO original) y el `Grapes Disease Dataset/` externo no
están en el repo (ver `.gitignore`, son pesados); hay que copiarlos a la raíz del
proyecto antes de correr `scripts/prepare_data.py` / `scripts/prepare_diagnosis_data.py`.
Si ya tenés `data/processed` y un checkpoint en `models/`, podés saltar directo a
"Ejecutar la API" o a la sección de Streamlit.

## Ejecutar la API

El modelo predeterminado es el ResNet50 de diagnóstico. La aplicación carga
automáticamente el `label_map.json` ubicado junto a su checkpoint, para garantizar
que los índices de salida se interpreten con las mismas clases usadas al entrenar.

```bash
source .venv/bin/activate  # o .\.venv\Scripts\Activate.ps1 en Windows
uvicorn api.main:app --reload
```

La documentación interactiva queda disponible en <http://127.0.0.1:8000/docs>.
Para clasificar una imagen:

```bash
curl -X POST -F "file=@hoja.jpg;type=image/jpeg" http://127.0.0.1:8000/predict
```

Se puede servir el MobileNetV3 anterior sin cambiar el código. Como ese directorio
no incluye un mapa propio, se usa el mapa global legado de `models/label_map.json`:

```bash
API_MODEL_NAME=mobilenet_v3 uvicorn api.main:app
```

## Interfaz gráfica con Streamlit

La interfaz funciona directamente con el modelo local; no es necesario iniciar la API.

```bash
source .venv/bin/activate  # o .\.venv\Scripts\Activate.ps1 en Windows
streamlit run streamlit_app.py --server.headless true
```

(`--server.headless true` evita el prompt interactivo de "Welcome to Streamlit"
la primera vez que se corre en una máquina nueva.)

Abrí <http://localhost:8501>, seleccioná una imagen JPEG o PNG y la aplicación
mostrará la predicción, su confianza y un gráfico con las nueve probabilidades.

## Datos, entrenamiento y evaluación

El dataset original está separado por clase en `data/processed/{train,val,test}`.
Para volver a generarlo desde `data/raw`:

```bash
python scripts/prepare_data.py
```

Los siguientes comandos reproducen únicamente el experimento inicial y no deben
usarse para sobrescribir el ResNet50 de diagnóstico actual:

```bash
python scripts/train.py --model resnet50 --run-name resnet50_legacy --epochs 5 --finetune-epochs 5
python scripts/evaluate.py --model resnet50 --run-name resnet50_legacy
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

El ResNet50 predeterminado ya fue reentrenado con este dataset. Para reproducir el
entrenamiento y sobrescribir sus artefactos:

```bash
python scripts/train.py \
  --model resnet50 \
  --data-dir data/diagnosis_processed \
  --label-map models/diagnosis_label_map.json \
  --run-name resnet50 \
  --epochs 8 --finetune-epochs 15

python scripts/evaluate.py \
  --model resnet50 \
  --data-dir data/diagnosis_processed \
  --label-map models/diagnosis_label_map.json \
  --run-name resnet50
```

Para servir un checkpoint alternativo después del entrenamiento:

```bash
API_MODEL_PATH=models/resnet50/best_model.pt \
streamlit run streamlit_app.py
```

`API_LABEL_MAP_PATH` normalmente no hace falta: se resuelve desde el directorio del
checkpoint. Puede definirse explícitamente para modelos almacenados con otra estructura.

`api/inference.py` acepta además:

- `API_CONFIDENCE_THRESHOLD` (default `0.5`): por debajo de este valor la respuesta
  incluye `"es_incierto": true` en vez de reportarse como diagnóstico confiable.
- `API_TEMPERATURE` (default `1.0`): temperatura de softmax para calibrar las
  probabilidades (ajustar sobre el split de validación, p. ej. minimizando NLL o ECE,
  antes de fijarla en producción).

## Endpoints

- `GET /health`: estado del servicio.
- `POST /predict`: acepta archivos JPEG o PNG y devuelve clase, confianza,
  probabilidades para las nueve clases y modelo utilizado.
