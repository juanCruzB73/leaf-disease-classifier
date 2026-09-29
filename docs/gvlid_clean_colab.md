# Experimento limpio en Google Colab

1. Abrir `notebooks/gvlid_clean_colab.ipynb` en Colab mediante **Archivo → Subir notebook**.
2. Activar GPU en **Entorno de ejecución → Cambiar tipo de entorno**.
3. Subir `colab_exports/gvlid_clean_colab.zip` a Google Drive, en Mi unidad.
4. Ejecutar las celdas en orden. Ajustar `ZIP_PATH` si se subió a otra carpeta.
5. Usar una carpeta `RUN` nueva: no se sobrescriben entrenamientos ni evaluaciones.

No se entrenó localmente. El paquete contiene únicamente los datasets limpios,
los scripts y auditorías necesarios. No contiene GVLiD original, el híbrido anterior
ni checkpoints antiguos. Tampoco vuelve a realizar la limpieza.

## Configuración y resultados

Se reutilizan `scripts/train.py`, `scripts/dataset.py` y `scripts/evaluate.py`.
ResNet50 ImageNet1K V2; cuatro clases; entrada 224; transforms originales;
Adam; batch 32; seed 42; 8 épocas congeladas a 0.001 + 15 completas a 0.0001;
cross entropy ponderada por frecuencia inversa. El comportamiento original de
BatchNorm durante warmup también se conserva. La selección usa Macro F1 de
validación controlada exclusivamente. No se usa natural validation ni test.

Híbrido: 4.684 train = 3.250 controladas + 1.434 naturales; validación controlada
406; test controlado 406; test natural 307. Las 307 naturales de validación quedan
reservadas. `vl_black_rot` aporta solamente 14 imágenes al test natural.

Antes de cada entrenamiento se verifican archivos actuales contra el manifiesto,
SHA-256, pHash, clases, cantidades y grupos confirmados entre particiones. Las rutas
absolutas de la auditoría original se traducen de forma explícita al directorio Colab.
Las cantidades y hashes de la referencia controlada se preservan; no se regeneran splits.

Resultados persistentes en `RUN` (por defecto Mi unidad/resultados_comparacion_clean):

- `experiment_manifest.json`: configuración, fecha, fuentes, hashes, GPU, selección y checkpoint.
- `models/hybrid/{best_model.pt,last_model.pt,train_history.json,label_map.json}`.
- `hybrid_training.log` y, si se reentrena baseline, `baseline_manifest.json`,
  `baseline_training.log` y `models/baseline/`.
- `evaluation/metricas_{hibrido_controlado,hibrido_natural,baseline_natural}.csv`.
- `evaluation/comparacion_global_natural.csv` y `comparacion_por_clase_natural.csv`.
- Matrices de confusión CSV/PNG absolutas y normalizadas, por conjunto/modelo.
- JSON de métricas y predicciones individuales, aciertos/errores, logs y manifiesto de evaluación.

## Baseline original

Existe `models/resnet50/best_model.pt`, cuatro clases e historial de 8+15 épocas.
Falta un manifiesto que vincule los hashes de ese checkpoint con los splits y la
configuración original completa; tampoco está disponible `data/controlled_processed`.
No se usa ese checkpoint como baseline verificado ni se reutilizan métricas antiguas.

Por defecto el notebook prepara una copia de las imágenes controladas del dataset
limpio y reentrena un baseline comparable. Se verifica byte a byte esa copia antes
de entrenar. Ambos modelos se evalúan sobre exactamente las mismas 307 imágenes.
Si `RETRAIN_BASELINE=False`, se exportan resultados del híbrido y una nota de baseline
pendiente; no se fabrican métricas o tablas de comparación.

## Comandos equivalentes (desde la raíz del paquete)

```bash
python scripts/run_gvlid_clean_experiment.py preflight
python scripts/run_gvlid_clean_experiment.py train --run-dir /ruta/resultados_comparacion_clean
python scripts/run_gvlid_clean_experiment.py prepare-baseline
python scripts/run_gvlid_clean_experiment.py train --baseline --run-dir /ruta/resultados_comparacion_clean
python scripts/run_gvlid_clean_experiment.py evaluate --run-dir /ruta/resultados_comparacion_clean
```

Los comandos exactos de `train.py` se guardan en cada manifiesto. `train.py` no se
modificó. `evaluate.py` incorpora `--output` para separar resultados y conservar
predicciones/aciertos/errores sin alterar la inferencia.

Para crear otro paquete local sin sobrescribir el actual:

```bash
python scripts/package_gvlid_clean_colab.py --output colab_exports/gvlid_clean_colab_v2.zip
```

Colab necesita conexión para instalar dependencias y descargar los pesos oficiales.
Mantiene su versión compatible de PyTorch/torchvision; el manifiesto registra el
entorno efectivo. La seed fija no garantiza identidad bit a bit entre GPUs/versiones.
No hay reanudación exacta del optimizador en el pipeline anterior: una ejecución
interrumpida debe repetirse completa en otra carpeta, sin evaluar un checkpoint
parcial. No modificar la configuración después de consultar el test.
