# Limpieza auditable de GVLiD

Ejecutar desde la raíz con Pillow e ImageHash instalados:

```bash
.venv/bin/python scripts/prepare_gvlid_clean.py
```

El original se abre únicamente para lectura. No se usan enlaces duros. El script
anterior, entrenamiento y evaluación no cambian. Las salidas deben ser nuevas;
para repetir la auditoría usar `--report-dir reports/gvlid_clean_run2`.

Se agrupan globalmente SHA-256 iguales y pHash de 64 bits iguales usando Union-Find.
pHash=0 es la regla operativa de duplicación visual de este procedimiento, no una
prueba matemática de identidad: las colisiones perceptuales son posibles.
El CSV por archivo permite auditar todas las exclusiones. Los grupos con etiquetas
diferentes se excluyen completos. El representante es el primer archivo en orden
estable. Los contadores SHA y pHash describen etapas previas a los conflictos y
no deben sumarse a las exclusiones por conflicto como categorías independientes.

Todas las parejas originales con distancia 1–5 se guardan en el CSV de candidatos
y tienen una comparación en `gvlid_near_duplicates_review`. No se descartan
silenciosamente candidatos asociados a grupos que ya fueron excluidos.
El código de salida **2** significa auditoría terminada con revisión pendiente;
no se crea ningún split. Las cantidades retenidas son provisionales hasta revisar.

Copiar `gvlid_near_duplicate_candidates.csv` a un archivo de decisiones y completar
cada celda `decision` con `duplicate` o `different`. No cambiar `pair_id`:
identifica los nombres y SHA de ambos originales. Se rechazan IDs desconocidos.
Los duplicados confirmados se unen transitivamente y se recalculan los conflictos.
Los pares declarados diferentes se pueden repartir; ya no se consideran relacionados
por duplicación. No existe opción para ignorar la revisión pendiente.

```bash
.venv/bin/python scripts/prepare_gvlid_clean.py \
  --report-dir reports/gvlid_clean_reviewed \
  --decisions reports/gvlid_decisions.csv \
  --controlled-data data/controlled_processed
```

Si la referencia baseline no está disponible, recuperarla sin regenerar sus splits.
Alternativamente, se permite reutilizar explícitamente la parte `controlled_*`
del híbrido anterior:

```bash
.venv/bin/python scripts/prepare_gvlid_clean.py \
  --report-dir reports/gvlid_clean_reviewed \
  --decisions reports/gvlid_decisions.csv \
  --controlled-data data/gvlid_hybrid_processed --controlled-from-hybrid
```

Esta alternativa valida etiquetas y cantidades contra el manifiesto anterior y
preserva exactamente el multiconjunto de bytes por clase y split. No demuestra
independientemente igualdad con un baseline ausente; el manifiesto identifica la
referencia usada. El archivo de etiquetas por defecto es
`models/resnet50/label_map.json` (configurable con `--source-label-map`).

Los splits naturales usan 70/15/15, redondeo como el script anterior y semilla 42
más el índice de clase ordenada. Solo train natural se agrega al híbrido;
val/test naturales permanecen en `data/gvlid_clean_natural_eval`. La validación del
híbrido contiene exclusivamente la referencia controlada.

La validación final recalcula SHA y pHash de las copias y comprueba todas las
intersecciones entre splits, incluyendo PlantVillage, así como grupos confirmados,
conflictos y conservación de todas las imágenes controladas. Si el baseline tiene
leakage, falla sin modificarlo. Un fallo puede dejar salidas parciales para
inspección: solo un `manifest.json` con `split_status: validated` indica éxito.
El manifiesto CSV permite rastrear cada destino. No se entrena ningún modelo.

Pruebas: `.venv/bin/python -m unittest discover -s tests -v`.
