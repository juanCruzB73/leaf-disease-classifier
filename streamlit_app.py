"""Streamlit interface for the local leaf-disease classifier."""

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from api.inference import load_model, predict


CLASS_NAMES = {
    "healthy_leaf": "Hoja sana",
    "vl_bacterial_spot": "Mancha bacteriana en hoja",
    "vg_black_rot": "Podredumbre negra en racimo",
    "vg_downy_mildew": "Mildiu en racimo",
    "vg_grey_mould": "Moho gris en racimo",
    "vg_powdery_mildew": "Oídio en racimo",
    "vines_grape": "Racimo de vid",
    "vines_leaf": "Hoja de vid",
    "vl_black_rot": "Podredumbre negra en hoja",
    "vl_downy_mildew": "Mildiu en hoja",
    "vl_powdery_mildew": "Oídio en hoja",
}


@st.cache_resource(show_spinner="Cargando el modelo…")
def initialize_model():
    load_model()


st.set_page_config(page_title="Clasificador de enfermedades de vid", page_icon="🍇")
st.title("Clasificador de enfermedades de vid")
st.write(
    "Subí una fotografía de una hoja o un racimo para estimar la clase y consultar "
    "las probabilidades del modelo."
)

try:
    initialize_model()
except Exception as exc:
    st.error(f"No se pudo cargar el modelo: {exc}")
    st.stop()

uploaded_file = st.file_uploader("Elegí una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("La predicción aparecerá aquí después de seleccionar una imagen.")
else:
    image_bytes = uploaded_file.getvalue()
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("El archivo seleccionado no es una imagen válida.")
        st.stop()

    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption=uploaded_file.name, width="stretch")

    try:
        result = predict(image_bytes)
    except Exception as exc:
        st.error(f"No se pudo analizar la imagen: {exc}")
        st.stop()

    predicted_class = result["clase_predicha"]
    confidence = result["confianza"]
    with right:
        st.subheader("Resultado")
        if result.get("es_incierto"):
            st.warning(
                f"Resultado incierto (mejor candidata: {CLASS_NAMES.get(predicted_class, predicted_class)})."
                " La confianza es demasiado baja para reportarlo como diagnóstico."
            )
        else:
            st.success(CLASS_NAMES.get(predicted_class, predicted_class))
        st.metric("Confianza", f"{confidence:.1%}")
        st.caption(f"Modelo: {result['modelo']}")
        if "healthy_leaf" in result["probabilidades"] and "healthy_grape" not in result["probabilidades"]:
            st.caption(
                "El modelo reconoce hojas sanas, pero todavía no dispone de una clase "
                "de racimo sano."
            )

    probabilities = pd.DataFrame(
        {
            "Clase": [CLASS_NAMES.get(name, name) for name in result["probabilidades"]],
            "Probabilidad": list(result["probabilidades"].values()),
        }
    ).sort_values("Probabilidad", ascending=False)

    st.subheader("Probabilidades por clase")
    st.bar_chart(probabilities, x="Clase", y="Probabilidad", horizontal=True)
    with st.expander("Ver valores exactos"):
        st.dataframe(
            probabilities.style.format({"Probabilidad": "{:.2%}"}),
            hide_index=True,
            width="stretch",
        )

st.divider()
st.caption(
    "Herramienta experimental: el resultado no sustituye la evaluación de un especialista."
)
