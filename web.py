import tempfile

import cv2
import easyocr
import pandas as pd
import streamlit as st

from ultralytics import YOLO

from py_files.detection import extract_plate
from py_files.preprocessing import preprocess_plate
from py_files.postprocessing import best_prediction

st.set_page_config(
    page_title="License Plate Recognition",
    layout="wide"
)

st.title("License Plate Recognition System")

st.write(
    """
    Deep Learning pipeline:

    YOLO11 → Plate Extraction → Image Preprocessing →
    EasyOCR → Post-processing
    """
)

@st.cache_resource
def load_models():

    model = YOLO("models/best.pt")

    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    return model, reader


model, reader = load_models()

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as tmp:

        tmp.write(uploaded.read())

        image_path = tmp.name

    plates = extract_plate(
        image_path,
        model
    )

    if len(plates) == 0:

        st.error("No license plate detected.")

        st.stop()

    plate = plates[0]

    result = best_prediction(
        plate,
        reader
    )

    processed = preprocess_plate(
        plate,
        method=result["best_method"],
        use_super_resolution=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("Original")

        original = cv2.imread(image_path)

        original = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            original,
            use_container_width=True
        )

    with col2:

        st.subheader("Detected Plate")

        st.image(
            plate,
            use_container_width=True
        )

    with col3:

        st.subheader("Preprocessed")

        st.image(
            processed,
            clamp=True,
            use_container_width=True
        )

    st.divider()

    st.subheader("Prediction")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Final prediction",
        result["prediction"]
    )

    c2.metric(
        "Raw OCR",
        result["raw_prediction"]
    )

    c3.metric(
        "Confidence",
        f"{result['confidence']:.3f}"
    )

    st.divider()

    st.subheader("OCR comparison")

    df = pd.DataFrame(
        result["predictions"]
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )