import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="CropCare AI",
    page_icon="🌱",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5fff7;
}

.title {
    font-size: 45px;
    font-weight: 700;
    color: #16803c;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #eefaf0;
    border: 1px solid #b7dfbd;
    margin-top: 20px;
}

.disease {
    font-size: 28px;
    font-weight: bold;
    color: #16803c;
}

.info-title {
    font-size: 20px;
    font-weight: bold;
    color: #16803c;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = os.path.join(
    "model",
    "multi_crop_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    "model",
    "class_names.json"
)

IMAGE_SIZE = (128, 128)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found: "
            + MODEL_PATH
        )
        st.stop()

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# --------------------------------------------------
# LOAD CLASS NAMES
# --------------------------------------------------

@st.cache_data
def load_class_names():

    if not os.path.exists(CLASS_NAMES_PATH):
        st.error(
            "class_names.json not found: "
            + CLASS_NAMES_PATH
        )
        st.stop()

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    # Handle dictionary format
    if isinstance(data, dict):

        # Example:
        # {"0": "Apple___healthy", ...}

        try:
            data = [
                data[str(i)]
                for i in range(len(data))
            ]

        except Exception:
            data = list(data.values())

    return data


# --------------------------------------------------
# DISEASE INFORMATION
# --------------------------------------------------

def get_disease_info(disease):

    # Default information
    info = {
        "description":
            "The uploaded leaf appears to match this disease class.",

        "precautions":
            "Remove badly affected leaves and maintain good field hygiene.",

        "treatment":
            "Use a treatment registered for this crop and disease and follow the product label.",

        "natural_care":
            "Keep the crop area clean, avoid excessive moisture on leaves, and maintain proper spacing.",

        "medicine":
            "Consult a local agriculture expert for a crop-specific registered product."
    }

    # Try to use the existing disease_info.py
    try:

        from disease_info import DISEASE_INFO

        if disease in DISEASE_INFO:

            existing = DISEASE_INFO[disease]

            if isinstance(existing, dict):

                info.update(existing)

            elif isinstance(existing, str):

                info["description"] = existing

    except Exception:
        pass

    return info


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_disease(image, model, class_names):

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image)

    image_array = image_array.astype("float32") / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = np.argmax(
        probabilities
    )

    confidence = (
        probabilities[predicted_index] * 100
    )

    disease = class_names[predicted_index]

    return disease, confidence


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">🌱 CropCare AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Crop Disease Detection System'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# INFORMATION
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.success("🌿 AI Disease Detection")

with col2:
    st.info("📷 Upload Leaf Image")

with col3:
    st.warning("💊 Treatment Information")


st.divider()


# --------------------------------------------------
# UPLOAD IMAGE
# --------------------------------------------------

st.subheader("🔍 Disease Detector")

uploaded_file = st.file_uploader(
    "Upload a crop leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = load_model()

class_names = load_class_names()


# --------------------------------------------------
# IMAGE + PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📷 Uploaded Image")

        st.image(
            image,
            caption="Selected Leaf",
            use_container_width=True
        )

    with col2:

        st.subheader("🤖 AI Prediction")

        if st.button(
            "🌱 Detect Disease",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "AI is analyzing the leaf..."
            ):

                try:

                    disease, confidence = predict_disease(
                        image,
                        model,
                        class_names
                    )

                    info = get_disease_info(
                        disease
                    )

                    st.markdown(
                        '<div class="result-box">',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'<div class="disease">'
                        f'{disease}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                    st.markdown("### 📋 Disease Information")

                    st.markdown(
                        '<div class="info-title">'
                        'Description'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.write(
                        info.get(
                            "description",
                            "Information not available."
                        )
                    )

                    st.markdown(
                        '<div class="info-title">'
                        '⚠️ Precautions'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.write(
                        info.get(
                            "precautions",
                            "Information not available."
                        )
                    )

                    st.markdown(
                        '<div class="info-title">'
                        '💊 Recommended Treatment'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.write(
                        info.get(
                            "treatment",
                            "Information not available."
                        )
                    )

                    st.markdown(
                        '<div class="info-title">'
                        '🌿 Natural Care'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.write(
                        info.get(
                            "natural_care",
                            "Information not available."
                        )
                    )

                    st.markdown(
                        '<div class="info-title">'
                        '🧪 Recommended Medicine'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.write(
                        info.get(
                            "medicine",
                            "Information not available."
                        )
                    )

                    st.caption(
                        "⚠️ Always follow the product label "
                        "and local agricultural guidance."
                    )

                except Exception as e:

                    st.error(
                        "Prediction failed."
                    )

                    st.exception(e)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center;">
        <b>🌱 CropCare AI</b><br>
        AI-powered crop disease detection
    </div>
    """,
    unsafe_allow_html=True
)