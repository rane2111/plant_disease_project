import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# -------------------- BACKGROUND IMAGE SETUP --------------------
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_image = get_base64_encoded_image(
    r"C:\Users\Raghunandan\OneDrive\Desktop\project\plant_disease_project\mango.jpg"
)
background_css = f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
    background-position: center;
    font-family: 'Segoe UI', sans-serif;
    color: white;
}}

h1, h2, h3, h4 {{
    color: #ffffff;
}}

.upload-box {{
    background-color: rgba(0, 0, 0, 0.6);
    border-radius: 10px;
    padding: 20px;
    animation: fadeIn 1.5s ease-in;
}}

.result-box, .remedy-box {{
    background-color: #fdbb2d;
    padding: 1.2rem;
    border: 2px solid #00cc66;
    border-left: 5px solid #00cc66;
    border-radius: 10px;
    margin: 1rem auto;
    font-size: 18px;
    width: fit-content;
    animation: bounceIn 1s;
    color: black;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to {{ opacity: 1; }}
}}

@keyframes bounceIn {{
    0% {{ transform: scale(0.8); opacity: 0; }}
    100% {{ transform: scale(1); opacity: 1; }}
}}
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)

# -------------------- MODEL + APP LOGIC --------------------
class_names = [
    "Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back",
    "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"
]

remedies = {
    "Bacterial Canker": {
        "treatment": [
            "Copper-based fungicides (e.g., Copper Oxychloride, Bordeaux Mixture).",
            "Streptomycin spray for bacterial suppression.",
            "Prune infected areas and disinfect tools."
        ],
        "best_time": "Apply fungicides in the early morning or late evening."
    },
    "Cutting Weevil": {
        "treatment": [
            "Use insecticides like Chlorpyrifos or Imidacloprid.",
            "Regular field monitoring to control spread.",
            "Introduce natural predators like parasitic wasps."
        ],
        "best_time": "Spray in the evening to prevent evaporation."
    },
    "Die Back": {
        "treatment": [
            "Apply fungicides such as Carbendazim or Copper Oxychloride.",
            "Remove and burn infected parts.",
            "Ensure proper drainage to prevent excess moisture."
        ],
        "best_time": "Apply treatment after pruning or in the early growth stage."
    },
    "Gall Midge": {
        "treatment": [
            "Use insecticides like Lambda-cyhalothrin or Spinosad.",
            "Remove infested plant parts and destroy them.",
            "Encourage natural predators like ladybugs."
        ],
        "best_time": "Apply insecticides in the late afternoon."
    },
    "Powdery Mildew": {
        "treatment": [
            "Use sulfur-based fungicides or neem oil.",
            "Improve air circulation by pruning dense growth.",
            "Avoid overhead watering."
        ],
        "best_time": "Apply fungicide in dry weather conditions, preferably in the morning."
    },
    "Sooty Mould": {
        "treatment": [
            "Control insect pests that produce honeydew (e.g., aphids, whiteflies).",
            "Wash leaves with mild soap water to remove mold.",
            "Use neem oil or insecticidal soaps."
        ],
        "best_time": "Apply treatment in the evening to avoid sunlight degradation."
    }
}

@st.cache_resource
def load_model():
    model_path = r"C:\Users\Raghunandan\OneDrive\Desktop\project\plant_disease_project\training\model.keras"
    return tf.keras.models.load_model(model_path)

model = load_model()

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, image

# -------------------- UI --------------------
st.title("🍃 Plant Disease Detection")
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>📤 Choose an image...</h4>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    image_array, image_display = preprocess_image(uploaded_file)

    st.image(image_display, caption="Uploaded Image", use_column_width=True)

    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100

    # Prediction box
    st.markdown(
        f"""
        <div class="result-box">
            <strong>✅ Prediction Result:</strong><br>
            <span style="font-size: 20px; font-weight: bold;">{predicted_class} ({confidence:.2f}%)</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if predicted_class == "Healthy":
        st.markdown(
            f"""
            <div class="remedy-box" style="background-color: #4caf50; text-align: center;">
                <h3>🌿 No Disease Detected</h3>
                <p>The plant is healthy and doesn't require treatment.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif predicted_class in remedies:
        remedy = remedies[predicted_class]
        with st.expander(f"📖 Remedy for {predicted_class}", expanded=True):
            st.markdown("<div class='remedy-box'>", unsafe_allow_html=True)

            st.markdown("<h4>💊 Treatment Steps:</h4>", unsafe_allow_html=True)
            for step in remedy["treatment"]:
                st.markdown(f"<li>{step}</li>", unsafe_allow_html=True)

            st.markdown(f"<h4>⏰ Best Time to Spray:</h4><p>{remedy['best_time']}</p>", unsafe_allow_html=True)

            st.markdown("<br><b style='color: red;'>⚠️ Use it under professional's advice.</b>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
