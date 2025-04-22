import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import io

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# -------------------- BACKGROUND IMAGE SETUP --------------------
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_image = get_base64_encoded_image(
    r"C:\Users\Raghunandan\OneDrive\Desktop\project\plant_disease_project\mango.jpg"
)

remedy_bg = get_base64_encoded_image(
    r"C:\Users\Raghunandan\OneDrive\Desktop\project\plant_disease_project\image.png"
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

h1 {{
    background-color: rgba(0, 0, 0, 0.5);
    padding: 1rem;
    border-radius: 15px;
    text-align: center;
}}

.upload-box {{
    background-color: rgba(0, 0, 0, 0.1);
    border-radius: 12px;
    padding: 25px;
    width: 100%;
    max-width: 500px;
    margin: 0 auto;
    animation: fadeIn 1.5s ease-in;
}}

.result-box {{
    background-color: rgba(253, 187, 45, 0.95);
    padding: 1.2rem;
    border: 2px solid #00cc66;
    border-left: 5px solid #00cc66;
    border-radius: 10px;
    margin: 1rem auto;
    font-size: 25px;
    font-weight: bold;
    width: fit-content;
    animation: bounceIn 1s;
    color: black;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}}

.remedy-box {{
    background-image: url("data:image/jpg;base64,{remedy_bg}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    padding: 2rem;
    border: 2px solid #00cc66;
    border-left: 6px solid #00cc66;
    border-radius: 12px;
    margin: 2rem auto;
    font-size: 26px;
    font-weight: bold;
    width: 85%;
    max-width: 750px;
    animation: bounceIn 1s;
    color: RED;
    text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.9);
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
}}

.download-container {{
    display: flex;
    justify-content: center;
    margin-top: -15px;
    margin-bottom: 20px;
}}

.download-btn {{
    background-color: #000000;  /* Changed to black */
    color: white;
    padding: 12px 24px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 8px;
    border: none;
    transition: all 0.3s;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}}

.download-btn:hover {{
    background-color: #333333;  /* Dark gray on hover */
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.3);
}}



.download-btn:active {{
    transform: translateY(0);
}}

a {{
    text-decoration: none !important;
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

# -------------------- MODEL + LOGIC --------------------
class_names = [
    "Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back",
    "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"
]

remedies = {
    "Anthracnose": {
        "en": [
            "1. 🌿 Spray fungicides like Mancozeb or Chlorothalonil.",
            "2. 🚯 Remove and destroy infected plant parts.",
            "3. 💧 Avoid overhead irrigation to reduce humidity.",
            "4. 🌬️ Ensure proper spacing between plants for airflow."
        ],
        "hi": [
            "1. 🌿 मेंकोजेब या क्लोरोथालोनिल जैसे कवकनाशकों का छिड़काव करें।",
            "2. 🚯 संक्रमित पौधों के हिस्सों को हटाकर नष्ट कर दें।",
            "3. 💧 आर्द्रता को कम करने के लिए ऊपरी सिंचाई से बचें।",
            "4. 🌬️ हवा के प्रवाह के लिए पौधों के बीच उचित दूरी बनाए रखें।"
        ],
        "mr": [
            "1. 🌿 मॅन्कोझेब किंवा क्लोरोथालोनिलसारख्या बुरशीनाशकांचा फवारणी करा.",
            "2. 🚯 संक्रमित भाग काढून टाका आणि जाळून टाका.",
            "3. 💧 हवेमधील आर्द्रता टाळण्यासाठी वरून पाणी टाकू नका.",
            "4. 🌬️ हवेचा योग्य प्रवाह राखण्यासाठी अंतर ठेवा."
        ],
        "time": "🕰️ Early morning or late evening during dry weather."
    },
    "Bacterial Canker": {
        "en": [
            "1. 🧴 Copper-based fungicides (e.g., Copper Oxychloride, Bordeaux Mixture).",
            "2. 💉 Streptomycin spray for bacterial suppression.",
            "3. ✂️ Prune infected areas and disinfect tools."
        ],
        "hi": [
            "1. 🧴 कॉपर आधारित कवकनाशक (जैसे, कॉपर ऑक्सीक्लोराइड, बोर्डो मिश्रण)।",
            "2. 💉 जीवाणुओं को दबाने के लिए स्ट्रेप्टोमाइसिन का छिड़काव करें।",
            "3. ✂️ संक्रमित हिस्सों की छंटाई करें और उपकरणों को कीटाणुरहित करें।"
        ],
        "mr": [
            "1. 🧴 कॉपर ऑक्सीक्लोराईड किंवा बोर्डो मिश्रणासारखे कॉपर-आधारित फंगिसाइड वापरा.",
            "2. 💉 बॅक्टेरियासाठी स्ट्रेप्टोमायसिन फवारणी करा.",
            "3. ✂️ संक्रमित भाग छाटून साधने निर्जंतुक करा."
        ],
        "time": "🕰️ Apply fungicides in the early morning or late evening."
    },
    "Cutting Weevil": {
        "en": [
            "1. 🐜 Use insecticides like Chlorpyrifos or Imidacloprid.",
            "2. 👀 Regular field monitoring to control spread.",
            "3. 🦗 Introduce natural predators like parasitic wasps."
        ],
        "hi": [
            "1. 🐜 क्लोरपायरीफोस या इमिडाक्लोप्रिड सारख्या कीटनाशकांचा वापर करें।",
            "2. 👀 फैलाव को नियंत्रित करने के लिए नियमित क्षेत्र निगरानी करें।",
            "3. 🦗 परजीवी ततैया जैसे प्राकृतिक शिकारी को पेश करें।"
        ],
        "mr": [
            "1. 🐜 क्लोरपायरीफोस किंवा इमिडाक्लोप्रिड सारख्या कीटकनाशकांचा वापर करा.",
            "2. 👀 प्रगतीवर नियंत्रण ठेवण्यासाठी नियमितपणे क्षेत्र निरीक्षण करा.",
            "3. 🦗 परजीवी ततैया यांसारखे नैतिक शिकार आणा."
        ],
        "time": "🕰️ Spray in the evening to prevent evaporation."
    },
    "Die Back": {
        "en": [
            "1. 💊 Apply fungicides such as Carbendazim or Copper Oxychloride.",
            "2. 🔥 Remove and burn infected parts.",
            "3. 🌧️ Ensure proper drainage to prevent excess moisture."
        ],
        "hi": [
            "1. 💊 कार्बेंडाझिम या कॉपर ऑक्सीक्लोराइड जैसे कवकनाशकों का छिड़काव करें।",
            "2. 🔥 संक्रमित हिस्सों को हटा कर जला दें।",
            "3. 🌧️ अत्यधिक नमी को रोकने के लिए उचित जल निकासी सुनिश्चित करें।"
        ],
        "mr": [
            "1. 💊 कार्बेंडाझिम किंवा कॉपर ऑक्सीक्लोराइडसारख्या बुरशीनाशकांचा वापर करा.",
            "2. 🔥 संक्रमित भाग काढून जाळून टाका.",
            "3. 🌧️ अतिरिक्त ओलावा टाळण्यासाठी योग्य निचऱ्याची खात्री करा."
        ],
        "time": "🕰️ Apply treatment after pruning or in the early growth stage."
    },
    "Gall Midge": {
        "en": [
            "1. 🦗 Use insecticides like Lambda-cyhalothrin or Spinosad.",
            "2. 🗑️ Remove infested plant parts and destroy them.",
            "3. 🐞 Encourage natural predators like ladybugs."
        ],
        "hi": [
            "1. 🦗 लैम्ब्डा-सायहॅलोथ्रिन या स्पिनोसाड जैसे कीटनाशकों का उपयोग करें।",
            "2. 🗑️ संक्रमित पौधों के हिस्सों को हटा कर नष्ट करें।",
            "3. 🐞 प्राकृतिक शिकारी जैसे लेडीबग्स को प्रोत्साहित करें।"
        ],
        "mr": [
            "1. 🦗 लैम्ब्डा-सायहॅलोथ्रिन किंवा स्पिनोसाडसारख्या कीटकनाशकांचा वापर करा.",
            "2. 🗑️ संक्रमित पौधांच्या भागांना काढून टाका आणि नष्ट करा.",
            "3. 🐞 नैतिक शिकार जसे की लेडीबग्सला प्रोत्साहन द्या."
        ],
        "time": "🕰️ Apply insecticides in the late afternoon."
    },
    "Powdery Mildew": {
        "en": [
            "1. 💧 Use sulfur-based fungicides or neem oil.",
            "2. ✂️ Improve air circulation by pruning dense growth.",
            "3. 🚫 Avoid overhead watering."
        ],
        "hi": [
            "1. 💧 सल्फर-आधारित कवकनाशकों या नीम तेल का उपयोग करें।",
            "2. ✂️ घने वृद्धि को छाँटकर हवा के संचार को सुधारें।",
            "3. 🚫 ऊपरी सिंचाई से बचें।"
        ],
        "mr": [
            "1. 💧 गंधक आधारित फंगिसाइड्स किंवा नीम तेल वापरा.",
            "2. ✂️ दाट वाढ छाटून हवेचा प्रवाह सुधारावा.",
            "3. 🚫 वरून पाणी देण्यापासून टाळा."
        ],
        "time": "🕰️ Apply fungicide in dry weather conditions, preferably in the morning."
    },
    "Sooty Mould": {
        "en": [
            "1. 🐜 Control insect pests that produce honeydew (e.g., aphids, whiteflies).",
            "2. 🚿 Wash leaves with mild soap water to remove mold.",
            "3. 🌿 Use neem oil or insecticidal soaps."
        ],
        "hi": [
            "1. 🐜 उनखंड (जैसे, एफिड्स, सफेद मक्खियाँ) उत्पन्न करने वाले कीटों को नियंत्रित करें।",
            "2. 🚿 पत्तियों को हल्के साबुन पानी से धोकर फफूंदी हटा दें।",
            "3. 🌿 नीम तेल या कीटनाशक साबुन का उपयोग करें।"
        ],
        "mr": [
            "1. 🐜 मधाचे हनीड्यू उत्पादन करणारे कीटक नियंत्रित करा (उदा., एफिड्स, पांढरी माशी).",
            "2. 🚿 फुलांसाठी हलक्या साबणाच्या पाण्याने पाणी घाला आणि फफुंदी काढा.",
            "3. 🌿 नीम तेल किंवा कीटकनाशक साबण वापरा."
        ],
        "time": "🕰️ Apply treatment in the evening to avoid sunlight degradation."
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

def generate_remedy_text(remedy_steps, best_time):
    remedy_text = "\n".join(remedy_steps)
    remedy_text += f"\n\nBest Time to Spray: {best_time}"
    return remedy_text

def create_download_link(text, filename):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a class="download-btn" href="data:file/txt;base64,{b64}" download="{filename}">Download Remedy</a>'

# -------------------- UI --------------------
st.title("🍃 Plant Disease Detection")

with st.container():
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>📤 Choose a plant leaf image:</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

lang = st.radio("🌐 Choose Language for Remedy:", ("English", "Hindi", "Marathi"), horizontal=True)
lang_map = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
selected_lang = lang_map[lang]

if uploaded_file is not None:
    image_array, image_display = preprocess_image(uploaded_file)
    st.image(image_display, caption="Uploaded Image", use_column_width=True)

    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100

    st.markdown(
        f"""
        <div class="result-box">
            <strong>✅ Prediction Result:</strong><br>
            <span style="font-size: 20px; font-weight: bold;">{predicted_class} ({confidence:.2f}%)</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    result_text = f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%\n"

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
        result_text += "\nNo remedy required. The plant is healthy."
    else:
        remedy = remedies.get(predicted_class, {})
        if remedy:
            steps = remedy[selected_lang]
            remedy_text = generate_remedy_text(steps, remedy["time"])
            st.markdown(
                f"""
                <div class="remedy-box">
                    {"<br>".join(steps)}
                    <br><br>
                    🕰️ <strong>Best Time to Spray:</strong> {remedy['time']}
                    <br><br>
                    <span style='font-size: 20px; font-weight: bold; color: yellow;'>⚠️ Use it under professional's advice.</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Optional download feature (enabled here, but you can comment it if not needed)
            with st.container():
                st.markdown("<div class='download-container'>", unsafe_allow_html=True)
                download_link = create_download_link(remedy_text, f"{predicted_class}_remedy.txt")
                st.markdown(download_link, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            result_text += f"\n\nRemedy Steps:\n{remedy_text}"
