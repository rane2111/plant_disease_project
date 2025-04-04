from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import io
import sys

app = FastAPI()

# Enable CORS for frontend integration
origins = ["http://localhost", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained deep learning model
MODEL_PATH = r"C:\Users\Raghunandan\OneDrive\Desktop\project\plant_disease_project\training\model.keras"
try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    MODEL = None  # Prevent crashes if model fails to load

# Define class labels
CLASS_NAMES = [
    "Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back",
    "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"
]

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive!"}

@app.get("/")
async def root():
    return {"message": "Welcome to the Plant Disease Detection API 🌿"}

    # Capture model summary as string
    buffer = io.StringIO()
    sys.stdout = buffer
    MODEL.summary()
    sys.stdout = sys.__stdout__
    model_summary = buffer.getvalue()

    return {
        "input_shape": MODEL.input_shape,
        "model_summary": model_summary
    }

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, image



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        contents = await file.read()
        image_array, image = read_file_as_image(contents)

        predictions = MODEL.predict(image_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions[0]))

        return {
            "class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
