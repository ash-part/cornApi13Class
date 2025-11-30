from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import os, io
import tensorflow as tf

app = FastAPI()

# ✅ Corn disease class array
class_names = [
    "Anthracnose Stalk Rot",
    "Aspergillus Ear Rot",
    "Charcoal Stalk Rot",
    "Common_Rust",
    "Diplodia Ear Rot",
    "Fusarium Ear Rot",
    "Giberella Stalk Rot",
    "Gray_Leaf_Spot",
    "Northern Leaf Blight",
    "Penicillium Ear Rot",
    "Sheath Blight",
    "Southern Leaf Blight",
    "YellowLeafSpot"
]

# ✅ Load saved model from current directory
MODEL_PATH = os.path.join(os.getcwd(), "corn_disease_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

@api.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # ✅ Resize image
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))

    # ✅ Normalize
    image_arr = np.array(img) / 255.0
    image_arr = np.expand_dims(image_arr, axis=0)

    # Inference
    preds = model.predict(image_arr)
    idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))

    return {
        "predicted_class": class_names[idx],
        "confidence": confidence,
        "probabilities": {name: float(preds[0][i]) for i, name in enumerate(class_names)}
    }
