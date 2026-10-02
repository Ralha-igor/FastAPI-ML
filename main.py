import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="API Doenças do Café")

# Libera chamadas vindas do app (necessário para o navegador)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

CLASSES = [
    'bicho_mineiro',
    'cercosporiose',
    'ferrugem',
    'mancha_aureolada',
    'mancha_mantegosa',
    'phoma'
]

# Carrega o modelo apenas uma vez
MODEL_PATH = "modelo_cafe.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

@app.get("/")
def root():
    return {"status": "online", "modelo": "MobileNetV2 DigiPathos"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Formato de imagem inválido.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch, verbose=0)[0]
    top_index = int(np.argmax(predictions))

    return {
        "diagnostico": CLASSES[top_index],
        "confianca": round(float(predictions[top_index]) * 100, 2),
        "probabilidades": {CLASSES[i]: round(float(predictions[i]) * 100, 2) for i in range(len(CLASSES))}
    }
