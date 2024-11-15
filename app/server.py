from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import io
import uvicorn

app = FastAPI()

# Load your model
model = load_model("app/classifier_resnet_model.h5")

# Preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224), Image.LANCZOS)  # Resize to the target size
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image
    return img_array

# Interpret the predictions
def interpret_predictions(predictions):
    class_labels = ["no crack", "crack"]
    predicted_class = np.argmax(predictions, axis=1)
    return class_labels[predicted_class[0]]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    # Preprocess the image
    img_array = preprocess_image(img)
    
    # Make prediction
    predictions = model.predict(img_array)
    result = interpret_predictions(predictions)

    return JSONResponse(content={"prediction": result})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)