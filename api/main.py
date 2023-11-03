from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import os
import tensorflow as tf

app = FastAPI()

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
# Construct the absolute path to the model directory
mixed_path = os.path.join(current_dir, '../../plant_disease_detection_backend/my_model_version1')
normalized_path = os.path.normpath(mixed_path)

normalized_path = normalized_path.replace("\\", "/")
print(normalized_path)
# Load the model
MODEL = tf.keras.models.load_model(normalized_path)

# # the old solution
# MODEL = tf.keras.models.load_model("../my_model_version1")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get('/ping')
async def ping():
    return "Hello, I am alive. yeah"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post('/predict')
async def predict(
        # file of type UploadFile and its default value is File(...)
        file: UploadFile = File(...)
):

    image = read_file_as_image(await file.read())
    # convert image from something like this [232,543,32] to this [[232,543,32]]
    # because predict function should be given a list of images not only one image
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)  # because he treated it as a batch of image not only one image
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
