from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import os
import tensorflow as tf
from rembg import remove
from starlette.responses import StreamingResponse
app = FastAPI()

input_shape = (256, 256, 3)
# Load the model

MODELS_DATA = {
    'apple': {
        'path': '../apple_model_version2',
        'class_names': ["Black Rot", "Cedar Rust", "Scab", "Healthy"],
    },
    'potato': {
        'path': '../potato_model_version`',
        'class_names': ["Early Blight", "Late Blight", "Healthy"],
    },
    'strawberry': {
        'path': '../strawberry_model_version1',
        'class_names': ["Leaf scorch", "Healthy"],
    },
    'cotton': {
        'path': '../cotton_model_version1',
        'class_names': ["Diseased plant", "Healthy plant"],
    },
    'cherry': {
        'path': '../cherry_model_version1',
        'class_names': ["Powdery mildew", "Healthy"],
    },
    'peach': {
        'path': '../peach_model_version1',
        'class_names': ["Healthy","Bacterial spot"],
    },
    'grape': {
        'path': '../grape_model_version1',
        'class_names': ["Leaf_blight (Isariopsis Leaf Spot)", "Healthy", "Esca (Black Measles)", "Black rot"],
    },
    'pepper': {
        'path': '../pepper_model_version1',
        'class_names': ["Bacterial spot", "Healthy"],
    },
    'corn': {
        'path': '../corn_model_version1',
        'class_names': ["Blight", "Common Rust", "Gray Leaf Spot", "Healthy"],
    },
    'wheat': {
        'path': '../wheat_model_version1',
        'class_names': ["Brown rust", "Healthy", "Yellow rust"],
    },
    'tomato': {
        'path': '../tomato_model_version1',
        'class_names': [
            "Bacterial spot",
            "Early blight",
            "Late blight",
            "Leaf Mold",
            "Septoria leaf spot",
            "Spider Mites (Two spotted spider mite)",
            "Target Spot",
            "Yellow Leaf Curl Virus",
            "Mosaic Virus",
            "Healthy",
        ],
    }
}



@app.get('/ping')
async def ping():
    return "Hello, I am alive. yeah"


def read_file_as_image(data, target_size=(256, 256)) -> np.ndarray: # Image.image
    image = Image.open(BytesIO(data)).resize(target_size)
    image = remove(image) # remove the image background
    image = image.convert('RGB')
    return np.array(image) # image


def process_file_path(path: str) -> str:
    # Get the absolute path of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)

    mixed_path = os.path.join(current_dir, path)

    # Construct the absolute path to the model directory

    normalized_path = os.path.normpath(mixed_path)

    normalized_path = normalized_path.replace("\\", "/")

    return normalized_path


def get_model(normalized_path: str):
    model = tf.keras.models.load_model(normalized_path)
    return model


def get_model_and_class_names(name: str):
    normalized_path = process_file_path(MODELS_DATA[name]['path'])  # '../apple_model_version1'
    model = get_model(normalized_path)
    class_names = MODELS_DATA[name]['class_names']
    return model, class_names


@app.post('/predict')
async def predict(
        # file of type UploadFile and its default value is File(...)
        file: UploadFile = File(...),
        name: str = None
):
    image = read_file_as_image(await file.read())
    # convert image from something like this [232,543,32] to this [[232,543,32]]
    # because predict function should be given a list of images not only one image
    img_batch = np.expand_dims(image, 0)
    if name is None:
        return {'error': 'You should specify the name of the plant'}
    elif name not in MODELS_DATA:
        return {'error': 'We don\'t have this plant , go somewhere else'}
    model, class_names = get_model_and_class_names(name)
    predictions = model.predict(img_batch)  # because he treated it as a batch of image not only one image
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    print(predictions[0])


    return {
        'plant_name': name,
        'class': predicted_class,
        'confidence': float(confidence)
    }


    # # return the image
    # buffered = BytesIO()
    # image.save(buffered, format="PNG")
    # buffered.seek(0)
    # return StreamingResponse(buffered, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
