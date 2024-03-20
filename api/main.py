from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import os
import tensorflow as tf
from rembg import remove
from starlette.responses import StreamingResponse
import cv2
app = FastAPI()

input_shape = (227, 227, 3)
# Load the model

MODELS_DATA = {
    'apple': {
        'path': '../models/apple_model_alexnet_without_augmentation2',
        'class_names': ["Scab", "Black Rot", "Cedar Rust", "Healthy"],
    },
    'potato': {
        'path': '../models/potato_model_version`',
        'class_names': ["Early Blight", "Late Blight", "Healthy"],
    },
    'strawberry': {
        'path': '../models/strawberry_model_version1',
        'class_names': ["Leaf scorch", "Healthy"],
    },
    'cotton': {
        'path': '../models/cotton_model_version1',
        'class_names': ["Diseased plant", "Healthy plant"],
    },
    'cherry': {
        'path': '../models/cherry_model_version1',
        'class_names': ["Powdery mildew", "Healthy"],
    },
    'peach': {
        'path': '../models/peach_model_version1',
        'class_names': ["Healthy","Bacterial spot"],
    },
    'grape': {
        'path': '../models/grape_model_version1',
        'class_names': ["Leaf_blight (Isariopsis Leaf Spot)", "Healthy", "Esca (Black Measles)", "Black rot"],
    },
    'pepper': {
        'path': '../models/pepper_model_version1',
        'class_names': ["Bacterial spot", "Healthy"],
    },
    'corn': {
        'path': '../models/corn_model_version1',
        'class_names': ["Blight", "Common Rust", "Gray Leaf Spot", "Healthy"],
    },
    'wheat': {
        'path': '../models/wheat_model_version1',
        'class_names': ["Brown rust", "Healthy", "Yellow rust"],
    },
    'tomato': {
        'path': '../models/tomato_model_version1',
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


def read_file_as_image(data, target_size=(227, 227)) :#-> np.ndarray: # Image.image
    image = Image.open(BytesIO(data)).resize(target_size)
    image = remove(image) # remove the image background
    image = image.convert('RGB')
    image = segment_image(image)

    print(image)
    return image#np.array(image) # image


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
    model = tf.keras.saving.load_model(normalized_path)
    return model


def get_model_and_class_names(name: str):
    normalized_path = process_file_path(MODELS_DATA[name]['path'])  # '../apple_model_version1'
    model = get_model(normalized_path)
    class_names = MODELS_DATA[name]['class_names']
    return model, class_names


def segment_image(image):
    # Convert PIL image to numpy
    image_np = np.array(image)
    # print(image)
    # Convert to BGR (OpenCV uses BGR by default)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Define color ranges for the spots (excluding green and including rgba(42, 60, 24, 255))
    lower_spots1 = np.array([5, 50, 50])  # Lower HSV values for spots, excluding green
    upper_spots1 = np.array([25, 255, 255])  # Upper HSV values for spots, excluding green
    mask_spots1 = cv2.inRange(hsv, lower_spots1, upper_spots1)

    # Define additional mask for very dark colors (low value) for spot2
    lower_spots_dark = np.array([0, 0, 0])  # Lower HSV values for very dark colors
    upper_spots_dark = np.array([180, 255, 50])  # Upper HSV values for very dark colors
    mask_spots_dark = cv2.inRange(hsv, lower_spots_dark, upper_spots_dark)


    lower_dark_green = np.array([30, 100, 0])
    upper_dark_green = np.array([47, 255, 100])
    mask_dark_green = cv2.inRange(hsv, lower_dark_green, upper_dark_green)

    # Combine the masks for all spots
    mask_spots_combined = cv2.bitwise_or(mask_spots1, mask_spots_dark)
    mask_spots_combined = cv2.bitwise_or(mask_spots_combined, mask_dark_green)

    # Apply the combined mask to get the segmented image (spots only)
    result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_spots_combined)

    # Convert masked image back to RGB for visualization
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # Convert number array to PIL image
    result_rgb_PIL = Image.fromarray(result_rgb)

    return result_rgb_PIL


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
    predictions = model(img_batch)  # because he treated it as a batch of image not only one image
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    print(predictions[0])


    # return {
    #     'plant_name': name,
    #     'class': predicted_class,
    #     'confidence': float(confidence)
    # }


    # # return the image
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    return StreamingResponse(buffered, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
