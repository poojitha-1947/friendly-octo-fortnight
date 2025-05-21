from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import uvicorn
import os
from tensorflow.keras.models import load_model

# Load the trained model
# MODEL_PATH = "land_use_model.h5"
MODEL_PATH ="C:/Users/pooji/OneDrive/Desktop/minimini/land_use_model.h5"

CLASS_NAMES_PATH = "classes.txt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Ensure the file exists.")
model = load_model(MODEL_PATH)

# Load class names
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"Class names file not found at {CLASS_NAMES_PATH}. Ensure the file exists.")
with open(CLASS_NAMES_PATH, "r") as file:
    class_names = [line.strip() for line in file]

app = FastAPI()

# Deforestation Detection Function
def detect_deforestation(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))

    diff = cv2.absdiff(img1, img2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marked_img = img2.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 20:  # Filter small changes
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    total_pixels = thresh.size
    deforested_pixels = np.sum(thresh > 0)
    deforested_percentage = (deforested_pixels / total_pixels) * 100

    return marked_img, deforested_percentage

def classify_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]
    return class_idx, confidence

@app.get("/")
def root():
    return HTMLResponse("""
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="image1" accept="image/*">
        <input type="file" name="image2" accept="image/*">
        <button type="submit">Upload</button>
    </form>
    """)

@app.post("/upload")
async def upload_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    img1_path = "temp1.jpg"
    img2_path = "temp2.jpg"

    with open(img1_path, "wb") as buffer:
        buffer.write(await image1.read())
    with open(img2_path, "wb") as buffer:
        buffer.write(await image2.read())

    class1_idx, conf1 = classify_image(img1_path)
    class2_idx, conf2 = classify_image(img2_path)
    marked_img, deforested_percentage = detect_deforestation(img1_path, img2_path)

    marked_path = "marked.jpg"
    cv2.imwrite(marked_path, marked_img)

    return {
        "message": "Analysis completed successfully!",
        "deforested_percentage": f"{deforested_percentage:.2f}%",
        "marked_image_path": marked_path,
        "image1_class": class_names[class1_idx],
        "image1_confidence": f"{conf1:.2f}",
        "image2_class": class_names[class2_idx],
        "image2_confidence": f"{conf2:.2f}"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
