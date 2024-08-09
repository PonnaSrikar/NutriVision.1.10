import io
import base64
import asyncio
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model with FP16 precision for faster inference if supported
model = YOLO("best.pt", device="cuda:0", half=True)

async def process_image(image):
    detections = model.predict(image, conf=0.25)[0]  # Batched prediction
    data = []

    # Process and plot detection directly in memory
    plot_image = detections.plot(show=False, render=True)
    _, encoded_img = cv2.imencode('.png', plot_image)
    processed_image = base64.b64encode(encoded_img).decode('utf-8')

    for box in detections.boxes:
        label = detections.names[int(box.cls)]
        confidence = float(box.conf)
        area = cv2.contourArea(cv2.convexHull(np.array(box.xy, dtype=np.int32).reshape(-1, 2)))

        data.append([label, confidence, area])

    return data, processed_image

@app.post("/detect/")
async def detect_objects(file: UploadFile):
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Process the image asynchronously to leverage multi-core CPUs/GPUs
    data, processed_image = await asyncio.to_thread(process_image, image)

    return JSONResponse(content={"detections": data, "image": processed_image})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
