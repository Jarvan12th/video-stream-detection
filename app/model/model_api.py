from fastapi import FastAPI, WebSocket
from starlette.responses import StreamingResponse
import cv2
import numpy as np
import asyncio
import torch
import requests
from pydantic import BaseModel
import os

app = FastAPI()

RTMP_URL = "rtmp://10.132.100.245:1900/live"

# Load the model
model = torch.load("./video_stream_detection_yolo_model/video_stream_detection_model.pth")

# Define the request and response models
class ImageURL(BaseModel):
    image_url: str


class PredictionResponse(BaseModel):
    prediction: str


def check_smoke(boxes):
    if boxes.shape[0] == 0:
        return "no smoke"
    else:
        return "smoke"


@app.post("/predict_url", response_model=PredictionResponse)
def predict_url(request: ImageURL):
    # Fetch the image from the provided URL
    response = requests.get(request.image_url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Get the provided image URL
    image_url = request.image_url

    # perform inference
    results = model.predict(image_url)

    # # observe results
    # print(results[0].boxes)
    # render = render_result(model=model, image=image_url, result=results[0])
    # render.show()

    # Delete the file after prediction
    filename = image_url.split("/")[-1]
    os.remove(filename)

    prediction = check_smoke(results[0].boxes)

    return {"prediction": prediction}

@app.websocket_route("/video_process")
async def video_process(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(RTMP_URL)  # Open the RTMP stream

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # If no frame is read, break the loop

            # Process the frame using the loaded model
            processed_frame = process_frame(frame)

            # Encode frame as JPEG and send it over WebSocket
            _, buffer = cv2.imencode('.jpg', processed_frame)
            await websocket.send_bytes(buffer.tobytes())
    finally:
        cap.release()  # Release the video capture object
        await websocket.close()

def process_frame(frame):
    return frame

    # # Convert the color space from BGR (OpenCV default) to RGB (expected by most PyTorch models)
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # # Convert the numpy array to a Torch tensor
    # frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()  # HWC to CHW and float
    # frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension

    # # You might need to adjust preprocessing steps based on your model requirements
    # # e.g., normalization, resizing, etc.

    # # Perform inference
    # with torch.no_grad():
    #     output = model(frame_tensor)

    # # Postprocess the output to draw bounding boxes or modify the frame based on model output
    # # This will depend on your model's specific output format

    # # Convert processed tensor back to numpy array
    # # Note: This step will depend on how you want to visualize the results
    # processed_frame = frame_tensor.squeeze(0).permute(1, 2, 0).int().numpy()  # Remove batch dim and CHW to HWC
    # processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV compatibility

    # return processed_frame


# The following code is for testing the API locally
if __name__ == "__main__":
    result = predict_url(
        ImageURL(
            # image_url="https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"
            # image_url="https://test-bucket-jarvan.s3.us-west-2.amazonaws.com/smoke.jpeg"
            image_url="https://test-bucket-jarvan.s3.us-west-2.amazonaws.com/apple.jpeg"
        )
    )
    print(result)
