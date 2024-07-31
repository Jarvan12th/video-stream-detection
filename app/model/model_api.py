from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import cv2
import ffmpeg
import numpy as np
import asyncio
import torch
import requests
from pydantic import BaseModel
import os

app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

RTMP_URL = "rtmp://54.253.170.76:1935/live/source"
STREAM_URL = "rtmp://54.253.170.76:1935/live/destination"  # Destination RTMP URL

# Load the model
model = torch.load("./video_stream_detection_yolo_model/video_stream_detection_model.pth")
# Move the model to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

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

@app.post("/start-streaming")
def start_streaming(background_tasks: BackgroundTasks):
    background_tasks.add_task(stream_video)
    return {"message": "Streaming started successfully"}

def stream_video():
    cap = cv2.VideoCapture(RTMP_URL)
    width, height = int(cap.get(3)), int(cap.get(4))
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}')
        .output(STREAM_URL, format='flv', vcodec='libx264', pix_fmt='yuv420p', preset='fast')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # If no frame is read, break the loop

            processed_frame = process_frame(frame)  # Process the frame using the loaded model
            # process.stdin.write(processed_frame.tobytes())  # Write the processed frame to FFmpeg stdin
            # Check if FFmpeg process is still running
            if process.poll() is None:
                process.stdin.write(processed_frame.tobytes())
            else:
                print("FFmpeg process has terminated unexpectedly.")
                break
    finally:
        cap.release()  # Release the video capture object
        # process.stdin.close()
        # process.wait()
        if process.poll() is None:
            process.stdin.close()
            process.wait()

@app.websocket_route("/video_process")
async def video_process(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(RTMP_URL)  # Open the RTMP stream

    # Setup FFmpeg stream process
    rtmp_url = "rtmp://192.168.1.3:1900/processed-video-stream"
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(int(cap.get(3)), int(cap.get(4))))
        .output(rtmp_url, format='flv', vcodec='libx264', pix_fmt='yuv420p', preset='fast')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # If no frame is read, break the loop

            # Process the frame using the loaded model
            processed_frame = process_frame(frame)

            # # Encode frame as JPEG and send it over WebSocket
            # _, buffer = cv2.imencode('.jpg', processed_frame)
            # await websocket.send_bytes(buffer.tobytes())

            # Write the processed frame to FFmpeg stdin
            process.stdin.write(
                processed_frame.tobytes()
            )
    finally:
        cap.release()  # Release the video capture object
        process.stdin.close()
        process.wait()
        await websocket.close()

def process_frame(frame):
    results = model.predict(frame)

    # Assuming 'results[0]' is a 'Boxes' object which has properties like 'xyxy' for coordinates
    boxes = results[0].boxes.xyxy  # This should give us the boxes in (x1, y1, x2, y2) format

    # Check and convert the boxes if it's a PyTorch tensor to a numpy array
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.numpy()

    annotated_frame = frame.copy()

    for box in boxes:
        # Convert box coordinates to integers for cv2.rectangle
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Draw rectangles on the frame
        annotated_frame = cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return annotated_frame

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
