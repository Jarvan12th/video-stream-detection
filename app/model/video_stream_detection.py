from ultralyticsplus import YOLO, render_result
import torch
import os
import shutil

# load model
model = YOLO('kittendev/YOLOv8m-smoke-detection')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# Delete the directory if it exists or create a new one
dir_path = './video_stream_detection_yolo_model'
if os.path.isdir(dir_path):
    # Remove the directory and all its content
    shutil.rmtree(dir_path)
    print(f"Directory '{dir_path}' has been removed.")

# Create the directory
os.makedirs(dir_path)
print(f"Directory '{dir_path}' has been created.")

# Save the model to a directory
torch.save(model, './video_stream_detection_yolo_model/video_stream_detection_model.pth')
