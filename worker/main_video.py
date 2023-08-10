import numpy as np
import cv2
import sys
import time
import os
import torch
import traceback
sys.path.insert(0, "/Users/priyesh/Workspace/greencross/yolov5")
from app.helper.detector import TorchDetector
model_path = "/Users/priyesh/Workspace/greencross/yolov5/runs/train/exp3/weights/best.pt"
video_path = "/Users/priyesh/Workspace/greencross/trial/videos/test3.mp4"

dimension = (640, 640)
detector = TorchDetector(model_path, dimension)
# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Process each frame in the video
while True:
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break
    resized_image = cv2.resize(frame, dimension)
    coordinates_list = detector.detect(resized_image, (1,1), dimension, 0.16)

    for coords in coordinates_list:
        x, y, width, height = map(int, coords)
        cv2.rectangle(resized_image, (x, y), (x + width, y + height), (0, 0, 255), 2)

    resized_image = cv2.resize(resized_image, (640,360))
    # Display the processed frame (optional)
    cv2.imshow("Processed Frame", resized_image)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()



