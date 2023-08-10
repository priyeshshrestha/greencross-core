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
# image_path = "/Users/priyesh/Workspace/greencross/trial/images"
image_path = "/Users/priyesh/Workspace/greencross/yolov5/test/images"

for imagename in os.listdir(image_path):
    try:
        image = cv2.imread(os.path.join(image_path,imagename))
        desired_size = (640, 640)
        resized_image = cv2.resize(image, desired_size)

        detector = TorchDetector(model_path, (640, 640))
        # coordinates_list = detector.detect(resized_image, (1,1), (640,640), 0.22)
        coordinates_list = detector.detect(resized_image, (1,1), (640,640), 0.5)
    
        for coords in coordinates_list:
            x, y, width, height = map(int, coords)
            cv2.rectangle(resized_image, (x, y), (x + width, y + height), (0, 0, 255), 2)

        # Display the image with drawn squares
        cv2.imshow('Image with Squares', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as inst:
        traceback.print_exc()

