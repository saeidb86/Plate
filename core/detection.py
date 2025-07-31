import cv2
import torch
import numpy as np
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.eval()
    
    def detect(self, image):
        results = self.model(image)
        plates = []

        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            plate_img = image[y1:y2, x1:x2]
            plates.append({
                'image': plate_img,
                'bbox': (x1, y1, x2, y2),
                'confidence': conf
            })

        return plates