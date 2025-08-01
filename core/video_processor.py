import cv2
import numpy as np
from io import BytesIO
import tempfile
import os

class VideoProcessor:
    def __init__(self, detector, recognizer):
        self.detector = detector
        self.recognizer = recognizer
        self.frame_skip = 3
        self.min_confidence = 0.6

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
            temp_path = temp_output.name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        frame_count = 0
        results = []  # List to store plate detection results

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue

            plates = self.detector.detect(frame)
            for plate in plates:
                if plate['confidence'] < self.min_confidence:
                    continue

                text = self.recognizer.recognize(plate['image'])
                x1, y1, x2, y2 = plate['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Store plate info
                results.append({
                    'text': text,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(plate['confidence']),
                    'frame_number': frame_count
                })

            out.write(frame)

        cap.release()
        out.release()

        # Read processed video into memory
        with open(temp_path, 'rb') as f:
            video_bytes = f.read()
        
        # Clean up
        try:
            os.unlink(temp_path)
        except Exception as e:
            print(f"Warning: Could not delete temp file {temp_path}: {e}")

        output_buffer = BytesIO()
        output_buffer.write(video_bytes)
        output_buffer.seek(0)
        
        return output_buffer, results