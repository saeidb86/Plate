from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
import cv2
import numpy as np
from core.detection import PlateDetector
from core.recognition import PlateRecognizer
from core.video_processor import VideoProcessor
import base64
from io import BytesIO
import tempfile
import os

app = FastAPI()

# Initialize models
CHAR_SET = "0123456789ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوه‍ی ()"
detector = PlateDetector("models/yolo_plate/best.pt")
recognizer = PlateRecognizer("models/best_model.pth", CHAR_SET)
video_processor = VideoProcessor(detector, recognizer)

@app.post("/recognize_image")
async def recognize_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        plates = detector.detect(image)
        results = []

        for plate in plates:
            text = recognizer.recognize(plate['image'])
            results.append({
                'text': text,
                'bbox': plate['bbox'],
                'confidence': plate['confidence']
            })

            x1, y1, x2, y2 = plate['bbox']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse(content={
            'results': results,
            'image': image_base64
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/recognize_video")
async def recognize_video(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Get video metadata
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            metadata = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            }
            cap.release()

            # Process video
            output_buffer, results = video_processor.process_video(tmp_path)
            
            # Convert video to base64
            video_bytes = output_buffer.getvalue()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            
            # Clean up
            try:
                os.unlink(tmp_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file {tmp_path}: {e}")
            
            return JSONResponse(content={
                'status': 'success',
                'results': results,
                'metadata': metadata,
                'video': video_base64
            })
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as ex:
                    print(f"Warning: Could not delete temp file {tmp_path}: {ex}")
            raise e
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/")
async def serve_demo():
    return FileResponse("demo.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8086)