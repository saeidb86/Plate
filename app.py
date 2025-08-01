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
from datetime import datetime
import uuid

app = FastAPI()

# Initialize models
CHAR_SET = "0123456789ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوه‍ی ()"
detector = PlateDetector("models/yolo_plate/best.pt")
recognizer = PlateRecognizer("models/best_model.pth", CHAR_SET)
video_processor = VideoProcessor(detector, recognizer)

# Directory to store temporary captures
TEMP_DIR = "temp_captures"
os.makedirs(TEMP_DIR, exist_ok=True)

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
                'confidence': plate['confidence'],
                'timestamp': datetime.now().isoformat(),
                'source': 'image_upload'
            })

            x1, y1, x2, y2 = plate['bbox']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save processed image
        processed_img_path = os.path.join(TEMP_DIR, f"processed_{uuid.uuid4()}.jpg")
        cv2.imwrite(processed_img_path, image)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse(content={
            'status': 'success',
            'results': results,
            'image': image_base64,
            'processed_image_path': processed_img_path,
            'timestamp': datetime.now().isoformat()
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
            
            # Save processed video
            processed_video_path = os.path.join(TEMP_DIR, f"processed_{uuid.uuid4()}.mp4")
            with open(processed_video_path, 'wb') as f:
                f.write(video_bytes)
            
            return JSONResponse(content={
                'status': 'success',
                'results': results,
                'metadata': metadata,
                'video': video_base64,
                'processed_video_path': processed_video_path,
                'timestamp': datetime.now().isoformat()
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

@app.post("/recognize_camera")
async def recognize_camera(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        # Enhance image for better detection (optional)
        image = enhance_image(image)
        
        plates = detector.detect(image)
        results = []

        for plate in plates:
            text = recognizer.recognize(plate['image'])
            results.append({
                'text': text,
                'bbox': plate['bbox'],
                'confidence': plate['confidence'],
                'timestamp': datetime.now().isoformat(),
                'source': 'camera_capture'
            })

            x1, y1, x2, y2 = plate['bbox']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save original and processed images
        original_path = os.path.join(TEMP_DIR, f"original_{uuid.uuid4()}.jpg")
        processed_path = os.path.join(TEMP_DIR, f"processed_{uuid.uuid4()}.jpg")
        
        cv2.imwrite(original_path, cv2.imdecode(np.frombuffer(contents, np.uint8)))
        cv2.imwrite(processed_path, image)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse(content={
            'status': 'success',
            'results': results,
            'image': image_base64,
            'original_image_path': original_path,
            'processed_image_path': processed_path,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing camera image: {str(e)}")

@app.get("/get_processed_file/{file_type}/{filename}")
async def get_processed_file(file_type: str, filename: str):
    try:
        file_path = os.path.join(TEMP_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        if file_type == 'image':
            return FileResponse(file_path, media_type="image/jpeg")
        elif file_type == 'video':
            return FileResponse(file_path, media_type="video/mp4")
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cleanup_temp_files")
async def cleanup_temp_files(hours_old: int = 24):
    try:
        now = datetime.now()
        deleted_files = []
        
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if (now - file_time).total_seconds() > hours_old * 3600:
                os.unlink(file_path)
                deleted_files.append(filename)
        
        return JSONResponse(content={
            'status': 'success',
            'deleted_files': deleted_files,
            'count': len(deleted_files)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def enhance_image(image):
    """Enhance image quality for better detection"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L channel with the a and b channel
    limg = cv2.merge((cl, a, b))
    
    # Convert back to BGR color space
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Optional: denoising
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    return enhanced

@app.get("/")
async def serve_demo():
    return FileResponse("demo.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8089)