
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
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
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
        if not contents:
            raise ValueError("Empty image data received")
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image file: Failed to decode")
        
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
        
        processed_img_path = os.path.join(TEMP_DIR, f"processed_{uuid.uuid4()}.jpg")
        cv2.imwrite(processed_img_path, image)
        
        _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse(content={
            'status': 'success',
            'results': results,
            'image': image_base64,
            'processed_image_path': processed_img_path,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/recognize_video")
async def recognize_video(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
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

            output_buffer, results = video_processor.process_video(tmp_path)
            
            video_bytes = output_buffer.getvalue()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            
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
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as ex:
                    logger.warning(f"Could not delete temp file {tmp_path}: {ex}")
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/recognize_camera")
async def recognize_camera(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise ValueError("Empty camera capture received")
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid camera capture: Failed to decode image")
        
        target_width = 640
        aspect_ratio = image.shape[1] / image.shape[0]
        target_height = int(target_width / aspect_ratio)
        image = cv2.resize(image, (target_width, target_height))
        
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
        
        original_path = os.path.join(TEMP_DIR, f"original_{uuid.uuid4()}.jpg")
        processed_path = os.path.join(TEMP_DIR, f"processed_{uuid.uuid4()}.jpg")
        
        original_image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if original_image is None:
            raise ValueError("Invalid camera capture: Failed to decode original image")
        cv2.imwrite(original_path, original_image)
        cv2.imwrite(processed_path, image)
        
        _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
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
        logger.error(f"Error processing camera image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing camera image: {str(e)}")

@app.websocket("/recognize_camera_stream")
async def recognize_camera_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    frame_count = 0
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=10.0)
                logger.debug(f"Received {len(data)} bytes")
                
                if not data:
                    logger.warning("Received empty data")
                    continue
                
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    logger.warning("Received invalid image data")
                    continue
                
                frame_count += 1
                if frame_count % 2 != 0:  # Process every other frame
                    logger.debug("Skipping frame")
                    continue
                
                # Resize image
                target_width = 640
                aspect_ratio = image.shape[1] / image.shape[0]
                target_height = int(target_width / aspect_ratio)
                image = cv2.resize(image, (target_width, target_height))
                
                plates = detector.detect(image)
                results = []

                for plate in plates:
                    text = recognizer.recognize(plate['image'])
                    results.append({
                        'text': text,
                        'bbox': plate['bbox'],
                        'confidence': plate['confidence'],
                        'timestamp': datetime.now().isoformat(),
                        'source': 'camera_stream'
                    })

                    x1, y1, x2, y2 = plate['bbox']
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Encode to base64
                _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                image_base64 = base64.b64encode(buffer).decode('utf-8')

                await websocket.send_json({
                    'status': 'success',
                    'results': results,
                    'image': image_base64,
                    'timestamp': datetime.now().isoformat()
                })
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for WebSocket data")
                continue
            except WebSocketDisconnect as e:
                logger.info(f"WebSocket client disconnected: {e.code}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket processing: {str(e)}", exc_info=True)
                await websocket.send_json({
                    'status': 'error',
                    'detail': str(e)
                })
    finally:
        try:
            await websocket.close()
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.warning(f"Error closing WebSocket: {str(e)}")

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
        logger.error(f"Error retrieving file: {str(e)}", exc_info=True)
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
        logger.error(f"Error cleaning up temp files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def enhance_image(image):
    """Lightweight image enhancement for better detection"""
    try:
        alpha = 1.2
        beta = 10
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}", exc_info=True)
        raise

@app.get("/")
async def serve_demo():
    return FileResponse("demo.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8089)
