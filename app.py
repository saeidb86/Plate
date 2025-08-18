from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Response
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
logging.basicConfig(level=logging.INFO)
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

# RTSP camera configuration
RTSP_URL = "rtsp://admin:Abc123456@192.168.100.104:554/main"
rtsp_cap = None

def get_rtsp_frame():
    global rtsp_cap
    try:
        if rtsp_cap is None:
            rtsp_cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            rtsp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            rtsp_cap.set(cv2.CAP_PROP_FPS, 15)  # محدود کردن نرخ فریم
            
        for _ in range(3):  # تلاش چندباره برای دریافت فریم
            ret, frame = rtsp_cap.read()
            if ret:
                return frame
            else:
                rtsp_cap.release()
                rtsp_cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                
        return None
    except Exception as e:
        logger.error(f"RTSP Error: {str(e)}")
        return None
@app.get("/test_rtsp_connection")
async def test_rtsp_connection():
    try:
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Cannot open RTSP stream")
        cap.release()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/camera_feed")
async def camera_feed():
    frame = get_rtsp_frame()
    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame")
    
    frame = cv2.resize(frame, (640, 480))
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.get("/recognize_camera")
async def recognize_camera():
    try:
        frame = get_rtsp_frame()
        if frame is None:
            raise ValueError("Failed to capture frame")
        
        frame = cv2.resize(frame, (640, 480))
        plates = detector.detect(frame)
        results = []

        for plate in plates:
            text = recognizer.recognize(plate['image'])
            results.append({
                'text': text,
                'confidence': plate['confidence']
            })

            x1, y1, x2, y2 = plate['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'status': 'success',
            'results': results,
            'image': image_base64
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            frame = get_rtsp_frame()
            if frame is None:
                await asyncio.sleep(0.05)
                continue
                
            frame = cv2.resize(frame, (640, 480))
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            
            await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.05)  # ~20 FPS
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

# سایر endpointها (recognize_image, recognize_video و...) مانند قبل باقی می‌مانند
# ...

@app.get("/")
async def serve_demo():
    return FileResponse("demo.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8092, ws_ping_interval=10, ws_ping_timeout=10)