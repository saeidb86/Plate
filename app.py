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
import requests
from datetime import datetime, timedelta
from collections import deque

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

# Global variables for plate detection history
DETECTION_HISTORY = deque(maxlen=1000)  # ذخیره تاریخچه پلاک‌ها
API_ENDPOINT = "http://127.0.0.1:3000/monitoring"
LAST_SENT_TIME = {}

def get_rtsp_frame():
    global rtsp_cap
    try:
        if rtsp_cap is None:
            rtsp_cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            rtsp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            rtsp_cap.set(cv2.CAP_PROP_FPS, 30)  # محدود کردن نرخ فریم
            
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

async def check_and_send_plates():
    """
    تابعی که هر 5 ثانیه اجرا شده و تصاویر را برای پلاک بررسی می‌کند
    و در صورت وجود پلاک جدید، به API ارسال می‌کند
    """
    while True:
        try:
            await asyncio.sleep(5)  # هر 5 ثانیه
            
            frame = get_rtsp_frame()
            if frame is None:
                continue
                
            frame = cv2.resize(frame, (640, 480))
            plates = detector.detect(frame)
            
            current_time = datetime.now()
            
            for plate in plates:
                text = recognizer.recognize(plate['image'])
                confidence = plate['confidence']
                
                # اگر اعتماد به تشخیص کم است، نادیده بگیر
                if confidence < 0.5:
                    continue
                    
                # بررسی اینکه آیا این پلاک در 10 دقیقه گذشته ارسال شده
                if text in LAST_SENT_TIME:
                    time_diff = current_time - LAST_SENT_TIME[text]
                    if time_diff.total_seconds() < 600:  # 10 دقیقه
                        continue
                
                # ذخیره تصویر به صورت موقت
                _, buffer = cv2.imencode('.jpg', frame)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # آماده کردن داده برای ارسال
                payload = {
                    "cameraNumber": 1,  # شماره دوربین (می‌تواند تغییر کند)
                    "plate": text,
                    "enterDate": current_time.strftime("%Y-%m-%d"),
                    "enterTime": current_time.strftime("%H:%M:%S"),
                    "exitDate": current_time.strftime("%Y-%m-%d"),
                    "exitTime": current_time.strftime("%H:%M:%S"),
                    "image": image_base64
                }
                
                try:
                    # ارسال به API
                    response = requests.post(API_ENDPOINT, json=payload, timeout=10)
                    
                    if response.status_code == 201:
                        logger.info(f"پلاک {text} با موفقیت ارسال شد")
                        LAST_SENT_TIME[text] = current_time
                        DETECTION_HISTORY.append({
                            "plate": text,
                            "timestamp": current_time,
                            "sent": True
                        })
                    else:
                        logger.error(f"خطا در ارسال پلاک: {response.status_code}")
                        DETECTION_HISTORY.append({
                            "plate": text,
                            "timestamp": current_time,
                            "sent": False,
                            "error": response.text
                        })
                        
                except Exception as e:
                    logger.error(f"خطا در ارتباط با API: {str(e)}")
                    DETECTION_HISTORY.append({
                        "plate": text,
                        "timestamp": current_time,
                        "sent": False,
                        "error": str(e)
                    })
                    
        except Exception as e:
            logger.error(f"خطا در بررسی پلاک: {str(e)}")
            continue

@app.on_event("startup")
async def startup_event():
    """شروع وظیفه بررسی خودکار پلاک هنگام راه‌اندازی سرور"""
    asyncio.create_task(check_and_send_plates())

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
                
                continue
                
            frame = cv2.resize(frame, (640, 480))
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            
            await websocket.send_bytes(buffer.tobytes())
           
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

@app.get("/detection_history")
async def get_detection_history():
    """دریافت تاریخچه تشخیص پلاک‌ها"""
    history_list = []
    for item in DETECTION_HISTORY:
        history_list.append({
            "plate": item["plate"],
            "timestamp": item["timestamp"].isoformat(),
            "sent": item.get("sent", False),
            "error": item.get("error", "")
        })
    
    return {"history": history_list}

@app.get("/")
async def serve_demo():
    return FileResponse("demo.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8092, ws_ping_interval=10, ws_ping_timeout=10)