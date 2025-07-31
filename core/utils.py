import cv2
import numpy as np

def draw_results(image, results):
    """رسم نتایج روی تصویر"""
    for result in results:
        x1, y1, x2, y2 = result['bbox']
        text = result['text']
        confidence = result['confidence']
        
        # رسم مستطیل دور پلاک
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # نمایش متن و اطمینان
        label = f"{text} ({confidence:.2f})"
        cv2.putText(image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return image

def preprocess_plate(plate_image, target_size=(128, 32)):
    """پیش‌پردازش تصویر پلاک برای CRNN"""
    plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    plate_image = cv2.resize(plate_image, target_size)
    plate_image = plate_image.astype(np.float32) / 255.0
    plate_image = (plate_image - 0.5) / 0.5  # نرمال‌سازی
    return plate_image