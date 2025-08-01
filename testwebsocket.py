import websockets
import asyncio
import cv2

async def test_client():
    uri = "ws://127.0.0.1:8089/recognize_camera_stream"
    async with websockets.connect(uri) as websocket:
        image = cv2.imread("test_image.jpg")  # یک تصویر JPEG معتبر
        if image is None:
            print("Failed to load test image")
            return
        _, buffer = cv2.imencode('.jpg', image)
        await websocket.send(buffer.tobytes())
        response = await websocket.recv()
        print("Response from server:", response)

asyncio.run(test_client())