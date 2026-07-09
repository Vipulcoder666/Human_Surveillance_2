from ultralytics import YOLO
import cv2
import os

# Set RTSP options
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

print("Loading custom model...")
model = YOLO("best.pt")

CAMERA_URL = "rtsp://admin:cctv%40321@192.168.1.72:554/cam/realmonitor?channel=6&subtype=0"

print("Connecting to camera stream...")
cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Could not connect to stream. Please check connection.")
    exit()

print("Capturing fresh frame...")
for _ in range(10):
    ret, frame = cap.read()

if ret and frame is not None:
    oh, ow = frame.shape[:2]
    
    # 1. Run on Full Frame
    print("Inference on full frame...")
    results_full = model(frame, conf=0.05)
    results_full[0].save("scores_check_full.jpg")
    
    # 2. Run on Cropped Zoom (Right 75% of the frame where desks are located)
    print("Inference on zoomed crop...")
    cx0 = ow // 4
    crop = frame[0:oh, cx0:ow]
    results_crop = model(crop, conf=0.05)
    results_crop[0].save("scores_check_crop.jpg")
    
    print("\nSaved debug images:")
    print(f"- Full Frame: {os.path.abspath('scores_check_full.jpg')}")
    print(f"- Zoomed Crop: {os.path.abspath('scores_check_crop.jpg')}")
else:
    print("Error: Failed to retrieve frame.")

cap.release()
