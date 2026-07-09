import os
# Force FFMPEG to use TCP transport
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "best.onnx"
CAMERA_URL = "rtsp://admin:cctv%40321@192.168.1.72:554/cam/realmonitor?channel=6&subtype=0"
INPUT_SIZE = 640

def letterbox(img, size=INPUT_SIZE):
    h, w    = img.shape[:2]
    scale   = size / max(h, w)
    nh, nw  = int(h * scale), int(w * scale)
    canvas  = np.full((size, size, 3), 114, dtype=np.uint8)
    pl = (size - nw) // 2
    pt = (size - nh) // 2
    canvas[pt:pt+nh, pl:pl+nw] = cv2.resize(img, (nw, nh))
    return canvas, scale, pl, pt

print("Loading ONNX model...")
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name

print("Connecting to camera stream...")
cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Could not open stream.")
    exit()

print("Capturing frame...")
for _ in range(10):
    ret, frame = cap.read()

if ret and frame is not None:
    # Downscale frame to 800 width (just like in the surveillance script)
    oh, ow = frame.shape[:2]
    if ow > 800:
        scale = 800.0 / ow
        frame = cv2.resize(frame, (800, int(oh * scale)))
    
    # Crop the right 75%
    oh_crop, ow_crop = frame.shape[:2]
    cx0 = ow_crop // 4
    crop = frame[0:oh_crop, cx0:ow_crop]
    
    # Letterbox and run inference
    lb, scale, pl, pt = letterbox(crop)
    blob = lb[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    
    out = sess.run(None, {inp_name: blob})[0][0]  # Shape (5, 8400)
    
    cx, cy, bw, bh = out[0], out[1], out[2], out[3]
    scores = out[4]
    
    # Sort detections by score descending
    sorted_indices = np.argsort(scores)[::-1]
    
    print("\n--- Top 15 Raw ONNX Detections in Zoomed Crop (Before NMS) ---")
    count = 0
    for idx in sorted_indices:
        score = scores[idx]
        if score < 0.05:
            break
        
        # Calculate full-frame coordinate
        x = cx[idx]
        y = cy[idx]
        w_box = bw[idx]
        h_box = bh[idx]
        
        x1 = int(((x - w_box/2) - pl) / scale) + cx0
        y1 = int(((y - h_box/2) - pt) / scale)
        x2 = int(((x + w_box/2) - pl) / scale) + cx0
        y2 = int(((y + h_box/2) - pt) / scale)
        
        print(f"Detection {count+1}: Score={score:.4f}, Coordinates=[{x1}, {y1}, {x2}, {y2}]")
        count += 1
        if count >= 15:
            break
else:
    print("Error: Failed to retrieve frame.")

cap.release()
