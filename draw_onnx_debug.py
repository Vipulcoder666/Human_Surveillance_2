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

def run_onnx_inference(sess, inp_name, frame):
    lb, scale, pl, pt = letterbox(frame)
    blob = lb[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    
    out = sess.run(None, {inp_name: blob})[0][0]  # Shape (5, 8400)
    
    cx, cy, bw, bh = out[0], out[1], out[2], out[3]
    scores = out[4]
    
    mask = scores >= 0.10
    if not mask.any():
        return [], []
        
    s = scores[mask]
    oh, ow = frame.shape[:2]
    
    x1 = np.clip(((cx[mask] - bw[mask]/2) - pl) / scale, 0, ow).astype(int)
    y1 = np.clip(((cy[mask] - bh[mask]/2) - pt) / scale, 0, oh).astype(int)
    x2 = np.clip(((cx[mask] + bw[mask]/2) - pl) / scale, 0, ow).astype(int)
    y2 = np.clip(((cy[mask] + bh[mask]/2) - pt) / scale, 0, oh).astype(int)
    
    return list(zip(x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist())), s.tolist()

print("Loading ONNX model...")
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name

print("Connecting to camera stream...")
cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Could not open stream.")
    exit()

print("Capturing frame...")
# Grab 30 frames to flush old buffers completely
for _ in range(30):
    ret, frame = cap.read()

if ret and frame is not None:
    oh, ow = frame.shape[:2]
    if ow > 800:
        scale = 800.0 / ow
        frame = cv2.resize(frame, (800, int(oh * scale)))
    
    # 1. Full Frame Inference
    boxes_full, scores_full = run_onnx_inference(sess, inp_name, frame)
    
    # Draw on Full Frame (Red boxes)
    canvas_full = frame.copy()
    for box, score in zip(boxes_full, scores_full):
        x1, y1, x2, y2 = box
        cv2.rectangle(canvas_full, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(canvas_full, f"{score:.2f}", (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    cv2.imwrite("onnx_debug_full.jpg", canvas_full)
    
    # 2. Cropped Zoom Inference
    cx0 = ow // 4
    crop = frame[0:oh, cx0:ow]
    boxes_crop, scores_crop = run_onnx_inference(sess, inp_name, crop)
    
    # Draw on Crop (Blue boxes, mapped to full frame coordinates)
    canvas_crop = frame.copy()
    for box, score in zip(boxes_crop, scores_crop):
        x1, y1, x2, y2 = box
        x1_f, x2_f = x1 + cx0, x2 + cx0
        cv2.rectangle(canvas_crop, (x1_f, y1), (x2_f, y2), (255, 0, 0), 1)
        cv2.putText(canvas_crop, f"{score:.2f}", (x1_f, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
    cv2.imwrite("onnx_debug_crop.jpg", canvas_crop)
    
    print("\nSaved ONNX debug images:")
    print(f"- Full Frame: {os.path.abspath('onnx_debug_full.jpg')}")
    print(f"- Zoomed Crop: {os.path.abspath('onnx_debug_crop.jpg')}")
else:
    print("Error: Failed to retrieve frame.")

cap.release()
