import os
# Force FFMPEG to use TCP transport and set connection timeouts globally on startup
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|timeout;5000000"

import cv2
import time
import threading
import numpy as np
import onnxruntime as ort

# ============================================================
# CONFIGURATION  ← Edit only here
# ============================================================
MODEL_PATH       = "best.onnx"      # Custom trained YOLO11n model
# Direct RTSP Stream:
CAMERA_URL       = "rtsp://admin:cctv%40321@192.168.1.72:554/cam/realmonitor?channel=6&subtype=0"
# Raspberry Pi HTTP Streamer:
# CAMERA_URL     = "http://<RASPBERRY_PI_IP>:8000/stream"
CONF_THRESH      = 0.20             
NMS_THRESH       = 0.45             
INPUT_SIZE       = 640
USE_DOUBLE_CROP  = True             
MAX_GONE         = 50               
MAX_DIST         = 250              
STABLE_SECS      = 4.0              

# ============================================================
# Centroid Tracker
# ============================================================
class CentroidTracker:
    def __init__(self):
        self.next_id   = 1
        self.objects   = {}   # id -> (cx,cy)
        self.bboxes    = {}   # id -> [x1,y1,x2,y2]
        self.gone      = {}   
        self.last_seen = {}   

    def compute_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0
            
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / max(union_area, 1e-6)

    def update(self, bboxes):
        centroids = [((b[0]+b[2])//2, (b[1]+b[3])//2) for b in bboxes]

        if not bboxes:
            for oid in list(self.gone):
                self.gone[oid] += 1
                if self.gone[oid] > MAX_GONE:
                    self._del(oid)
            return

        if not self.objects:
            for i, b in enumerate(bboxes):
                self._reg(b, centroids[i])
            return

        ids   = list(self.objects)
        num_objects = len(ids)
        num_detections = len(bboxes)

        # 1. First Pass: Match existing active objects using IoU (high overlap)
        iou_matrix = np.zeros((num_objects, num_detections), dtype="float32")
        for i, oid in enumerate(ids):
            old_box = self.bboxes[oid]
            for j, new_box in enumerate(bboxes):
                iou_matrix[i, j] = self.compute_iou(old_box, new_box)

        used_objs = set()
        used_dets = set()
        
        matches = []
        for i in range(num_objects):
            for j in range(num_detections):
                if iou_matrix[i, j] >= 0.25:  # IoU matching threshold
                    matches.append((iou_matrix[i, j], i, j))
        matches.sort(key=lambda x: x[0], reverse=True)

        for iou, i, j in matches:
            if i in used_objs or j in used_dets:
                continue
            oid = ids[i]
            self.objects[oid]   = centroids[j]
            self.bboxes[oid]    = bboxes[j]
            self.gone[oid]      = 0
            self.last_seen[oid] = time.time()
            used_objs.add(i)
            used_dets.add(j)

        # 2. Second Pass: Fallback to centroid distance for remaining unmatched objects
        unmatched_objs = [i for i in range(num_objects) if i not in used_objs]
        unmatched_dets = [j for j in range(num_detections) if j not in used_dets]

        if unmatched_objs and unmatched_dets:
            cents_objs = [self.objects[ids[i]] for i in unmatched_objs]
            cents_dets = [centroids[j] for j in unmatched_dets]
            D = np.array([[np.hypot(oc[0]-nc[0], oc[1]-nc[1])
                           for nc in cents_dets] for oc in cents_objs], dtype="float32")

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            for r, c in zip(rows, cols):
                i = unmatched_objs[r]
                j = unmatched_dets[c]
                if i in used_objs or j in used_dets or D[r, c] > MAX_DIST:
                    continue
                oid = ids[i]
                self.objects[oid]   = centroids[j]
                self.bboxes[oid]    = bboxes[j]
                self.gone[oid]      = 0
                self.last_seen[oid] = time.time()
                used_objs.add(i)
                used_dets.add(j)

        # 3. Mark unmatched objects as missing/gone
        for i in range(num_objects):
            if i not in used_objs:
                oid = ids[i]
                self.gone[oid] += 1
                if self.gone[oid] > MAX_GONE:
                    self._del(oid)

        # 4. Register new detections (with duplicate suppression)
        for j in range(num_detections):
            if j not in used_dets:
                # Suppress double boxes: if new box overlaps heavily with an already-active tracked person
                is_duplicate = False
                for oid in self.objects:
                    if self.gone[oid] == 0:  # Active object in current frame
                        if self.compute_iou(self.bboxes[oid], bboxes[j]) > 0.55:
                            is_duplicate = True
                            break
                if not is_duplicate:
                    self._reg(bboxes[j], centroids[j])

        # 5. Clean up duplicate active trackers (if two active IDs overlap by IoU > 0.55)
        active_ids = [oid for oid in self.objects if self.gone[oid] == 0]
        to_delete = set()
        for i, oid1 in enumerate(active_ids):
            for oid2 in active_ids[i+1:]:
                if oid1 in to_delete or oid2 in to_delete:
                    continue
                if self.compute_iou(self.bboxes[oid1], self.bboxes[oid2]) > 0.55:
                    # Mark the newer tracker (larger ID) for deletion
                    newer_id = max(oid1, oid2)
                    to_delete.add(newer_id)
                    
        for oid in to_delete:
            self._del(oid)

    def _reg(self, bbox, centroid):
        self.objects[self.next_id]   = centroid
        self.bboxes[self.next_id]    = bbox
        self.gone[self.next_id]      = 0
        self.last_seen[self.next_id] = time.time()
        self.next_id += 1

    def _del(self, oid):
        for d in (self.objects, self.bboxes, self.gone, self.last_seen):
            d.pop(oid, None)

    def visible_boxes(self):
        return {k: v for k, v in self.bboxes.items() if self.gone.get(k, 1) == 0}

    def stable_count(self):
        return len(self.visible_boxes())

# ============================================================
# ONNX helpers
# ============================================================
def letterbox(img, size=INPUT_SIZE):
    h, w    = img.shape[:2]
    scale   = size / max(h, w)
    nh, nw  = int(h * scale), int(w * scale)
    canvas  = np.full((size, size, 3), 114, dtype=np.uint8)
    pl = (size - nw) // 2
    pt = (size - nh) // 2
    canvas[pt:pt+nh, pl:pl+nw] = cv2.resize(img, (nw, nh))
    return canvas, scale, pl, pt

def _detect_single(sess, input_name, frame):
    """Run YOLO on one frame, return (boxes, scores) before NMS."""
    lb, scale, pl, pt = letterbox(frame)
    blob = lb[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    out  = sess.run(None, {input_name: blob})[0][0]  # (84,8400)

    scores = out[4]
    mask   = scores >= CONF_THRESH
    if not mask.any():
        return [], []

    cx, cy, bw, bh = out[0][mask], out[1][mask], out[2][mask], out[3][mask]
    s  = scores[mask]
    oh, ow = frame.shape[:2]

    x1 = np.clip(((cx - bw/2) - pl) / scale, 0, ow).astype(int)
    y1 = np.clip(((cy - bh/2) - pt) / scale, 0, oh).astype(int)
    x2 = np.clip(((cx + bw/2) - pl) / scale, 0, ow).astype(int)
    y2 = np.clip(((cy + bh/2) - pt) / scale, 0, oh).astype(int)

    return list(zip(x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist())), s.tolist()


def detect(sess, input_name, frame):
    """
    Multi-crop detection:
    Pass 1 — full frame (catches everyone at normal scale)
    Pass 2 — (Optional) zoomed into the main seating area (right 75% of frame)
              People appear ~1.5x larger → catches occluded/rear-facing people
    Final  — global NMS to remove duplicates across passes
    """
    oh, ow = frame.shape[:2]
    all_boxes, all_scores = [], []

    # Pass 1: full frame
    b1, s1 = _detect_single(sess, input_name, frame)
    all_boxes.extend(b1)
    all_scores.extend(s1)

    # Pass 2: zoom into right 3/4 of frame (where people sit)
    if USE_DOUBLE_CROP:
        cx0 = ow // 4          # start at 25% from left
        crop = frame[0:oh, cx0:ow]
        b2, s2 = _detect_single(sess, input_name, crop)
        # Translate crop-relative coordinates back to full-frame coordinates
        for (x1, y1, x2, y2), sc in zip(b2, s2):
            all_boxes.append((x1 + cx0, y1, x2 + cx0, y2))
            all_scores.append(sc)

    if not all_boxes:
        return []

    # Global NMS across both passes
    boxes_f = [[float(x) for x in b] for b in all_boxes]
    idxs = cv2.dnn.NMSBoxes(boxes_f, all_scores, CONF_THRESH, NMS_THRESH)
    if not len(idxs):
        return []
    return [list(all_boxes[i]) for i in idxs.flatten()]

# ============================================================
# Shared state — written by threads, read by display
# ============================================================
# Separate locks so camera never waits for AI and vice versa
_frame_lock   = threading.Lock()
_results_lock = threading.Lock()
_latest_frame  = None
_latest_boxes  = {}
_latest_count  = 0

# Performance & Thread Sync Metrics
_frame_id = 0               # Incremented whenever camera thread gets a new frame
_processed_frame_id = 0     # Last processed frame_id by AI
_camera_fps = 0.0           # Actual FPS of the camera stream
_inference_fps = 0.0        # Actual FPS of the AI inference

# ============================================================
# Thread 1 — Camera: grab at full camera speed, expose latest raw frame
# ============================================================
def camera_thread():
    global _latest_frame, _frame_id, _camera_fps

    def _open():
        if isinstance(CAMERA_URL, str) and CAMERA_URL.startswith("rtsp://"):
            print("[Camera] Connecting to RTSP stream (TCP)...")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|timeout;5000000"
            cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15_000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10_000)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            print(f"[Camera] Opening camera source: {CAMERA_URL}...")
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            cap = cv2.VideoCapture(CAMERA_URL)
        return cap

    cap = _open()
    consecutive_failures = 0
    fps_timer = time.time()
    frames_counted = 0

    while True:
        ret = cap.grab()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= 100:  # Allow 10 seconds before reconnecting
                print(f"[Camera] Stream lost after {consecutive_failures} failures — reconnecting...")
                cap.release()
                time.sleep(5)
                cap = _open()
                consecutive_failures = 0
            time.sleep(0.1)   # brief pause before retry
            continue
        

        consecutive_failures = 0
        ret, frame = cap.retrieve()
        if ret and frame is not None:
            # Downscale immediately to save CPU and memory bandwidth on Pi 5
            oh, ow = frame.shape[:2]
            if ow > 800:
                scale = 800.0 / ow
                frame = cv2.resize(frame, (800, int(oh * scale)))

            with _frame_lock:
                _latest_frame = frame
                _frame_id += 1

            # Measure camera FPS
            frames_counted += 1
            now = time.time()
            if now - fps_timer >= 2.0:
                _camera_fps = frames_counted / (now - fps_timer)
                frames_counted = 0
                fps_timer = now
        else:
            time.sleep(0.01)
            
    cap.release()


# ============================================================
# Thread 2 — AI: inference as fast as CPU allows
# ============================================================
def inference_thread():
    global _latest_boxes, _latest_count, _processed_frame_id, _inference_fps
    from collections import deque
    print("[AI] Loading ONNX model...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2  # Limit ONNX CPU threads to 2 (leaves cores free for camera/GUI on Pi 5)
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    inp_name   = sess.get_inputs()[0].name
    tracker    = CentroidTracker()

    # Sliding window: last 5 AI frames (reduces counting lag from 5s to 1s)
    count_history = deque(maxlen=5)

    # Target AI FPS to prevent CPU overheating and thermal throttling
    MAX_AI_FPS = 3.0
    MIN_INFERENCE_INTERVAL = 1.0 / MAX_AI_FPS
    last_inference_time = 0.0

    last_processed_id = -1
    fps_timer = time.time()
    frames_processed = 0

    while True:
        now = time.time()
        # Throttle AI loop if running faster than target FPS
        elapsed = now - last_inference_time
        if elapsed < MIN_INFERENCE_INTERVAL:
            time.sleep(MIN_INFERENCE_INTERVAL - elapsed)
            continue

        with _frame_lock:
            frame = _latest_frame
            curr_frame_id = _frame_id

        # Skip inference if frame is None or if we already processed this frame
        if frame is None or curr_frame_id == last_processed_id:
            time.sleep(0.01)
            continue

        last_inference_time = time.time()
        bboxes = detect(sess, inp_name, frame)
        tracker.update(bboxes)

        raw_count = tracker.stable_count()
        count_history.append(raw_count)

        # Mode of recent history = most stable count
        stable = max(set(count_history), key=count_history.count)

        with _results_lock:
            _latest_boxes = tracker.visible_boxes()
            _latest_count = stable
            _processed_frame_id = curr_frame_id

        last_processed_id = curr_frame_id

        # Measure inference FPS
        frames_processed += 1
        now = time.time()
        if now - fps_timer >= 2.0:
            _inference_fps = frames_processed / (now - fps_timer)
            frames_processed = 0
            fps_timer = now


# ============================================================
# Main — Display: draw only when new frames or AI updates arrive
# ============================================================
if __name__ == "__main__":
    threading.Thread(target=camera_thread,   daemon=True).start()
    threading.Thread(target=inference_thread, daemon=True).start()

    print("Press Q to quit.")
    fps_timer = time.time()
    fps_display = 0.0

    last_rendered_frame_id = -1
    last_rendered_processed_id = -1
    last_render_time = time.time()

    while True:
        # Check if there is anything new to display
        with _frame_lock:
            curr_frame_id = _frame_id
        with _results_lock:
            curr_processed_id = _processed_frame_id

        # Throttle loop: only redraw if camera frame or AI results updated
        if curr_frame_id == last_rendered_frame_id and curr_processed_id == last_rendered_processed_id:
            time.sleep(0.01)
            continue

        # Cap display rendering at max 15 FPS to prevent thread/GIL starvation
        now = time.time()
        if now - last_render_time < 0.066:
            time.sleep(0.01)
            continue
        last_render_time = now

        # Grab latest raw frame + AI results atomically
        with _frame_lock:
            frame = _latest_frame
        with _results_lock:
            boxes = dict(_latest_boxes)
            count = _latest_count
            inf_fps = _inference_fps
            cam_fps = _camera_fps

        if frame is None:
            time.sleep(0.01)
            continue

        # Scale down display if resolution is very high (saves CPU copying & window render lag)
        oh, ow = frame.shape[:2]
        if ow > 1024:
            scale = 1024.0 / ow
            display = cv2.resize(frame, (1024, int(oh * scale)))
            # Scale boxes
            boxes_resized = {}
            for tid, (x1, y1, x2, y2) in boxes.items():
                boxes_resized[tid] = (int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale))
            boxes = boxes_resized
        else:
            display = frame.copy()

        # Draw bounding boxes
        for tid, (x1, y1, x2, y2) in boxes.items():
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{tid}"
            cv2.putText(display, label,
                        (max(x1, 0), max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Display FPS metrics (Display, Camera, and Inference)
        now = time.time()
        fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - fps_timer, 1e-6))
        fps_timer = now

        cv2.putText(display, f"Cam FPS: {cam_fps:.1f}",       (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 60, 0), 2)
        cv2.putText(display, f"AI FPS:  {inf_fps:.1f}",       (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 2)
        cv2.putText(display, f"UI FPS:  {fps_display:.1f}",    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.putText(display, f"Persons: {count}",              (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 200), 2)

        cv2.imshow("Human Surveillance", display)
        
        last_rendered_frame_id = curr_frame_id
        last_rendered_processed_id = curr_processed_id

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
