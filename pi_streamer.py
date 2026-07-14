import os
# Force FFMPEG to use TCP transport and set connection timeouts globally on startup
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|timeout;5000000"

import cv2
import time
import threading
from flask import Flask, Response

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
CAMERA_URL = "rtsp://admin:cctv%40321@192.168.1.72:554/cam/realmonitor?channel=6&subtype=0"
PORT = 8000
STREAM_WIDTH = 800  # Downscale to 800px width immediately to save network bandwidth

# Thread safety lock and global variables
frame_lock = threading.Lock()
latest_frame = None

def camera_thread():
    """Continuously fetches frames from the camera in a background thread."""
    global latest_frame
    
    def open_camera():
        print(f"[Streamer] Connecting to RTSP source: {CAMERA_URL}")
        cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15_000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10_000)
        return cap

    cap = open_camera()
    consecutive_failures = 0

    while True:
        ret = cap.grab()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= 100:  # ~10 seconds of failures
                print(f"[Streamer] Stream lost. Reconnecting in 5 seconds...")
                cap.release()
                time.sleep(5)
                cap = open_camera()
                consecutive_failures = 0
            time.sleep(0.1)
            continue

        consecutive_failures = 0
        ret, frame = cap.retrieve()
        
        if ret and frame is not None:
            # Resize frame immediately to reduce CPU and bandwidth
            h, w = frame.shape[:2]
            if w > STREAM_WIDTH:
                scale = STREAM_WIDTH / w
                frame = cv2.resize(frame, (STREAM_WIDTH, int(h * scale)))
                
            with frame_lock:
                latest_frame = frame
        else:
            time.sleep(0.01)

    cap.release()

def generate_mjpeg_stream():
    """Generates the multipart JPEG response stream."""
    while True:
        with frame_lock:
            if latest_frame is None:
                frame_to_send = None
            else:
                frame_to_send = latest_frame.copy()

        if frame_to_send is None:
            time.sleep(0.03)  # Wait for first frame
            continue

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame_to_send, [int(cv2.imwrite_jpeg_quality), 80])
        if not ret:
            continue            
        frame_bytes = buffer.tobytes()
        
        # Yield multipart frame format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Cap stream generation rate around 30 FPS to prevent server overload
        time.sleep(0.033)

@app.route('/stream')
def stream_endpoint():
    """HTTP Stream Route."""
    return Response(generate_mjpeg_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the camera capture thread as daemon
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()
    
    print(f"\n[Streamer] Starting HTTP server on port {PORT}...")
    print(f"[Streamer] Open browser or connect client to: http://localhost:{PORT}/stream\n")
    
    # Run Flask server
    app.run(host='0.0.0.0', port=PORT, threaded=True, debug=False)
