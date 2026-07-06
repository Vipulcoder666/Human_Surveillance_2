import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import cv2
import time

CAMERA_URL = "rtsp://admin:cctv@321@192.168.1.72:554/cam/realmonitor?channel=6&subtype=0"
OUTPUT_DIR = "cctv_images"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Connecting to CCTV...")
cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)

print("\n=== Dataset Capture Tool ===")
print("Instructions:")
print("1. Press 's' to save the current frame as an image.")
print("2. Press 'q' to quit.")
print(f"Images will be saved to: {os.path.abspath(OUTPUT_DIR)}")
print("============================\n")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Reconnecting...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)
        continue

    # Show the live feed
    cv2.imshow("Dataset Capture (Press 's' to Save, 'q' to Quit)", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        filename = f"{OUTPUT_DIR}/frame_{count:03d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[{count}] Saved: {filename}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Saved {count} images.")
