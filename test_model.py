from ultralytics import YOLO
import cv2

# Load the pretrained model
model = YOLO("yolo11n.pt")

# Print the class names of this model
print("Model Class Names:", model.names)

# Run inference on test.jpeg with normal threshold
results = model("test.jpeg", conf=0.35)

# Print all detections
print("--- Detections ---")
for i, box in enumerate(results[0].boxes):
    xyxy = box.xyxy[0].tolist()
    conf = box.conf[0].item()
    cls = box.cls[0].item()
    name = model.names[int(cls)]
    print(f"Detection {i}: Class={cls} ({name}), Conf={conf:.4f}, Box={xyxy}")
