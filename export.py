from ultralytics import YOLO

print("Exporting YOLO11n model to ONNX...")
model = YOLO("yolo11n.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True)

print("\nDone! Saved as: yolo11n.onnx")