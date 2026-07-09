from ultralytics import YOLO

print("Exporting custom model to ONNX...")
model = YOLO("best.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True)

print("\nDone! Saved as: best.onnx")