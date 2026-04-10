from ultralytics import YOLO

model = YOLO("models/violence_yolo.pt")
print("Classes:", model.names)
