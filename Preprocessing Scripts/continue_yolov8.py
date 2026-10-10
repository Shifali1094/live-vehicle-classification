# Continue training from the previous model
model = YOLO(
    "runs/detect/yolov8_vehicle_detection-3/weights/last.pt"
)

results = model.train(
    data=os.path.abspath(class_names_path),
    imgsz=416,
    batch=8,
    epochs=5,
    patience=3,
    workers=2,
    seed=42,
    device="mps",
    name="yolov8_vehicle_detection_10_epoches_aggregate"
)