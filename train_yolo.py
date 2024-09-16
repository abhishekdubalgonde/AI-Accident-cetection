from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # or 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'

# Train the model
model.train(data='data.yaml', epochs=50, imgsz=640, batch=16, device='cpu')  # Use 'cpu' instead of 'cuda'

# Save the trained model
model.save('yolov8_trained.pt')
