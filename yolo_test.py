from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Run inference on an image
results = model('https://ultralytics.com/images/bus.jpg')

# Display the image with predictions
results[0].plot()  # Plot the first result with predictions

# Print the detected objects and their confidence scores
results[0].print()  # Print the first result
