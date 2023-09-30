from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8s.pt")

# Use the model
results = model.train(data="config.yaml", epochs=10)  # train the model