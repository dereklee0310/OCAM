from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-obb.yaml")  # build a new model from YAML
model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-obb.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="configs/fold1.yaml", batch=8, epochs=10, imgsz=1200)