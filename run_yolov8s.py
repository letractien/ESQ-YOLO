from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8s.yaml")  # build a new model from scratch

    # Train the model
    results = model.train(data="bccd.yaml", epochs=150, imgsz=640, batch=16, cache='ram', device=0)

    # Validate the model
    # results = model.val()  # evaluate model performance on the validation set
