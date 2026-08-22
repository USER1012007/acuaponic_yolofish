from ultralytics import YOLO
import sys

def compile_model(weights_path, dataset_yaml):
    model = YOLO(weights_path)
    model.export(
        format="hailo",
        name="hailo8l",
        data=dataset_yaml, 
        imgsz=640
    )

if __name__ == "__main__":
# Uso: python scripts/compile_for_hailo.py output/yolov8n/weights/best.pt configs/dataset.yaml
    weights = sys.argv[1]
    data = sys.argv[2]

compile_model(weights, data)
