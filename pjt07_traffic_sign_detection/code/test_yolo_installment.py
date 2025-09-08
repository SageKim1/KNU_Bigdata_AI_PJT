import torch
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


import ultralytics
print(ultralytics.__version__)


from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO("yolov8n.pt").to(device)  # 모델을 GPU로 이동
results = model.predict(source="https://ultralytics.com/images/bus.jpg", device=device, show=True)
