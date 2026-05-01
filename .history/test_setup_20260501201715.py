import cv2
import numpy as np
from ultralytics import YOLO

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)

# Load model
model = YOLO("yolov8n.pt")
print("YOLOv8 loaded successfully!")

# Test on a blank image
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
results = model(dummy, verbose=False)
print("Inference test passed!")
print("\n✅ All good — you're ready to build!")