import cv2
import time
import random
from ultralytics import YOLO
from tracker import LineCounter
from speed_estimator import SpeedEstimator


class Detector:
    def __init__(
        self,
        model_path="yolov8n.pt",
        conf=0.5,
        iou=0.45,
        classes=None
        pixel_per_meter=8.0   
    ):
        self.model   = YOLO(model_path)
        self.conf    = conf
        self.iou     = iou
        self.classes = classes
        self.colors  = {}
        self.speed_estimator = SpeedEstimator(   # ← ADD THIS
            pixel_per_meter=pixel_per_meter,
            fps=30
        )

    # ── helpers ───────────────────────────────────────────────────
    def _get_color(self, track_id):
        if track_id not in self.colors:
            random.seed(int(track_id))        # ✅ fixed: int() cast
            self.colors[track_id] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255),
            )
        return self.colors[track_id]

    