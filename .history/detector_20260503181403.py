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

    def _draw_tracked_boxes(self, frame, results, counter=None):
        if results[0].boxes.id is None:
            return frame

        boxes   = results[0].boxes.xyxy.cpu().numpy()
        ids     = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs   = results[0].boxes.conf.cpu().numpy()

        for box, track_id, class_id, conf in zip(
                boxes, ids, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            class_name      = self.model.names[class_id]

            # ── Speed estimation ──────────────────────────────────
            speed     = self.speed_estimator.update(track_id, box)
            spd_color = self.speed_estimator.get_color(speed)
            color     = self._get_color(track_id)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Corner accents in speed color
            clen = 16
            for px, py, dx, dy in [
                (x1,y1,1,1),(x2,y1,-1,1),
                (x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame,(px,py),(px+dx*clen,py),spd_color,2)
                cv2.line(frame,(px,py),(px,py+dy*clen),spd_color,2)

            # Label with speed
            label = f"{class_name} [{track_id}]  {speed:.1f} km/h"
            (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame,
                (x1, y1-h-10), (x1+w+8, y1),
                (15, 15, 15), -1)
            cv2.rectangle(frame,
                (x1, y1-h-10), (x1+w+8, y1),
                spd_color, 1)
            cv2.putText(frame, label,
                (x1+4, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, spd_color, 1)

            # Speed bar below the box
            bar_w = x2 - x1
            max_spd = 100
            filled  = int(bar_w * min(speed, max_spd) / max_spd)
            cv2.rectangle(frame,
                (x1, y2+2), (x2, y2+6),
                (40, 40, 40), -1)
            cv2.rectangle(frame,
                (x1, y2+2), (x1+filled, y2+6),
                spd_color, -1)

            if counter:
                counter.update(track_id, box)

        return frame