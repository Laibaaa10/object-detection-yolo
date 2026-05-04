import cv2
import time
import random
from ultralytics import YOLO
from tracker import LineCounter
from speed_estimator import SpeedEstimator
from heatmap import Heatmap
from database import DetectionDB


class Detector:
    def __init__(
        self,
        model_path="yolov8n.pt",
        conf=0.5,
        iou=0.45,
        classes=None,
        pixel_per_meter=8.0,
        enable_heatmap=True
        enable_db=True    
    ):
        self.model           = YOLO(model_path)
        self.conf            = conf
        self.iou             = iou
        self.classes         = classes
        self.colors          = {}
        self.speed_estimator = SpeedEstimator(
            pixel_per_meter=pixel_per_meter,
            fps=30
        )
        self.heatmap = Heatmap() if enable_heatmap else None
        self.db      = DetectionDB() if enable_db else None

    # ── helpers ───────────────────────────────────────────────────
    def _get_color(self, track_id):
        if track_id not in self.colors:
            random.seed(int(track_id))
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
            color           = self._get_color(track_id)

            # Speed estimation
            speed     = self.speed_estimator.update(track_id, box)
            spd_color = self.speed_estimator.get_color(speed)

            # Trail dot at bottom center
            cx, cy = (x1 + x2) // 2, y2
            cv2.circle(frame, (cx, cy), 4, color, -1)

            # Main bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Corner accents colored by speed
            clen = 16
            for px, py, dx, dy in [
                (x1, y1,  1,  1),
                (x2, y1, -1,  1),
                (x1, y2,  1, -1),
                (x2, y2, -1, -1)
            ]:
                cv2.line(frame, (px, py), (px + dx * clen, py), spd_color, 2)
                cv2.line(frame, (px, py), (px, py + dy * clen), spd_color, 2)

            # Label: class + ID + speed
            label = f"{class_name} [{track_id}]  {speed:.1f} km/h"
            fs, th = 0.55, 1
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            cv2.rectangle(frame,
                (x1, y1 - lh - 10),
                (x1 + lw + 8, y1),
                (15, 15, 15), -1)
            cv2.rectangle(frame,
                (x1, y1 - lh - 10),
                (x1 + lw + 8, y1),
                spd_color, 1)
            cv2.putText(frame, label,
                (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs, spd_color, th)

            # Speed bar below bounding box
            bar_w  = x2 - x1
            filled = int(bar_w * min(speed, 120) / 120)
            cv2.rectangle(frame,
                (x1, y2 + 3), (x2, y2 + 7),
                (40, 40, 40), -1)
            if filled > 0:
                cv2.rectangle(frame,
                    (x1, y2 + 3), (x1 + filled, y2 + 7),
                    spd_color, -1)

            if counter:
                counter.update(track_id, box)

        return frame

    def _draw_fps(self, frame, fps):
        label = f"FPS: {fps:.1f}"
        (w, h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = frame.shape[1] - w - 12
        cv2.rectangle(frame,
            (x - 4, 8), (x + w + 4, h + 16),
            (0, 0, 0), -1)
        cv2.putText(frame, label, (x, h + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
        return frame

    def _draw_info(self, frame, count):
        # Object count
        cv2.rectangle(frame, (8, 8), (240, 36), (0, 0, 0), -1)
        cv2.putText(frame, f"Tracked: {count}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 220, 255), 2)

        # Active filter label
        if self.classes:
            names = [self.model.names[c] for c in self.classes]
            filter_label = "Filter: " + ", ".join(names)
        else:
            filter_label = "Filter: ALL"

        cv2.rectangle(frame, (8, 42), (400, 68), (0, 0, 0), -1)
        cv2.putText(frame, filter_label,
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (180, 255, 180), 2)
        return frame

    def _draw_speed_legend(self, frame):
        h = frame.shape[0]
        items = [
            ((50, 200, 50),  "< 20 km/h  slow"),
            ((0, 200, 255),  "20-60 km/h  medium"),
            ((0, 60, 220),   "> 60 km/h  fast"),
        ]
        for i, (color, text) in enumerate(items):
            y = h - 20 - i * 22
            cv2.circle(frame, (16, y - 4), 6, color, -1)
            cv2.putText(frame, text,
                (28, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1)
        return frame

    # ── webcam loop ───────────────────────────────────────────────
def run_webcam(self, camera_index=0, save_output=False,
                   enable_counter=True):
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("Cannot open webcam")
            return

        counter = None
        if enable_counter:
            counter = LineCounter(
                start_point=(50,  430),
                end_point  =(1230, 430)
            )

        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                "output/tracked_output.mp4", fourcc, 20, (1280,720))

        # ── Start DB session ──────────────────────────────────────
        if self.db:
            class_names = [self.model.names[c]
                           for c in self.classes] if self.classes else None
            self.db.start_session(
                source="webcam",
                model="yolov8n",
                classes=class_names
            )

        prev_time   = time.time()
        fps_history = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(
                frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=640,
                classes=self.classes,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False
            )

            # Heatmap
            if self.heatmap:
                boxes_list = []
                if results[0].boxes is not None:
                    boxes_list = results[0].boxes.xyxy.cpu().numpy()
                self.heatmap.update(boxes_list)
                frame = self.heatmap.draw(frame, alpha=0.45)
                frame = self.heatmap.draw_legend(frame)

            # FPS
            curr_time = time.time()
            fps       = 1 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time
            fps_history.append(fps)
            self.speed_estimator.fps = fps

            # ── Log detections to MongoDB ─────────────────────────
            if self.db and results[0].boxes.id is not None:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                ids     = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confs   = results[0].boxes.conf.cpu().numpy()

                for box, tid, cid, conf in zip(
                        boxes, ids, classes, confs):
                    spd  = self.speed_estimator.get_speed(tid)
                    name = self.model.names[cid]
                    self.db.log_detection(
                        class_name=name,
                        confidence=conf,
                        track_id=tid,
                        bbox=box,
                        speed=spd,
                        alert=False
                    )

            # Draw
            count = len(results[0].boxes) if results[0].boxes else 0
            frame = self._draw_tracked_boxes(frame, results, counter)
            frame = self._draw_fps(frame, fps)
            frame = self._draw_info(frame, count)
            frame = self._draw_speed_legend(frame)

            if counter:
                frame = counter.draw(frame)

            cv2.imshow("SENTINEL  |  Q quit  R reset heatmap", frame)

            if writer:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r") and self.heatmap:
                self.heatmap.reset()

        # ── End session ───────────────────────────────────────────
        avg_fps = sum(fps_history) / max(len(fps_history), 1)
        if self.db:
            self.db.end_session(avg_fps=avg_fps)
            self.db.close()

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()