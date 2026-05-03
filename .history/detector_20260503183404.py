import cv2
import time
import random
from ultralytics import YOLO
from tracker import LineCounter
from heatmap import Heatmap    

class Detector:
    def __init__(
        self,
        model_path="yolov8n.pt",
        conf=0.5,
        iou=0.45,
        classes=None
        enable_heatmap=True
    ):
        self.model   = YOLO(model_path)
        self.conf    = conf
        self.iou     = iou
        self.classes = classes
        self.colors  = {}
        self.heatmap         = Heatmap() if enable_heatmap else None

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
            color      = self._get_color(track_id)
            class_name = self.model.names[class_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cx, cy = (x1 + x2) // 2, y2
            cv2.circle(frame, (cx, cy), 4, color, -1)

            label = f"{class_name} ID:{track_id} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame,
                          (x1, y1 - h - 8),
                          (x1 + w, y1),
                          color, -1)
            cv2.putText(frame, label,
                        (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

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
        cv2.rectangle(frame, (8, 8), (240, 36), (0, 0, 0), -1)
        cv2.putText(frame, f"Tracked: {count}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 220, 255), 2)

        if self.classes:
            names = [self.model.names[c] for c in self.classes]
            filter_label = "Filter: " + ", ".join(names)
        else:
            filter_label = "Filter: ALL"

        cv2.rectangle(frame, (8, 42), (350, 68), (0, 0, 0), -1)
        cv2.putText(frame, filter_label,
                    (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (180, 255, 180), 2)
        return frame

    def run_webcam(self, camera_index=0, save_output=False,
                   enable_counter=True):
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("❌ Cannot open webcam. Try camera_index=1")
            return

        counter = None
        if enable_counter:
            counter = LineCounter(
                start_point=(50, 430),
                end_point=(1230, 430)
            )
            print("✅ Line counter enabled at y=430")

        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                "output/tracked_output.mp4",
                fourcc, 20, (1280, 720)
            )
            print("✅ Recording to output/tracked_output.mp4")

        if self.classes:
            names = [self.model.names[c] for c in self.classes]
            print(f"✅ Tracking: {', '.join(names)}")
        else:
            print("✅ Tracking all classes")

        print("   Press Q to quit\n")

        prev_time = time.time()

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

            curr_time = time.time()
            fps       = 1 / (curr_time - prev_time)
            prev_time = curr_time

            count = len(results[0].boxes) if results[0].boxes else 0
            frame = self._draw_tracked_boxes(frame, results, counter)
            frame = self._draw_fps(frame, fps)
            frame = self._draw_info(frame, count)

            if counter:
                frame = counter.draw(frame)

            cv2.imshow("YOLOv8 Tracking  |  Q to quit", frame)

            if writer:
                writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("👋 Quitting...")
                if counter:
                    print(f"   Total crossings: {counter.count}")
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    def detect_image(self, image_path, save=True):
        frame   = cv2.imread(image_path)
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            verbose=False
        )
        count     = len(results[0].boxes)
        annotated = results[0].plot()

        if self.classes:
            names = [self.model.names[c] for c in self.classes]
            print(f"✅ Detected {count} | Filter: {', '.join(names)}")
        else:
            print(f"✅ Detected {count} objects")

        if save:
            out = f"output/filtered_{image_path}"
            cv2.imwrite(out, annotated)
            print(f"   Saved → {out}")

        cv2.imshow("Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()