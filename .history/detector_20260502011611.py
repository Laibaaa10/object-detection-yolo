import cv2
import time
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path="yolov8n.pt", conf=0.5, iou=0.45):
        self.model  = YOLO(model_path)
        self.conf   = conf
        self.iou    = iou
        self.colors = {}   # unique color per class

    # ── helpers ───────────────────────────────────────────────────
    def _get_color(self, class_id):
        """Return a consistent BGR color for each class."""
        if class_id not in self.colors:
            import random
            random.seed(class_id)
            self.colors[class_id] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255),
            )
        return self.colors[class_id]

    def _draw_boxes(self, frame, results):
        """Draw bounding boxes, labels and confidence on frame."""
        for box in results[0].boxes:
            class_id   = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = self._get_color(class_id)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background + text
            label = f"{class_name} {confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w, y1), color, -1)
            cv2.putText(frame, label,
                        (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
        return frame

    def _draw_fps(self, frame, fps):
        """Draw FPS counter in top-right corner."""
        label = f"FPS: {fps:.1f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = frame.shape[1] - w - 12
        cv2.rectangle(frame, (x - 4, 8), (x + w + 4, h + 16), (0, 0, 0), -1)
        cv2.putText(frame, label,
                    (x, h + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        return frame

    def _draw_count(self, frame, count):
        """Draw total object count in top-left corner."""
        label = f"Objects: {count}"
        cv2.rectangle(frame, (8, 8), (180, 36), (0, 0, 0), -1)
        cv2.putText(frame, label,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 220, 255), 2)
        return frame

    # ── main webcam loop ──────────────────────────────────────────
    def run_webcam(self, camera_index=0, save_output=False):
        cap = cv2.VideoCapture(camera_index)

        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("❌ Cannot open webcam. Try camera_index=1")
            return

        # Optional: save output video
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                "output/webcam_output.mp4", fourcc, 20,
                (1280, 720)
            )

        print("✅ Webcam started — press Q to quit")

        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break

            # ── Run YOLO inference ────────────────────────────────
            results = self.model(
                frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=640,
                verbose=False
            )

            # ── Calculate FPS ─────────────────────────────────────
            curr_time = time.time()
            fps       = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # ── Draw everything on frame ──────────────────────────
            count = len(results[0].boxes)
            frame = self._draw_boxes(frame, results)
            frame = self._draw_fps(frame, fps)
            frame = self._draw_count(frame, count)

            # ── Show frame ────────────────────────────────────────
            cv2.imshow("YOLOv8 Real-Time Detection  |  Q to quit", frame)

            # ── Save frame if recording ───────────────────────────
            if writer:
                writer.write(frame)

            # ── Quit on Q key ─────────────────────────────────────
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("👋 Quitting...")
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()