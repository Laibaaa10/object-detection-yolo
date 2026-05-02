from detector import Detector

def main():
    detector = Detector()
    detector.run_webcam()

if __name__ == "__main__":
    main()

from detector import Detector

def main():
    detector = Detector(
        model_path="yolov8n.pt",
        conf=0.5,     # adjust: lower = more detections
        iou=0.45
    )

    # Run webcam (set save_output=True to record video)
    detector.run_webcam(camera_index=0, save_output=False)

if __name__ == "__main__":
    main()
    from detector import Detector

# ── Common filter presets ─────────────────────────────────────────
ALL_CLASSES      = None          # detect everything
PEOPLE_ONLY      = [0]           # person
VEHICLES         = [2, 3, 5, 7]  # car, motorcycle, bus, truck
PEOPLE_VEHICLES  = [0, 2, 3, 5, 7]
ANIMALS          = [15, 16, 17]  # cat, dog, horse
ELECTRONICS      = [63, 64, 65]  # laptop, mouse, keyboard

def main():
    detector = Detector(
        model_path="yolov8n.pt",
        conf=0.5,
        iou=0.45,
        classes=PEOPLE_VEHICLES   # ← change this to switch filters
    )

    detector.run_webcam(camera_index=0, save_output=False)

if __name__ == "__main__":
    main()

    # Test on static image with filter
detector.detect_image("test_image.jpg")