from detector import Detector

ALL_CLASSES     = None
PEOPLE_ONLY     = [0]
VEHICLES        = [2, 3, 5, 7]
PEOPLE_VEHICLES = [0, 2, 3, 5, 7]

MODE       = "webcam"
IMAGE_PATH = "test_image.jpg"


def main():
    print("=" * 50)
    print("  YOLOv8 Tracking + Line Counter  — Day 2")
    print("=" * 50)

    detector = Detector(
        model_path="yolov8n.pt",
        conf=0.5,
        iou=0.45,
        classes=PEOPLE_VEHICLES
    )

    if MODE == "webcam":
        print("\n▶ Starting tracked webcam detection...")
        detector.run_webcam(
            camera_index=0,
            save_output=True,       # saves tracked_output.mp4
            enable_counter=True     # enables line crossing counter
        )

    elif MODE == "image":
        print(f"\n▶ Running detection on: {IMAGE_PATH}")
        detector.detect_image(image_path=IMAGE_PATH, save=True)


if __name__ == "__main__":
    main()