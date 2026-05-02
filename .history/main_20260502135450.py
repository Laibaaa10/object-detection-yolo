from detector import Detector

# ─────────────────────────────────────────────────────────────────
#  CLASS FILTER PRESETS
#  Change the 'classes' parameter in Detector() to switch filters
# ─────────────────────────────────────────────────────────────────

ALL_CLASSES     = None           # detect all 80 COCO classes
PEOPLE_ONLY     = [0]            # person
VEHICLES        = [2, 3, 5, 7]  # car, motorcycle, bus, truck
PEOPLE_VEHICLES = [0, 2, 3, 5, 7]  # person + all vehicles
ANIMALS         = [15, 16, 17]  # cat, dog, horse
ELECTRONICS     = [63, 64, 65]  # laptop, mouse, keyboard


# ─────────────────────────────────────────────────────────────────
#  MODE SELECTOR
#  Set mode = "webcam" or "image"
# ─────────────────────────────────────────────────────────────────

MODE       = "webcam"          # "webcam" or "image"
IMAGE_PATH = "test_image.jpg"  # used only when MODE = "image"


def main():
    print("=" * 50)
    print("  YOLOv8 Real-Time Object Detection System")
    print("=" * 50)

    # ── Initialize detector ───────────────────────────────────────
    detector = Detector(
        model_path="yolov8n.pt",   # yolov8n / yolov8s / yolov8m
        conf=0.5,                  # confidence threshold (0.0-1.0)
        iou=0.45,                  # IoU threshold for NMS
        classes=PEOPLE_VEHICLES    # change filter preset here
    )

    # ── Run selected mode ─────────────────────────────────────────
    if MODE == "webcam":
        print("\n▶ Starting webcam detection...")
        print("  Press Q to quit\n")
        detector.run_webcam(
            camera_index=0,        # try 1 if webcam not found
            save_output=False      # True = saves output/webcam_output.mp4
        )

    elif MODE == "image":
        print(f"\n▶ Running detection on: {IMAGE_PATH}\n")
        detector.detect_image(
            image_path=IMAGE_PATH,
            save=True              # True = saves to output/filtered_<name>.jpg
        )

    else:
        print(f"❌ Unknown mode '{MODE}'. Use 'webcam' or 'image'.")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()