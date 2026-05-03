from detector import Detector

PEOPLE_VEHICLES = [0, 2, 3, 5, 7]

def main():
    print("=" * 50)
    print("  SENTINEL — Heatmap Active")
    print("=" * 50)

    detector = Detector(
        model_path="yolov8n.pt",
        conf=0.5,
        iou=0.45,
        classes=PEOPLE_VEHICLES,
        pixel_per_meter=8.0,
        enable_heatmap=True          # ← toggle on/off
    )

    detector.run_webcam(
        camera_index=0,
        save_output=True,
        enable_counter=True
    )

if __name__ == "__main__":
    main()