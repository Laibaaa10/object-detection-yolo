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