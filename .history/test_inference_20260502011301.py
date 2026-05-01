import cv2
from ultralytics import YOLO

# ── 1. Load pretrained YOLOv8 model ──────────────────────────────
model = YOLO("yolov8n.pt")   # downloads automatically if not present
print("✅ Model loaded!")
print(f"   Model type : {type(model)}")
print(f"   Classes    : {model.names}")  # shows all 80 COCO class names

# ── 2. Run inference on the image ─────────────────────────────────
image_path = "test_image.jpg"
results = model(
    source=image_path,
    conf=0.5,       # confidence threshold (0.0 - 1.0)
    iou=0.45,       # IoU threshold for NMS
    imgsz=640,      # image size for inference
    verbose=True    # print detection summary
)

# ── 3. Extract and print detection details ────────────────────────
print("\n── Detection Results ──────────────────────────")
for result in results:
    boxes = result.boxes
    print(f"Total objects detected: {len(boxes)}")
    print()

    for box in boxes:
        class_id   = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        print(f"  Object : {class_name}")
        print(f"  Conf   : {confidence:.2f}")
        print(f"  Box    : ({x1}, {y1}) → ({x2}, {y2})")
        print()

# ── 4. Draw boxes and display the result ─────────────────────────
annotated_frame = results[0].plot()   # draws boxes + labels automatically

# Save annotated image to output folder
cv2.imwrite("output/result.jpg", annotated_frame)
print("✅ Saved to output/result.jpg")

# Show the image in a window
cv2.imshow("YOLOv8 Detection", annotated_frame)
cv2.waitKey(0)          # press any key to close
cv2.destroyAllWindows()