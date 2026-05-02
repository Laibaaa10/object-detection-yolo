import cv2
import time
import random
import tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO
from tracker import LineCounter

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLOv8 Detection System",
    page_icon="🎯",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
st.title("🎯 Real-Time Object Detection System")
st.markdown("**YOLOv8 + OpenCV + ByteTrack** — AI Course Project")
st.divider()

# ─────────────────────────────────────────────────────────────────
#  SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Model selection
    model_name = st.selectbox(
        "YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        index=0,
        help="n=fastest, s=balanced, m=most accurate"
    )

    st.divider()

    # Thresholds
    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Lower = more detections, Higher = fewer but more accurate"
    )

    iou_thresh = st.slider(
        "IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Controls overlap tolerance between boxes"
    )

    st.divider()

    # Class filter
    st.subheader("🏷️ Class Filter")
    ALL_CLASSES = {
        "person": 0, "bicycle": 1, "car": 2,
        "motorcycle": 3, "bus": 5, "truck": 7,
        "cat": 15, "dog": 16, "laptop": 63,
        "chair": 56, "bottle": 39, "phone": 67
    }

    selected_classes = st.multiselect(
        "Detect only these classes",
        options=list(ALL_CLASSES.keys()),
        default=["person", "car"],
        help="Leave empty to detect all 80 classes"
    )
    class_ids = [ALL_CLASSES[c] for c in selected_classes] \
        if selected_classes else None

    st.divider()

    # Alert settings
    st.subheader("🚨 Alert Settings")
    alert_enabled = st.toggle("Enable Alert", value=True)
    alert_class   = st.selectbox(
        "Alert on detection of",
        options=list(ALL_CLASSES.keys()),
        index=0
    )
    alert_class_id = ALL_CLASSES[alert_class]

    st.divider()

    # Counter line
    enable_counter = st.toggle("Enable Line Counter", value=True)

    st.divider()

    # Source
    st.subheader("📥 Input Source")
    source = st.radio(
        "Select source",
        ["Webcam", "Upload Image", "Upload Video"]
    )

# ─────────────────────────────────────────────────────────────────
#  LOAD MODEL (cached so it doesn't reload on every interaction)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

model = load_model(model_name)

# ─────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────
colors = {}

def get_color(track_id):
    if track_id not in colors:
        random.seed(int(track_id))
        colors[track_id] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
    return colors[track_id]


def draw_boxes(frame, results, counter=None):
    alert_triggered = False

    if results[0].boxes.id is not None:
        boxes   = results[0].boxes.xyxy.cpu().numpy()
        ids     = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs   = results[0].boxes.conf.cpu().numpy()

        for box, track_id, class_id, conf in zip(
                boxes, ids, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            color      = get_color(track_id)
            class_name = model.names[class_id]

            # Alert check
            if alert_enabled and class_id == alert_class_id:
                alert_triggered = True
                # Red flashing border on alerted object
                cv2.rectangle(frame,
                              (x1-3, y1-3), (x2+3, y2+3),
                              (0, 0, 255), 4)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name} ID:{track_id} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame,
                          (x1, y1 - h - 8), (x1 + w, y1),
                          color, -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

            if counter:
                counter.update(track_id, box)

    return frame, alert_triggered


def process_frame(frame, counter=None):
    results = model.track(
        frame,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=640,
        classes=class_ids,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )
    frame, alert = draw_boxes(frame, results, counter)
    count = len(results[0].boxes) if results[0].boxes else 0
    return frame, count, alert


# ─────────────────────────────────────────────────────────────────
#  MAIN LAYOUT — metric cards
# ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
metric_objects  = col1.empty()
metric_fps      = col2.empty()
metric_crossings = col3.empty()
metric_alert    = col4.empty()

# Default metric display
metric_objects.metric("🎯 Objects", "0")
metric_fps.metric("⚡ FPS", "0")
metric_crossings.metric("🚶 Crossings", "0")
metric_alert.metric("🚨 Alert", "OFF")

st.divider()

# ─────────────────────────────────────────────────────────────────
#  SOURCE: UPLOAD IMAGE
# ─────────────────────────────────────────────────────────────────
if source == "Upload Image":
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = model(
            frame,
            conf=conf_thresh,
            iou=iou_thresh,
            classes=class_ids,
            verbose=False
        )
        annotated = results[0].plot()
        count     = len(results[0].boxes)

        # Show results side by side
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Original")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                     use_column_width=True)
        with c2:
            st.subheader(f"Detected ({count} objects)")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     use_column_width=True)

        metric_objects.metric("🎯 Objects", count)

        # Detection table
        if count > 0:
            st.subheader("📋 Detection Details")
            data = []
            for box in results[0].boxes:
                cid  = int(box.cls[0])
                data.append({
                    "Class":      model.names[cid],
                    "Confidence": f"{float(box.conf[0]):.2f}",
                    "X1": int(box.xyxy[0][0]),
                    "Y1": int(box.xyxy[0][1]),
                    "X2": int(box.xyxy[0][2]),
                    "Y2": int(box.xyxy[0][3]),
                })
            st.dataframe(data, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
#  SOURCE: UPLOAD VIDEO
# ─────────────────────────────────────────────────────────────────
elif source == "Upload Video":
    uploaded = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov"])

    if uploaded:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4")
        tfile.write(uploaded.read())

        cap     = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        counter = LineCounter(
            start_point=(50, 360),
            end_point=(1230, 360)
        ) if enable_counter else None

        stop_btn = st.button("⏹ Stop")

        prev_time = time.time()

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            frame, count, alert = process_frame(frame, counter)

            curr_time = time.time()
            fps       = 1 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time

            if counter:
                frame = counter.draw(frame)

            stframe.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                use_column_width=True
            )

            crossings = counter.count if counter else 0
            metric_objects.metric("🎯 Objects", count)
            metric_fps.metric("⚡ FPS", f"{fps:.1f}")
            metric_crossings.metric("🚶 Crossings", crossings)
            metric_alert.metric(
                "🚨 Alert",
                f"⚠️ {alert_class}!" if alert else "OFF"
            )

        cap.release()
        st.success("✅ Video processing complete!")

# ─────────────────────────────────────────────────────────────────
#  SOURCE: WEBCAM
# ─────────────────────────────────────────────────────────────────
elif source == "Webcam":
    st.info("👆 Click **Start** to begin webcam detection")

    start = st.button("▶ Start Webcam")
    stop  = st.button("⏹ Stop Webcam")

    if start:
        cap     = cv2.VideoCapture(0)
        stframe = st.empty()

        counter = LineCounter(
            start_point=(50, 360),
            end_point=(1230, 360)
        ) if enable_counter else None

        prev_time = time.time()

        while not stop:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Cannot access webcam")
                break

            frame, count, alert = process_frame(frame, counter)

            curr_time = time.time()
            fps       = 1 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time

            if counter:
                frame = counter.draw(frame)

            stframe.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                use_column_width=True
            )

            crossings = counter.count if counter else 0
            metric_objects.metric("🎯 Objects", count)
            metric_fps.metric("⚡ FPS", f"{fps:.1f}")
            metric_crossings.metric("🚶 Crossings", crossings)
            metric_alert.metric(
                "🚨 Alert",
                f"⚠️ {alert_class}!" if alert else "OFF"
            )

        cap.release()

# ─────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "Built with **YOLOv8** · **OpenCV** · **Streamlit** · **ByteTrack**"
)