import cv2
import time
import random
import tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO
from tracker import LineCounter
from collections import deque, defaultdict
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SENTINEL — Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
#  GLASSMORPHISM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Global ── */
*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#e8f4fd 0%,#f0e8ff 40%,#fde8f4 70%,#e8fdf4 100%) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(120,180,255,0.18), transparent 70%);
    top: -100px; left: -80px;
    pointer-events: none; z-index: 0;
}
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(200,140,255,0.15), transparent 70%);
    bottom: -80px; right: -60px;
    pointer-events: none; z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.55) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255,255,255,0.8) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: rgba(100,90,140,0.75) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    color: rgba(91,143,249,0.85) !important;
    text-transform: uppercase !important;
}

/* ── Hide defaults ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Buttons ── */
[data-testid="stButton"] button {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    background: linear-gradient(135deg, #5b8ff9, #a855f7) !important;
    border: none !important;
    color: white !important;
    border-radius: 10px !important;
    width: 100% !important;
    padding: 0.65rem 1rem !important;
    box-shadow: 0 4px 15px rgba(91,143,249,0.3) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] button:hover {
    box-shadow: 0 6px 20px rgba(91,143,249,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.5) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255,255,255,0.75) !important;
    border-radius: 14px !important;
    padding: 14px 16px !important;
}
[data-testid="stMetric"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 2px !important;
    color: rgba(100,90,140,0.5) !important;
    text-transform: uppercase !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 26px !important;
    font-weight: 500 !important;
    background: linear-gradient(135deg, #5b8ff9, #a855f7) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
}

/* ── Selectbox & Multiselect ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: rgba(255,255,255,0.6) !important;
    border: 1px solid rgba(200,190,230,0.5) !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: rgba(80,70,120,0.8) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] [data-baseweb="slider"] div div div div {
    background: linear-gradient(90deg,#5b8ff9,#a855f7) !important;
}
[data-testid="stSlider"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    color: rgba(100,90,140,0.65) !important;
    letter-spacing: 1px !important;
}

/* ── Toggle ── */
[data-testid="stToggle"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: rgba(100,90,140,0.75) !important;
}

/* ── Radio ── */
[data-testid="stRadio"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: rgba(100,90,140,0.75) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.45) !important;
    border: 1.5px dashed rgba(91,143,249,0.35) !important;
    border-radius: 12px !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(180,170,220,0.2) !important;
    margin: 0.75rem 0 !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.5) !important;
    border: 1px solid rgba(91,143,249,0.25) !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    backdrop-filter: blur(8px) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.5) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.75) !important;
    backdrop-filter: blur(10px) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(91,143,249,0.3);
    border-radius: 2px;
}

/* ── Custom components ── */
.sentinel-title {
    font-family: 'DM Mono', monospace;
    font-size: 2.4rem;
    font-weight: 500;
    letter-spacing: 6px;
    background: linear-gradient(135deg, #5b8ff9 0%, #a855f7 50%, #f43f5e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    padding: 1rem 0 0.2rem;
}
.sentinel-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: rgba(100,90,140,0.45);
    text-align: center;
    letter-spacing: 3px;
    margin-bottom: 1.5rem;
}
.glass-card {
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: 16px;
    padding: 1rem 1.25rem;
}
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    color: rgba(91,143,249,0.6);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid rgba(180,170,220,0.2);
    padding-bottom: 0.4rem;
}
.log-container {
    background: rgba(255,255,255,0.4);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.7);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    height: 260px;
    overflow-y: auto;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
}
.log-entry {
    display: flex;
    gap: 10px;
    padding: 4px 0;
    border-bottom: 1px solid rgba(180,170,220,0.12);
    align-items: center;
}
.log-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    flex-shrink: 0;
}
.log-time  { color: rgba(100,90,140,0.35); min-width: 55px; }
.log-class { min-width: 55px; font-weight: 500; }
.log-conf  { color: rgba(100,90,140,0.45); }
.stat-bar-wrap {
    background: rgba(180,170,220,0.15);
    border-radius: 2px;
    height: 3px;
    margin-top: 4px;
}
.stat-bar {
    height: 3px;
    border-radius: 2px;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: rgba(80,140,80,0.8);
}
.live-dot {
    width: 6px; height: 6px;
    background: #4ade80;
    border-radius: 50%;
    animation: blink 1.4s infinite;
}
@keyframes blink {
    0%,100% { opacity:1; box-shadow: 0 0 0 3px rgba(74,222,128,0.2); }
    50%      { opacity:.4; box-shadow: none; }
}
.standby-screen {
    min-height: 380px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,0.35);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.75);
    border-radius: 16px;
}
.metric-alert [data-testid="stMetricValue"] {
    background: linear-gradient(135deg,#f43f5e,#fb923c) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────
if "detection_log"    not in st.session_state:
    st.session_state.detection_log = deque(maxlen=60)
if "class_counts"     not in st.session_state:
    st.session_state.class_counts = defaultdict(int)
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0
if "running"          not in st.session_state:
    st.session_state.running = False
if "fps_history"      not in st.session_state:
    st.session_state.fps_history = deque(maxlen=30)

# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="sentinel-title">SENTINEL</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sentinel-sub">'
    '⬡ real-time object detection system · yolov8 + bytetrack'
    '</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
ALL_CLASSES = {
    "person":0,"bicycle":1,"car":2,"motorcycle":3,
    "bus":5,"truck":7,"cat":15,"dog":16,
    "laptop":63,"chair":56,"bottle":39,"phone":67,
    "bird":14,"backpack":24
}

with st.sidebar:
    st.markdown("### ⬡ SYSTEM CONFIG")
    st.divider()

    st.markdown("**MODEL**")
    model_name = st.selectbox(
        "", ["yolov8n.pt","yolov8s.pt","yolov8m.pt"],
        index=0, label_visibility="collapsed",
        help="n=fastest · s=balanced · m=most accurate"
    )

    st.divider()
    st.markdown("**THRESHOLDS**")
    conf_thresh = st.slider("Confidence", 0.1, 1.0, 0.5, 0.05)
    iou_thresh  = st.slider("IoU / NMS",  0.1, 1.0, 0.45, 0.05)
    img_size    = st.select_slider(
        "Image size",
        options=[320, 416, 512, 640], value=640
    )

    st.divider()
    st.markdown("**CLASS FILTER**")
    selected_classes = st.multiselect(
        "", list(ALL_CLASSES.keys()),
        default=["person","car"],
        label_visibility="collapsed"
    )
    class_ids = [ALL_CLASSES[c] for c in selected_classes] \
        if selected_classes else None

    st.divider()
    st.markdown("**ALERT SYSTEM**")
    alert_enabled  = st.toggle("Enable alert", value=True)
    alert_class    = st.selectbox(
        "Alert trigger class", list(ALL_CLASSES.keys())
    )
    alert_class_id = ALL_CLASSES[alert_class]

    st.divider()
    st.markdown("**FEATURES**")
    enable_counter = st.toggle("Line counter",  value=True)
    show_trails    = st.toggle("Object trails", value=True)
    save_output    = st.toggle("Save video",    value=False)

    st.divider()
    st.markdown("**INPUT SOURCE**")
    source = st.radio(
        "", ["Webcam","Upload Image","Upload Video"],
        label_visibility="collapsed"
    )

    st.divider()
    if st.button("⬡  RESET STATISTICS"):
        st.session_state.detection_log.clear()
        st.session_state.class_counts.clear()
        st.session_state.total_detections = 0
        st.session_state.fps_history.clear()
        st.rerun()

# ─────────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(name):
    return YOLO(name)

model  = load_model(model_name)
colors = {}
trail_history = defaultdict(lambda: deque(maxlen=25))

# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────
# Glassmorphism palette for bounding boxes
BOX_COLORS = [
    (91, 143, 249),   # blue
    (168, 85, 247),   # purple
    (244, 63, 94),    # red/pink
    (34, 197, 94),    # green
    (245, 158, 11),   # amber
    (14, 165, 233),   # sky
    (236, 72, 153),   # pink
    (99, 102, 241),   # indigo
]

def get_color(track_id):
    return BOX_COLORS[int(track_id) % len(BOX_COLORS)]


def draw_frame(frame, results, counter=None):
    alert_triggered = False
    detected_now    = []
    h, w = frame.shape[:2]

    # Subtle vignette overlay
    overlay = np.zeros_like(frame, dtype=np.uint8)
    cv2.circle(overlay, (w//2, h//2),
               int(max(w,h)*0.75), (255,255,255), -1)
    frame = cv2.addWeighted(frame, 0.92, overlay, 0.08, 0)

    if results[0].boxes.id is not None:
        boxes   = results[0].boxes.xyxy.cpu().numpy()
        ids     = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs   = results[0].boxes.conf.cpu().numpy()

        for box, tid, cid, conf in zip(boxes, ids, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            r, g, b         = get_color(tid)
            bgr              = (b, g, r)
            name             = model.names[cid]
            cx, cy           = (x1+x2)//2, (y1+y2)//2

            # Object trail
            if show_trails:
                trail_history[tid].append((cx, y2))
                pts = list(trail_history[tid])
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    tc = (
                        int(b * alpha),
                        int(g * alpha),
                        int(r * alpha)
                    )
                    cv2.line(frame, pts[i-1], pts[i], tc, 2)

            # Alert highlight
            is_alert = alert_enabled and cid == alert_class_id
            if is_alert:
                alert_triggered = True
                cv2.rectangle(frame,
                    (x1-4, y1-4), (x2+4, y2+4),
                    (63, 63, 244), 3)

            # Frosted box fill
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                frost = cv2.addWeighted(
                    roi, 0.85,
                    np.full_like(roi, [b//4, g//4, r//4]), 0.15, 0
                )
                frame[y1:y2, x1:x2] = frost

            # Main box border
            cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 1)

            # Corner accents
            clen = 16
            for px,py,dx,dy in [
                (x1,y1, 1, 1),(x2,y1,-1, 1),
                (x1,y2, 1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame,(px,py),(px+dx*clen,py),bgr,2)
                cv2.line(frame,(px,py),(px,py+dy*clen),bgr,2)

            # Center dot
            cv2.circle(frame, (cx,(y1+y2)//2), 3, bgr, -1)

            # Label background (frosted)
            label  = f"{name}  [{tid}]  {conf:.0%}"
            fs, th = 0.5, 1
            (tw, tht), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            lx1, ly1 = x1, y1 - tht - 12
            lx2, ly2 = x1 + tw + 10, y1
            label_roi = frame[max(0,ly1):ly2, lx1:min(w,lx2)]
            if label_roi.size > 0:
                frost_lbl = cv2.addWeighted(
                    label_roi, 0.6,
                    np.full_like(label_roi, [b//3, g//3, r//3]), 0.4, 0
                )
                frame[max(0,ly1):ly2, lx1:min(w,lx2)] = frost_lbl
            cv2.rectangle(frame, (lx1, max(0,ly1)),
                          (min(w,lx2), ly2), bgr, 1)
            cv2.putText(frame, label,
                        (x1+5, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (255,255,255), th)

            if counter:
                counter.update(tid, box)

            detected_now.append((name, conf, tid))

    return frame, alert_triggered, detected_now


def draw_hud(frame, fps, count, crossings, alert):
    h, w = frame.shape[:2]

    # Top bar frosted
    bar = frame[0:52, 0:w].copy()
    bar = cv2.addWeighted(bar, 0.4,
          np.full_like(bar, [240,240,255]), 0.6, 0)
    frame[0:52, 0:w] = bar
    cv2.line(frame, (0,52), (w,52), (200,190,230), 1)

    cv2.putText(frame, "SENTINEL",
        (14,34), cv2.FONT_HERSHEY_SIMPLEX,
        0.85, (91,143,249), 2)

    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts,
        (w-110,34), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (168,85,247), 1)

    # Bottom bar frosted
    bar2 = frame[h-44:h, 0:w].copy()
    bar2 = cv2.addWeighted(bar2, 0.4,
           np.full_like(bar2, [240,240,255]), 0.6, 0)
    frame[h-44:h, 0:w] = bar2
    cv2.line(frame, (0,h-44), (w,h-44), (200,190,230), 1)

    items = [
        (f"FPS  {fps:.0f}",       (91,143,249)),
        (f"OBJ  {count}",         (168,85,247)),
        (f"CROSS  {crossings}",   (245,158,11)),
        (f"ALERT  {'ON' if alert else 'OFF'}",
                                  (244,63,94) if alert else (160,150,190)),
    ]
    spacing = w // len(items)
    for i,(txt,col) in enumerate(items):
        b,g,r = col[2],col[1],col[0]
        cv2.putText(frame, txt,
            (i*spacing+20, h-16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (b,g,r), 1)

    return frame


def process_frame(frame, counter=None):
    results = model.track(
        frame,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=img_size,
        classes=class_ids,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )
    frame, alert, detected = draw_frame(frame, results, counter)
    count = len(results[0].boxes) if results[0].boxes else 0
    return frame, count, alert, detected


# ─────────────────────────────────────────────────────────────────
#  METRIC CARDS
# ─────────────────────────────────────────────────────────────────
def render_metrics(fps=0.0, objects=0,
                   crossings=0, alert=False, total=0):
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🎯  Objects",   objects,   help="Currently detected")
    c2.metric("⚡  Avg FPS",   f"{fps:.1f}", help="Frames per second")
    c3.metric("🚶  Crossings", crossings, help="Line crossing events")
    c4.metric("📊  Total",     total,     help="All time detections")
    c5.metric("🚨  Alert",
              f"⚠ {alert_class.upper()}" if alert else "CLEAR",
              help=f"Triggers on: {alert_class}")


metric_ph = st.empty()
with metric_ph.container():
    render_metrics()

st.divider()

# ─────────────────────────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────
def update_log():
    entries = list(st.session_state.detection_log)[-20:]
    CLASS_COLORS = {
        "person":"#5b8ff9","car":"#a855f7","bus":"#f59e0b",
        "truck":"#22c55e","cat":"#f43f5e","dog":"#0ea5e9",
        "motorcycle":"#ec4899","bicycle":"#6366f1",
    }
    html = '<div class="log-container">'
    for e in reversed(entries):
        col = CLASS_COLORS.get(e["class"], "#a855f7")
        cls = "log-alert" if e["alert"] else "log-class"
        html += f"""
        <div class="log-entry">
            <div class="log-dot" style="background:{col}"></div>
            <span class="log-time">{e['time']}</span>
            <span class="{cls}" style="color:{col}">{e['class']}</span>
            <span class="log-conf">{e['conf']:.0%}</span>
        </div>"""
    html += "</div>"
    log_ph.markdown(html, unsafe_allow_html=True)


def update_stats():
    counts = dict(sorted(
        st.session_state.class_counts.items(),
        key=lambda x: x[1], reverse=True
    )[:6])
    if not counts:
        stats_ph.markdown(
            '<div class="log-container" style="height:160px;'
            'display:flex;align-items:center;justify-content:center;">'
            '<span style="color:rgba(100,90,140,0.3);'
            'font-family:DM Mono,monospace;font-size:0.7rem;">'
            'no data yet...</span></div>',
            unsafe_allow_html=True
        )
        return

    GRADIENTS = [
        "linear-gradient(90deg,#5b8ff9,#818cf8)",
        "linear-gradient(90deg,#a855f7,#c084fc)",
        "linear-gradient(90deg,#f59e0b,#fbbf24)",
        "linear-gradient(90deg,#22c55e,#4ade80)",
        "linear-gradient(90deg,#f43f5e,#fb7185)",
        "linear-gradient(90deg,#0ea5e9,#38bdf8)",
    ]
    mx   = max(counts.values())
    html = '<div class="log-container" style="height:200px;">'
    for i,(cls,cnt) in enumerate(counts.items()):
        pct = cnt/mx*100
        grad = GRADIENTS[i % len(GRADIENTS)]
        html += f"""
        <div class="stat-row" style="margin-bottom:11px;">
          <div style="display:flex;justify-content:space-between;
            font-family:DM Mono,monospace;font-size:0.72rem;margin-bottom:4px;">
            <span style="color:rgba(80,70,120,0.75)">{cls}</span>
            <span style="color:rgba(100,90,140,0.4)">{cnt}</span>
          </div>
          <div class="stat-bar-wrap">
            <div class="stat-bar" style="width:{pct}%;background:{grad}"></div>
          </div>
        </div>"""
    html += "</div>"
    stats_ph.markdown(html, unsafe_allow_html=True)


vid_col, right_col = st.columns([3,1])

with vid_col:
    video_ph = st.empty()

with right_col:
    st.markdown('<div class="section-label">Detection log</div>',
                unsafe_allow_html=True)
    log_ph = st.empty()
    st.markdown('<div class="section-label" style="margin-top:10px">Class stats</div>',
                unsafe_allow_html=True)
    stats_ph = st.empty()

update_log()
update_stats()

# ─────────────────────────────────────────────────────────────────
#  SOURCE: UPLOAD IMAGE
# ─────────────────────────────────────────────────────────────────
if source == "Upload Image":
    up = st.file_uploader("Drop image here",
                           type=["jpg","jpeg","png"])
    if up:
        file_bytes = np.frombuffer(up.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results   = model(frame,
            conf=conf_thresh, iou=iou_thresh,
            classes=class_ids, verbose=False)
        annotated = results[0].plot()
        count     = len(results[0].boxes)
        annotated = draw_hud(annotated, 0, count, 0, False)

        video_ph.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_column_width=True
        )

        for box in results[0].boxes:
            cid  = int(box.cls[0])
            name = model.names[cid]
            conf = float(box.conf[0])
            is_alert = alert_enabled and cid == alert_class_id
            st.session_state.detection_log.append({
                "time":  datetime.now().strftime("%H:%M:%S"),
                "class": name, "conf": conf, "alert": is_alert
            })
            st.session_state.class_counts[name] += 1
            st.session_state.total_detections   += 1

        with metric_ph.container():
            render_metrics(
                objects=count,
                total=st.session_state.total_detections
            )
        update_log()
        update_stats()

        if count > 0:
            st.markdown('<div class="section-label" style="margin-top:1rem">Detection table</div>',
                        unsafe_allow_html=True)
            rows = []
            for box in results[0].boxes:
                cid = int(box.cls[0])
                rows.append({
                    "Class":      model.names[cid],
                    "Confidence": f"{float(box.conf[0]):.1%}",
                    "X1": int(box.xyxy[0][0]),
                    "Y1": int(box.xyxy[0][1]),
                    "X2": int(box.xyxy[0][2]),
                    "Y2": int(box.xyxy[0][3]),
                })
            st.dataframe(rows,
                         use_container_width=True,
                         hide_index=True)

# ─────────────────────────────────────────────────────────────────
#  SOURCE: WEBCAM / VIDEO
# ─────────────────────────────────────────────────────────────────
else:
    btn1, btn2 = st.columns(2)
    with btn1:
        start_btn = st.button("⬡  INITIALIZE SYSTEM")
    with btn2:
        stop_btn  = st.button("⬡  TERMINATE")

    status_ph = st.empty()

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False

    if st.session_state.running:
        status_ph.markdown(
            '<div class="status-pill" style="margin:0.5rem auto;'
            'width:fit-content;display:flex;">'
            '<div class="live-dot"></div>system active</div>',
            unsafe_allow_html=True
        )

        # Open source
        if source == "Upload Video":
            up = st.file_uploader("Drop video here",
                                   type=["mp4","avi","mov"])
            if not up:
                st.session_state.running = False
                st.stop()
            tfile = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4")
            tfile.write(up.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            st.error("❌ Cannot open source.")
            st.session_state.running = False
            st.stop()

        counter = LineCounter(
            start_point=(50,  360),
            end_point  =(1230,360)
        ) if enable_counter else None

        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                "output/sentinel_output.mp4",
                fourcc, 20, (1280,720))

        prev_time   = time.time()
        frame_count = 0

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame, count, alert, detected = process_frame(
                frame, counter)

            curr_time = time.time()
            fps       = 1 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time
            st.session_state.fps_history.append(fps)
            avg_fps   = float(np.mean(st.session_state.fps_history))

            crossings = counter.count if counter else 0
            frame     = draw_hud(frame, fps, count, crossings, alert)

            if counter:
                frame = counter.draw(frame)

            video_ph.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                use_column_width=True
            )

            if writer:
                writer.write(frame)

            if frame_count % 8 == 0:
                for name, conf, _ in detected:
                    st.session_state.detection_log.append({
                        "time":  datetime.now().strftime("%H:%M:%S"),
                        "class": name, "conf": conf,
                        "alert": alert_enabled and name == alert_class
                    })
                    st.session_state.class_counts[name] += 1
                    st.session_state.total_detections   += 1

                with metric_ph.container():
                    render_metrics(
                        fps=avg_fps, objects=count,
                        crossings=crossings, alert=alert,
                        total=st.session_state.total_detections
                    )
                update_log()
                update_stats()

            frame_count += 1

        cap.release()
        if writer:
            writer.release()

        status_ph.markdown(
            '<div style="text-align:center;font-family:DM Mono,monospace;'
            'font-size:0.7rem;color:rgba(100,90,140,0.4);'
            'letter-spacing:2px;margin-top:0.5rem;">system offline</div>',
            unsafe_allow_html=True
        )
    else:
        video_ph.markdown("""
        <div class="standby-screen">
            <div style="font-size:2.5rem;margin-bottom:1rem;opacity:.2">🎯</div>
            <div style="font-family:DM Mono,monospace;font-size:0.75rem;
                color:rgba(100,90,140,0.35);letter-spacing:3px;">
                awaiting initialization
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;font-family:DM Mono,monospace;
    font-size:0.65rem;color:rgba(100,90,140,0.3);letter-spacing:2px;">
    sentinel v1.0 &nbsp;·&nbsp; yolov8 + opencv + bytetrack + streamlit
    &nbsp;·&nbsp; ai course project
</div>
""", unsafe_allow_html=True)