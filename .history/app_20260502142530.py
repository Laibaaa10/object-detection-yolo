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
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
#  CUSTOM CSS — Dark Cyberpunk Professional Theme
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #020408 !important;
    color: #c8d8e8 !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 20% 50%, rgba(0,255,180,0.03) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(0,150,255,0.04) 0%, transparent 50%),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 40px,
            rgba(0,255,180,0.015) 40px,
            rgba(0,255,180,0.015) 41px
        ),
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 40px,
            rgba(0,255,180,0.015) 40px,
            rgba(0,255,180,0.015) 41px
        ),
        #020408 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #050c10 !important;
    border-right: 1px solid rgba(0,255,180,0.15) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00ffb4, #00aaff, #00ffb4);
    animation: scanline 3s linear infinite;
}
@keyframes scanline {
    0%   { opacity: 1; }
    50%  { opacity: 0.4; }
    100% { opacity: 1; }
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 2px !important;
}
p, label, div, span {
    font-family: 'Rajdhani', sans-serif !important;
}
code, pre {
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Main title ── */
.sentinel-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: 6px;
    background: linear-gradient(135deg, #00ffb4 0%, #00aaff 50%, #00ffb4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    padding: 1rem 0 0.2rem;
    filter: drop-shadow(0 0 20px rgba(0,255,180,0.3));
}
.sentinel-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: rgba(0,255,180,0.5);
    text-align: center;
    letter-spacing: 4px;
    margin-bottom: 1.5rem;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, #050f15 0%, #081820 100%);
    border: 1px solid rgba(0,255,180,0.2);
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #00ffb4, #00aaff);
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, #00ffb4, transparent);
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: rgba(0,255,180,0.6);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #00ffb4;
    line-height: 1;
    text-shadow: 0 0 20px rgba(0,255,180,0.5);
}
.metric-value.alert-active {
    color: #ff4444;
    text-shadow: 0 0 20px rgba(255,68,68,0.6);
    animation: pulse-alert 0.8s ease-in-out infinite;
}
@keyframes pulse-alert {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
}
.metric-unit {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: rgba(200,216,232,0.4);
    margin-top: 0.3rem;
}

/* ── Detection Log ── */
.log-container {
    background: #020a0e;
    border: 1px solid rgba(0,255,180,0.1);
    border-radius: 4px;
    padding: 1rem;
    height: 280px;
    overflow-y: auto;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
}
.log-entry {
    padding: 3px 0;
    border-bottom: 1px solid rgba(0,255,180,0.05);
    display: flex;
    gap: 12px;
}
.log-time  { color: rgba(0,255,180,0.4); min-width: 85px; }
.log-class { color: #00aaff; min-width: 100px; }
.log-conf  { color: rgba(200,216,232,0.6); }
.log-alert { color: #ff4444; font-weight: bold; }

/* ── Section Headers ── */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    letter-spacing: 4px;
    color: rgba(0,255,180,0.5);
    text-transform: uppercase;
    border-bottom: 1px solid rgba(0,255,180,0.1);
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
}

/* ── Video Feed Border ── */
.video-wrapper {
    border: 1px solid rgba(0,255,180,0.2);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
    background: #000;
}
.video-wrapper::before {
    content: '⬡ LIVE FEED';
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #00ffb4;
    letter-spacing: 3px;
    position: absolute;
    top: 10px; left: 14px;
    z-index: 10;
    background: rgba(0,0,0,0.7);
    padding: 2px 8px;
    border: 1px solid rgba(0,255,180,0.3);
}

/* ── Sliders ── */
[data-testid="stSlider"] .st-emotion-cache-1gv3huu {
    background: rgba(0,255,180,0.15) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div div div div {
    background: #00ffb4 !important;
}

/* ── Buttons ── */
[data-testid="stButton"] button {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 3px !important;
    background: transparent !important;
    border: 1px solid rgba(0,255,180,0.4) !important;
    color: #00ffb4 !important;
    border-radius: 2px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
[data-testid="stButton"] button:hover {
    background: rgba(0,255,180,0.1) !important;
    border-color: #00ffb4 !important;
    box-shadow: 0 0 20px rgba(0,255,180,0.2) !important;
}

/* ── Selectbox + Multiselect ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #050f15 !important;
    border-color: rgba(0,255,180,0.2) !important;
    color: #c8d8e8 !important;
    border-radius: 2px !important;
}

/* ── Radio ── */
[data-testid="stRadio"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
    color: rgba(200,216,232,0.7) !important;
}

/* ── Sidebar labels ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    color: rgba(0,255,180,0.7) !important;
    letter-spacing: 1px !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.8rem !important;
    color: #00ffb4 !important;
    letter-spacing: 3px !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(0,255,180,0.1) !important;
    margin: 1rem 0 !important;
}

/* ── Status indicator ── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00ffb4;
    box-shadow: 0 0 8px #00ffb4;
    animation: blink 1.5s ease-in-out infinite;
    margin-right: 8px;
}
.status-dot.inactive {
    background: rgba(200,216,232,0.2);
    box-shadow: none;
    animation: none;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* ── Class badges ── */
.class-badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 8px;
    border: 1px solid rgba(0,170,255,0.4);
    border-radius: 2px;
    color: #00aaff;
    background: rgba(0,170,255,0.08);
    margin: 2px;
    letter-spacing: 1px;
}

/* ── Progress bars ── */
.conf-bar-wrap {
    background: rgba(0,255,180,0.08);
    border-radius: 2px;
    height: 4px;
    margin-top: 4px;
}
.conf-bar {
    background: linear-gradient(90deg, #00ffb4, #00aaff);
    height: 4px;
    border-radius: 2px;
    transition: width 0.3s ease;
}

/* ── Table ── */
[data-testid="stDataFrame"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #020408; }
::-webkit-scrollbar-thumb {
    background: rgba(0,255,180,0.3);
    border-radius: 2px;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #050f15 !important;
    border: 1px dashed rgba(0,255,180,0.3) !important;
    border-radius: 4px !important;
}

/* ── Info/warning boxes ── */
[data-testid="stAlert"] {
    background: rgba(0,255,180,0.05) !important;
    border: 1px solid rgba(0,255,180,0.2) !important;
    border-radius: 2px !important;
    color: #c8d8e8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────
if "detection_log"   not in st.session_state:
    st.session_state.detection_log = deque(maxlen=50)
if "class_counts"    not in st.session_state:
    st.session_state.class_counts = defaultdict(int)
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0
if "running"         not in st.session_state:
    st.session_state.running = False
if "fps_history"     not in st.session_state:
    st.session_state.fps_history = deque(maxlen=30)

# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="sentinel-title">SENTINEL</div>', unsafe_allow_html=True)
st.markdown('<div class="sentinel-sub">⬡ REAL-TIME OBJECT DETECTION SYSTEM ⬡ YOLOv8 + BYTETRACK</div>', unsafe_allow_html=True)

# Live clock
clock_placeholder = st.empty()
now = datetime.now().strftime("%Y.%m.%d  //  %H:%M:%S")
clock_placeholder.markdown(
    f'<div style="text-align:center;font-family:Share Tech Mono,monospace;'
    f'font-size:0.7rem;color:rgba(0,255,180,0.35);letter-spacing:4px;'
    f'margin-bottom:1.5rem;">{now}</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⬡ SYSTEM CONFIG")
    st.divider()

    st.markdown("**MODEL**")
    model_name = st.selectbox(
        "", ["yolov8n.pt","yolov8s.pt","yolov8m.pt"],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**DETECTION THRESHOLDS**")

    conf_thresh = st.slider("CONFIDENCE", 0.1, 1.0, 0.5, 0.05)
    iou_thresh  = st.slider("IoU NMS",    0.1, 1.0, 0.45, 0.05)
    img_size    = st.select_slider(
        "INFERENCE SIZE",
        options=[320, 416, 512, 640],
        value=640
    )

    st.divider()
    st.markdown("**CLASS FILTER**")

    ALL_CLASSES = {
        "person":0,"bicycle":1,"car":2,"motorcycle":3,
        "bus":5,"truck":7,"cat":15,"dog":16,
        "laptop":63,"chair":56,"bottle":39,"phone":67,
        "bird":14,"backpack":24,"umbrella":25
    }

    selected_classes = st.multiselect(
        "",
        options=list(ALL_CLASSES.keys()),
        default=["person","car"],
        label_visibility="collapsed"
    )
    class_ids = [ALL_CLASSES[c] for c in selected_classes] \
        if selected_classes else None

    st.divider()
    st.markdown("**ALERT SYSTEM**")
    alert_enabled  = st.toggle("ENABLE ALERT", value=True)
    alert_class    = st.selectbox(
        "TRIGGER CLASS", list(ALL_CLASSES.keys()),
        label_visibility="visible"
    )
    alert_class_id = ALL_CLASSES[alert_class]

    st.divider()
    st.markdown("**FEATURES**")
    enable_counter = st.toggle("LINE COUNTER",   value=True)
    show_trails    = st.toggle("OBJECT TRAILS",  value=True)
    save_output    = st.toggle("SAVE VIDEO",     value=False)

    st.divider()
    st.markdown("**INPUT SOURCE**")
    source = st.radio(
        "",
        ["⬡  Webcam", "⬡  Image", "⬡  Video"],
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

def get_color(tid):
    if tid not in colors:
        random.seed(int(tid))
        colors[tid] = (
            random.randint(50,255),
            random.randint(50,255),
            random.randint(50,255),
        )
    return colors[tid]

# Trail history
trail_history = defaultdict(lambda: deque(maxlen=25))

# ─────────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────
def draw_frame(frame, results, counter=None):
    alert_triggered = False
    detected_now    = []

    h, w = frame.shape[:2]

    # Subtle scanline overlay
    overlay = frame.copy()
    for y in range(0, h, 3):
        cv2.line(overlay, (0,y), (w,y), (0,0,0), 1)
    frame = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)

    if results[0].boxes.id is not None:
        boxes   = results[0].boxes.xyxy.cpu().numpy()
        ids     = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs   = results[0].boxes.conf.cpu().numpy()

        for box, tid, cid, conf in zip(boxes, ids, classes, confs):
            x1,y1,x2,y2 = map(int, box)
            color        = get_color(tid)
            name         = model.names[cid]
            cx,cy        = (x1+x2)//2, (y1+y2)//2

            # Trail
            if show_trails:
                trail_history[tid].append((cx, y2))
                pts = list(trail_history[tid])
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    tc    = tuple(int(c * alpha) for c in color)
                    cv2.line(frame, pts[i-1], pts[i], tc, 2)

            # Alert highlight
            is_alert = alert_enabled and cid == alert_class_id
            if is_alert:
                alert_triggered = True
                cv2.rectangle(frame,
                    (x1-5, y1-5), (x2+5, y2+5),
                    (0,0,220), 3)
                # Corners glow
                for cx2,cy2,dx,dy in [
                    (x1,y1,1,1),(x2,y1,-1,1),
                    (x1,y2,1,-1),(x2,y2,-1,-1)]:
                    cv2.line(frame,(cx2,cy2),(cx2+dx*20,cy2),(0,0,255),3)
                    cv2.line(frame,(cx2,cy2),(cx2,cy2+dy*20),(0,0,255),3)

            # Corner-style bounding box (not full rectangle)
            clen = 18
            for px,py,dx,dy in [
                (x1,y1, 1, 1),(x2,y1,-1, 1),
                (x1,y2, 1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame,(px,py),(px+dx*clen,py),color,2)
                cv2.line(frame,(px,py),(px,py+dy*clen),color,2)

            # Center dot
            cv2.circle(frame,(cx,(y1+y2)//2),3,color,-1)

            # Label
            label  = f"{name}  [{tid}]  {conf:.0%}"
            fs     = 0.55
            th     = 2
            (tw,tht),_ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            cv2.rectangle(frame,
                (x1, y1-tht-12),(x1+tw+8, y1),
                (10,10,10), -1)
            cv2.rectangle(frame,
                (x1, y1-tht-12),(x1+tw+8, y1),
                color, 1)
            cv2.putText(frame, label,
                (x1+4, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs, color, th)

            if counter:
                counter.update(tid, box)

            detected_now.append((name, conf, tid))

    return frame, alert_triggered, detected_now


def draw_hud(frame, fps, count, crossings, alert):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0,0), (w,52), (5,10,15), -1)
    cv2.line(frame, (0,52), (w,52), (0,255,180), 1)

    cv2.putText(frame, "SENTINEL",
        (14,34), cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (0,255,180), 2)

    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts,
        (w-120,34), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0,200,140), 1)

    # Bottom bar
    cv2.rectangle(frame, (0,h-52), (w,h), (5,10,15), -1)
    cv2.line(frame, (0,h-52), (w,h-52), (0,255,180), 1)

    metrics = [
        (f"FPS  {fps:.0f}",        (0,255,180)),
        (f"OBJ  {count}",          (0,200,255)),
        (f"CROSS  {crossings}",    (0,170,255)),
        (f"ALERT  {'ON' if alert else 'OFF'}", (0,80,255) if alert else (80,80,80)),
    ]
    spacing = w // len(metrics)
    for i,(text,col) in enumerate(metrics):
        cv2.putText(frame, text,
            (i*spacing+20, h-18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, col, 1)

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
def render_metrics(fps=0, objects=0, crossings=0,
                   alert=False, total=0):
    c1,c2,c3,c4,c5 = st.columns(5)

    cards = [
        (c1, "OBJECTS",    str(objects),       "detected",       False),
        (c2, "AVG FPS",    f"{fps:.1f}",       "frames/sec",     False),
        (c3, "CROSSINGS",  str(crossings),     "line events",    False),
        (c4, "TOTAL",      str(total),         "all time",       False),
        (c5, "ALERT",      "⚠ ACTIVE" if alert else "CLEAR",
                                               alert_class,      alert),
    ]
    for col,(label,val,unit,is_alert) in [(c[0],c[1:]) for c in cards]:
        alert_cls = "alert-active" if is_alert else ""
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {alert_cls}">{val}</div>
            <div class="metric-unit">{unit}</div>
        </div>
        """, unsafe_allow_html=True)


metric_placeholder = st.empty()
with metric_placeholder.container():
    render_metrics()

st.markdown('<div class="section-header">⬡ DETECTION FEED</div>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────
vid_col, log_col = st.columns([3, 1])

with vid_col:
    st.markdown('<div class="video-wrapper">', unsafe_allow_html=True)
    video_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with log_col:
    st.markdown('<div class="section-header">⬡ DETECTION LOG</div>',
                unsafe_allow_html=True)
    log_placeholder = st.empty()

    st.markdown('<div class="section-header">⬡ CLASS STATS</div>',
                unsafe_allow_html=True)
    stats_placeholder = st.empty()


def update_log():
    entries = list(st.session_state.detection_log)[-18:]
    html = '<div class="log-container">'
    for e in reversed(entries):
        cls  = "log-alert" if e["alert"] else "log-class"
        html += f"""
        <div class="log-entry">
            <span class="log-time">{e['time']}</span>
            <span class="{cls}">{e['class']}</span>
            <span class="log-conf">{e['conf']:.0%}</span>
        </div>"""
    html += "</div>"
    log_placeholder.markdown(html, unsafe_allow_html=True)


def update_stats():
    counts = dict(sorted(
        st.session_state.class_counts.items(),
        key=lambda x: x[1], reverse=True
    )[:8])
    if not counts:
        stats_placeholder.markdown(
            '<div class="log-container" style="height:140px;">'
            '<span style="color:rgba(0,255,180,0.3);'
            'font-family:Share Tech Mono,monospace;font-size:0.7rem;">'
            'NO DATA YET...</span></div>',
            unsafe_allow_html=True)
        return
    mx  = max(counts.values())
    html = '<div class="log-container" style="height:200px;">'
    for cls, cnt in counts.items():
        pct = cnt / mx * 100
        html += f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;
                font-family:Share Tech Mono,monospace;font-size:0.7rem;">
                <span style="color:#00aaff;">{cls}</span>
                <span style="color:rgba(200,216,232,0.5);">{cnt}</span>
            </div>
            <div class="conf-bar-wrap">
                <div class="conf-bar" style="width:{pct}%"></div>
            </div>
        </div>"""
    html += "</div>"
    stats_placeholder.markdown(html, unsafe_allow_html=True)


# Initial render
update_log()
update_stats()

# ─────────────────────────────────────────────────────────────────
#  SOURCE: IMAGE
# ─────────────────────────────────────────────────────────────────
if "Image" in source:
    uploaded = st.file_uploader(
        "DROP IMAGE FILE", type=["jpg","jpeg","png"])

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = model(frame,
            conf=conf_thresh, iou=iou_thresh,
            classes=class_ids, verbose=False)

        annotated = results[0].plot()
        count     = len(results[0].boxes)

        # Draw HUD on image
        annotated = draw_hud(annotated, 0, count, 0, False)
        video_placeholder.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_column_width=True
        )

        # Log detections
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

        with metric_placeholder.container():
            render_metrics(fps=0, objects=count,
                           crossings=0,
                           total=st.session_state.total_detections)
        update_log()
        update_stats()

        # Detail table
        st.markdown('<div class="section-header">⬡ DETECTION TABLE</div>',
                    unsafe_allow_html=True)
        rows = []
        for box in results[0].boxes:
            cid = int(box.cls[0])
            rows.append({
                "CLASS":      model.names[cid],
                "CONFIDENCE": f"{float(box.conf[0]):.1%}",
                "X1": int(box.xyxy[0][0]),
                "Y1": int(box.xyxy[0][1]),
                "X2": int(box.xyxy[0][2]),
                "Y2": int(box.xyxy[0][3]),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────
#  SOURCE: VIDEO / WEBCAM
# ─────────────────────────────────────────────────────────────────
else:
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        start_btn = st.button("⬡  INITIALIZE SYSTEM")
    with btn_col2:
        stop_btn  = st.button("⬡  TERMINATE")

    status_ph = st.empty()

    if start_btn:
        st.session_state.running = True

    if stop_btn:
        st.session_state.running = False

    if st.session_state.running:
        status_ph.markdown(
            '<div style="font-family:Share Tech Mono,monospace;'
            'font-size:0.75rem;color:#00ffb4;letter-spacing:2px;">'
            '<span class="status-dot"></span>SYSTEM ACTIVE</div>',
            unsafe_allow_html=True
        )

        # Open source
        if "Video" in source:
            uploaded = st.file_uploader(
                "DROP VIDEO FILE", type=["mp4","avi","mov"])
            if not uploaded:
                st.session_state.running = False
                st.stop()
            tfile = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            st.error("❌ Cannot open source. Check webcam or file.")
            st.session_state.running = False
            st.stop()

        # Counter line
        counter = LineCounter(
            start_point=(50,  360),
            end_point  =(1230,360)
        ) if enable_counter else None

        # Video writer
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

            # FPS
            curr_time = time.time()
            fps       = 1 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time
            st.session_state.fps_history.append(fps)
            avg_fps = np.mean(st.session_state.fps_history)

            # HUD
            crossings = counter.count if counter else 0
            frame = draw_hud(frame, fps, count, crossings, alert)

            # Draw counter line
            if counter:
                frame = counter.draw(frame)

            # Show frame
            video_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                use_column_width=True
            )

            # Save
            if writer:
                writer.write(frame)

            # Log every 10 frames
            if frame_count % 10 == 0:
                for name, conf, tid in detected:
                    is_alert = alert_enabled and \
                        model.names.get(
                            next((int(b.cls[0])
                                  for b in (
                                      results[0].boxes
                                      if hasattr(results[0],'boxes')
                                      else [])
                                  if True), -1),
                            "") == alert_class
                    st.session_state.detection_log.append({
                        "time":  datetime.now().strftime("%H:%M:%S"),
                        "class": name,
                        "conf":  conf,
                        "alert": alert_enabled and name == alert_class
                    })
                    st.session_state.class_counts[name] += 1
                    st.session_state.total_detections += 1

                with metric_placeholder.container():
                    render_metrics(
                        fps=avg_fps,
                        objects=count,
                        crossings=crossings,
                        alert=alert,
                        total=st.session_state.total_detections
                    )
                update_log()
                update_stats()

            frame_count += 1

        cap.release()
        if writer:
            writer.release()

        status_ph.markdown(
            '<div style="font-family:Share Tech Mono,monospace;'
            'font-size:0.75rem;color:rgba(200,216,232,0.4);'
            'letter-spacing:2px;">'
            '<span class="status-dot inactive"></span>'
            'SYSTEM OFFLINE</div>',
            unsafe_allow_html=True
        )
    else:
        video_placeholder.markdown("""
        <div style="
            height:400px;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            border:1px solid rgba(0,255,180,0.1);
            background:radial-gradient(
                ellipse at center,
                rgba(0,255,180,0.03) 0%,
                transparent 70%);
            border-radius:4px;">
            <div style="
                font-family:Orbitron,monospace;
                font-size:3rem;
                color:rgba(0,255,180,0.15);
                letter-spacing:8px;">⬡</div>
            <div style="
                font-family:Share Tech Mono,monospace;
                font-size:0.75rem;
                color:rgba(0,255,180,0.25);
                letter-spacing:4px;
                margin-top:1rem;">
                AWAITING INITIALIZATION</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="
    text-align:center;
    font-family:Share Tech Mono,monospace;
    font-size:0.65rem;
    color:rgba(0,255,180,0.2);
    letter-spacing:3px;">
    SENTINEL v1.0  ⬡  YOLOv8 + OPENCV + BYTETRACK + STREAMLIT
    ⬡  AI COURSE PROJECT
</div>
""", unsafe_allow_html=True)