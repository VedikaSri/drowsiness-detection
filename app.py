"""
╔══════════════════════════════════════════════════════════════════════╗
║      DRIVER SAFETY MONITORING SYSTEM — app.py  (Friendly UI)         ║
║  Stack: Streamlit · OpenCV · MediaPipe · NumPy · Pandas · Pygame     ║
╚══════════════════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

# ─────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import datetime
import os
import math
import random
from collections import deque
import winsound

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drive Safe",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────
# DETECTION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────
EAR_THRESHOLD        = 0.20   # below this = eyes closing
EAR_CONSEC_FRAMES    = 30     # frames below threshold → drowsy
MAR_THRESHOLD        = 0.65   # mouth open ratio → yawn
MAR_CONSEC_FRAMES    = 30     # frames open → yawn alert
NOD_THRESHOLD        = 0.15   # nose-tip Y drop → head nod
DISTRACTION_SECONDS  = 2.5    # seconds off-centre → distraction
HEAD_OFF_CENTRE      = 0.28   # fraction of frame width

# MediaPipe landmark indices (468-point face mesh)
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]
MOUTH_IDX     = [61,  291, 39,  181, 0,   17,  269, 405]

SAFETY_QUOTES = [
    "Someone is waiting for you at home. 🏠",
    "Speed thrills, but it also kills. Drive safe. ❤️",
    "Patience on the road saves lives. 🛡️",
    "No destination is worth your life. 🌟",
    "Eyes on road, hands on wheel. 👁️",
    "Rest when tired — your family needs you. 💛",
]

INDIA_STATS = [
    "🇮🇳  India records ~1.5 lakh road fatalities every year",
    "😴  Drowsy driving causes 40% of highway accidents in India",
    "⏰  Most accidents occur between midnight and 6 AM",
    "🚛  Fatigued truck drivers cause 30% of highway deaths",
    "👀  A 2-second distraction at 80 km/h = 44 metres driven blind",
    "🌙  Night-time driving risk is 3× higher than daytime",
]

# ─────────────────────────────────────────────────────────────────────
# CSS — Crextio-inspired warm card dashboard
# ─────────────────────────────────────────────────────────────────────
def inject_css(dark: bool):
    if dark:
        bg        = "#18181b"
        surface   = "#27272a"
        card      = "#3f3f46"
        accent    = "#f5c518"
        text      = "#fafafa"
        muted     = "#a1a1aa"
        border    = "#52525b"
        green_bg  = "#14532d"
        green_txt = "#86efac"
        yellow_bg = "#713f12"
        yellow_txt= "#fde68a"
        red_bg    = "#7f1d1d"
        red_txt   = "#fca5a5"
        hero_grad = "linear-gradient(135deg, #27272a 0%, #3f3f46 100%)"
    else:
        bg        = "#f5f5f0"
        surface   = "#fafaf7"
        card      = "#ffffff"
        accent    = "#e8b800"
        text      = "#1c1c1e"
        muted     = "#6b7280"
        border    = "#e5e7eb"
        green_bg  = "#dcfce7"
        green_txt = "#166534"
        yellow_bg = "#fef9c3"
        yellow_txt= "#854d0e"
        red_bg    = "#fee2e2"
        red_txt   = "#991b1b"
        hero_grad = "linear-gradient(135deg, #fffbeb 0%, #fef3c7 50%, #fdf8f0 100%)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Serif+Display:ital@0;1&display=swap');

    :root {{
        --bg:        {bg};
        --surface:   {surface};
        --card:      {card};
        --accent:    {accent};
        --text:      {text};
        --muted:     {muted};
        --border:    {border};
        --green-bg:  {green_bg};
        --green-txt: {green_txt};
        --yellow-bg: {yellow_bg};
        --yellow-txt:{yellow_txt};
        --red-bg:    {red_bg};
        --red-txt:   {red_txt};
        --hero-grad: {hero_grad};
        --radius:    16px;
        --shadow:    0 1px 4px rgba(0,0,0,.08), 0 4px 16px rgba(0,0,0,.06);
    }}

    html, body, [data-testid="stApp"] {{
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif;
    }}

    #MainMenu, footer, header {{ visibility: hidden; }}
    [data-testid="stSidebar"] {{ display: none; }}
    .block-container {{ padding: 1.2rem 2rem !important; max-width: 1380px; margin: 0 auto; }}
    [data-testid="stVerticalBlock"] > div {{ gap: 0.7rem; }}

    /* ── Top nav bar ── */
    .nav-bar {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.75rem 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: var(--shadow);
        margin-bottom: 1.2rem;
    }}
    .nav-logo {{
        font-family: 'DM Serif Display', serif;
        font-size: 1.3rem;
        font-weight: 400;
        color: var(--text);
        background: var(--accent);
        padding: 0.2rem 0.8rem;
        border-radius: 8px;
        letter-spacing: -0.01em;
    }}
    .nav-links {{
        display: flex;
        gap: 0.5rem;
        align-items: center;
    }}
    .nav-link {{
        padding: 0.35rem 0.9rem;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 500;
        color: var(--muted);
        cursor: pointer;
        transition: background .15s;
    }}
    .nav-link.active {{
        background: var(--text);
        color: {'#fff' if not dark else '#18181b'};
    }}

    /* ── Hero welcome section ── */
    .hero-section {{
        background: var(--hero-grad);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 2rem 2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow);
    }}
    .hero-greeting {{
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.25rem;
    }}
    .hero-title {{
        font-family: 'DM Serif Display', serif;
        font-size: clamp(2rem, 3.5vw, 3rem);
        font-weight: 400;
        color: var(--text);
        line-height: 1.15;
        margin-bottom: 0.4rem;
    }}
    .hero-sub {{
        font-size: 0.95rem;
        color: var(--muted);
        margin-bottom: 1rem;
    }}

    /* ── Progress pills (top stats) ── */
    .progress-row {{
        display: flex;
        gap: 1rem;
        margin-top: 0.5rem;
        flex-wrap: wrap;
    }}
    .pill {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.82rem;
        color: var(--muted);
        font-weight: 500;
    }}
    .pill-bar {{
        width: 80px;
        height: 26px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 600;
    }}
    .pill-dark  {{ background: var(--text); color: {'#fff' if not dark else '#18181b'}; }}
    .pill-accent {{ background: var(--accent); color: #1c1c1e; }}
    .pill-outline {{ background: transparent; border: 2px dashed var(--border); color: var(--muted); }}

    /* ── Big stat numbers (top right) ── */
    .big-stats {{
        display: flex;
        gap: 2rem;
        align-items: center;
        justify-content: flex-end;
    }}
    .big-stat {{
        text-align: center;
    }}
    .big-stat-num {{
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem;
        line-height: 1;
        color: var(--text);
    }}
    .big-stat-label {{
        font-size: 0.78rem;
        color: var(--muted);
        margin-top: 0.1rem;
    }}

    /* ── Cards ── */
    .ds-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.2rem 1.4rem;
        box-shadow: var(--shadow);
        height: 100%;
    }}
    .card-label {{
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.6rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .card-arrow {{
        font-size: 0.9rem;
        color: var(--muted);
        cursor: pointer;
    }}

    /* ── Quote box ── */
    .quote-box {{
        background: var(--accent);
        color: #1c1c1e;
        border-radius: var(--radius);
        padding: 1rem 1.2rem;
        font-size: 0.95rem;
        font-weight: 500;
        font-style: italic;
        line-height: 1.5;
        margin-bottom: 0.8rem;
    }}

    /* ── Stat ticker ── */
    .stat-ticker {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-size: 0.82rem;
        color: var(--muted);
        line-height: 1.5;
    }}

    /* ── How-it-works list ── */
    .how-item {{
        display: flex;
        gap: 0.8rem;
        align-items: flex-start;
        padding: 0.6rem 0;
        border-bottom: 1px solid var(--border);
    }}
    .how-item:last-child {{ border-bottom: none; }}
    .how-icon {{
        font-size: 1.3rem;
        min-width: 2rem;
        text-align: center;
    }}
    .how-text {{ font-size: 0.88rem; color: var(--text); line-height: 1.45; }}
    .how-title {{ font-weight: 600; display: block; margin-bottom: 0.1rem; }}

    /* ── Name input ── */
    .stTextInput input {{
        background: var(--surface) !important;
        border: 2px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1rem !important;
        padding: 0.7rem 1rem !important;
        transition: border-color .2s;
    }}
    .stTextInput input:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px {'rgba(232,184,0,.15)' if not dark else 'rgba(245,197,24,.15)'} !important;
    }}
    label {{ color: var(--muted) !important; font-size: 0.82rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.04em; }}

    /* ── Primary start button ── */
    .stButton > button {{
        background: var(--text) !important;
        color: {'#fff' if not dark else '#18181b'} !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: opacity .18s, transform .15s !important;
        width: 100% !important;
    }}
    .stButton > button:hover {{ opacity: 0.85; transform: translateY(-2px); }}

    .accent-btn > button {{
        background: var(--accent) !important;
        color: #1c1c1e !important;
    }}

    .stop-btn > button {{
        background: var(--red-bg) !important;
        color: var(--red-txt) !important;
        border: 2px solid var(--red-txt) !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1rem !important;
    }}

    /* ── Toggle ── */
    [data-testid="stToggle"] {{ accent-color: var(--accent); }}

    /* ── Alert banners ── */
    .alert-banner {{
        border-radius: 12px;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        animation: pulse-border 1.5s ease-in-out infinite;
        margin-bottom: 0.8rem;
    }}
    .alert-safe    {{ background: var(--green-bg);  color: var(--green-txt);  border: 2px solid var(--green-txt); animation: none; }}
    .alert-drowsy  {{ background: var(--red-bg);    color: var(--red-txt);    border: 2px solid var(--red-txt); }}
    .alert-yawn    {{ background: var(--yellow-bg); color: var(--yellow-txt); border: 2px solid var(--yellow-txt); }}
    .alert-distract{{ background: var(--yellow-bg); color: var(--yellow-txt); border: 2px solid var(--yellow-txt); }}
    .alert-nod     {{ background: var(--red-bg);    color: var(--red-txt);    border: 2px solid var(--red-txt); }}

    @keyframes pulse-border {{
        0%, 100% {{ box-shadow: 0 0 0 0 rgba(239,68,68,0); }}
        50%  {{ box-shadow: 0 0 0 6px rgba(239,68,68,.2); }}
    }}

    /* ── Status panel ── */
    .status-panel {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.2rem;
        box-shadow: var(--shadow);
        margin-bottom: 0.7rem;
    }}
    .status-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        border-bottom: 1px solid var(--border);
        font-size: 0.88rem;
    }}
    .status-row:last-child {{ border-bottom: none; }}
    .status-key   {{ color: var(--muted); font-weight: 500; }}
    .status-val   {{ font-weight: 700; color: var(--text); }}
    .badge-green  {{ background: var(--green-bg);  color: var(--green-txt);  border-radius: 6px; padding: 2px 10px; }}
    .badge-yellow {{ background: var(--yellow-bg); color: var(--yellow-txt); border-radius: 6px; padding: 2px 10px; }}
    .badge-red    {{ background: var(--red-bg);    color: var(--red-txt);    border-radius: 6px; padding: 2px 10px; }}

    /* ── Fatigue meter ── */
    .fatigue-meter {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.7rem;
    }}
    .fatigue-label {{
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.6rem;
    }}
    .fatigue-bar-wrap {{
        background: var(--border);
        border-radius: 99px;
        height: 12px;
        overflow: hidden;
        margin-bottom: 0.4rem;
    }}
    .fatigue-bar {{
        height: 100%;
        border-radius: 99px;
        transition: width .5s ease;
    }}
    .fatigue-bar-safe   {{ background: linear-gradient(90deg, #4ade80, #22c55e); }}
    .fatigue-bar-warn   {{ background: linear-gradient(90deg, #fbbf24, #f59e0b); }}
    .fatigue-bar-crit   {{ background: linear-gradient(90deg, #f87171, #ef4444); }}
    .fatigue-num {{
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1;
    }}
    .fatigue-state {{
        font-size: 0.85rem;
        color: var(--muted);
        margin-top: 0.2rem;
    }}

    /* ── Alert count cards ── */
    .alert-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.6rem;
        margin-top: 0.5rem;
    }}
    .alert-item {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        text-align: center;
    }}
    .alert-item-num  {{ font-size: 1.5rem; font-weight: 700; color: var(--text); }}
    .alert-item-name {{ font-size: 0.75rem; color: var(--muted); font-weight: 500; margin-top: 0.1rem; }}

    /* ── Trip summary ── */
    .summary-card {{
        background: var(--hero-grad);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem;
        box-shadow: var(--shadow);
        text-align: center;
    }}
    .summary-title {{
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }}
    .summary-msg {{
        font-size: 0.95rem;
        color: var(--muted);
        margin-bottom: 1.2rem;
    }}
    .summary-stats {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.8rem;
        margin-top: 1rem;
    }}
    .summary-stat {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.9rem;
    }}
    .stDownloadButton > button {{
    background-color: #111827 !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    }}
    .summary-stat-num  {{ font-size: 1.6rem; font-weight: 700; }}
    .summary-stat-name {{ font-size: 0.78rem; color: var(--muted); margin-top: 0.15rem; }}

    [data-testid="stMetric"] {{ display: none; }}

    .divider {{ border: none; border-top: 1px solid var(--border); margin: 0.8rem 0; }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "monitoring":      False,
        "trip_done":       False,
        "dark_mode":       False,
        "driver_name":     "",
        "trip_start":      None,
        "log_rows":        [],
        "alert_counts":    {"drowsy": 0, "yawn": 0, "distract": 0, "nod": 0},
        "ear_history":     deque(maxlen=100),
        "fatigue_history": deque(maxlen=100),
        "ear_counter":     0,
        "mar_counter":     0,
        "distract_counter":0,
        "nod_counter":     0,
        "prev_nose_y":     None,
        "cam_error":       "",
        "quote_idx":       0,
        "stat_idx":        0,
        "last_quote_time": time.time(),
        "last_stat_time":  time.time(),
        "trip_duration":   0,
        "peak_fatigue":    0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────
# HELPER: ROTATING TEXT
# ─────────────────────────────────────────────────────────────────────
def get_rotating(idx_key, items, interval=8):
    now = time.time()
    time_key = f"last_{idx_key}_time"
    if time_key not in st.session_state:
        st.session_state[time_key] = now
    if now - st.session_state[time_key] > interval:
        st.session_state[idx_key] = (st.session_state[idx_key] + 1) % len(items)
        st.session_state[time_key] = now
    return items[st.session_state[idx_key]]


# ─────────────────────────────────────────────────────────────────────
# AUDIO
# ─────────────────────────────────────────────────────────────────────
def beep(freq=1000, duration_ms=500):
    try:
        winsound.Beep(freq, duration_ms)
    except:
        pass

_last_beep: dict = {}

def alert_sound(kind: str):
    now = time.time()
    delay = {"drowsy": 3, "yawn": 4, "distract": 3, "nod": 2}

    if now - _last_beep.get(kind, 0) > delay.get(kind, 3):
        beep()
        _last_beep[kind] = now


# ─────────────────────────────────────────────────────────────────────
# DETECTION MATH
# ─────────────────────────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    A = math.dist(pts[1], pts[5])
    B = math.dist(pts[2], pts[4])
    C = math.dist(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def mouth_aspect_ratio(landmarks, indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    ver = math.dist(pts[2], pts[6]) + math.dist(pts[3], pts[7])
    hor = math.dist(pts[0], pts[1])
    return ver / (2.0 * hor + 1e-6)


def nose_position(landmarks, w, h):
    nx = landmarks[1].x * w
    ny = landmarks[1].y * h
    return nx, ny


# ─────────────────────────────────────────────────────────────────────
# FATIGUE SCORE (0-100, simple + friendly)
# ─────────────────────────────────────────────────────────────────────
def compute_fatigue(ear, ear_consec, mar_consec, distract_consec):
    score = 0
    if ear < EAR_THRESHOLD:
        score += min(40, (ear_consec / EAR_CONSEC_FRAMES) * 40)
    if mar_consec > 0:
        score += min(25, (mar_consec / MAR_CONSEC_FRAMES) * 25)
    if distract_consec > 0:
        score += min(25, (distract_consec / (DISTRACTION_SECONDS * 30)) * 25)
    score = min(100, score)
    if score < 35:
        return score, "Safe", "safe"
    elif score < 65:
        return score, "⚠️ Getting Tired", "warn"
    else:
        return score, "🔴 Very Tired!", "crit"


# ─────────────────────────────────────────────────────────────────────
# HUD OVERLAY ON FRAME
# ─────────────────────────────────────────────────────────────────────
def draw_overlay(frame, ear, mar, state, fatigue):
    h, w = frame.shape[:2]

    # Colour based on state
    colour_map = {
        "ALERT":      (80, 200, 120),
        "DROWSY!":    (60, 60, 220),
        "YAWNING!":   (30, 180, 230),
        "DISTRACTED!":(30, 160, 230),
        "HEAD NOD!":  (20, 20, 220),
    }
    col = colour_map.get(state, (80, 200, 120))

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # State badge
    cv2.putText(frame, state, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, col, 2)

    # EAR + MAR small text
    cv2.putText(frame, f"EYE:{ear:.2f}  MOUTH:{mar:.2f}",
                (w - 230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Fatigue bar at bottom
    bar_w = int(w * fatigue / 100)
    bar_col = (80, 200, 120) if fatigue < 35 else (30, 180, 230) if fatigue < 65 else (60, 60, 220)
    cv2.rectangle(frame, (0, h - 8), (w, h), (40, 40, 40), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (0, h - 8), (bar_w, h), bar_col, -1)

    return frame


# ─────────────────────────────────────────────────────────────────────
# MAIN MONITORING LOOP
# ─────────────────────────────────────────────────────────────────────
def run_monitoring(frame_ph, alert_ph, status_ph, fatigue_ph, counts_ph):
    ss  = st.session_state
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        ss.cam_error = "❌ Cannot open camera. Check webcam connection."
        return

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    distract_start = None
    looking_away   = False
    ear_buf        = deque(maxlen=60)

    try:
        while ss.monitoring:
            ret, frame = cap.read()
            if not ret:
                ss.cam_error = "⚠️ Camera feed lost."
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = face_mesh.process(rgb)

            state = "ALERT"
            ear   = 0.30
            mar   = 0.0
            fatigue = 0

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark

                # ── Eye Aspect Ratio ──────────────────────────────────
                ear_l = eye_aspect_ratio(lm, LEFT_EYE_EAR,  w, h)
                ear_r = eye_aspect_ratio(lm, RIGHT_EYE_EAR, w, h)
                ear   = (ear_l + ear_r) / 2.0
                ear_buf.append(ear)

                if ear < EAR_THRESHOLD:
                    ss.ear_counter += 1
                    if ss.ear_counter == EAR_CONSEC_FRAMES:
                        state = "DROWSY!"
                        alert_sound("drowsy")
                        ss.alert_counts["drowsy"] += 1
                else:
                    ss.ear_counter = 0  # reset only when eyes open again

                # ── Mouth / Yawn ──────────────────────────────────────
                mar = mouth_aspect_ratio(lm, MOUTH_IDX, w, h)
                if mar > MAR_THRESHOLD:
                    ss.mar_counter += 1
                    if ss.mar_counter == MAR_CONSEC_FRAMES:
                        if state == "ALERT":
                            state = "YAWNING!"
                        alert_sound("yawn")
                        ss.alert_counts["yawn"] += 1
                else:
                    ss.mar_counter = 0  # reset only when mouth closes

                # ── Distraction (looking away) ────────────────────────
                nx, ny = nose_position(lm, w, h)
                off_centre = abs(nx - w / 2) / w

                # If driver looking away
                if off_centre > HEAD_OFF_CENTRE:
                    if distract_start is None:
                        distract_start = time.time()

                    elapsed = time.time() - distract_start

                    if elapsed > DISTRACTION_SECONDS:
                        state = "DISTRACTED!"
                        alert_sound("distract")

                        # COUNT ONLY ONCE PER EVENT
                        if ss.distract_counter == 0:
                            ss.alert_counts["distract"] += 1
                            ss.distract_counter = 1

                # If driver looks back
                else:
                    distract_start = None
                    ss.distract_counter = 0

                # ── Head Nod ──────────────────────────────────────────
                norm_y = lm[1].y
                if ss.prev_nose_y is not None:
                    drop = norm_y - ss.prev_nose_y
                    if drop > NOD_THRESHOLD:
                        ss.nod_counter += 1
                        if ss.nod_counter >= 2:
                            if state == "ALERT":
                                state = "HEAD NOD!"
                            alert_sound("nod")
                            ss.alert_counts["nod"] += 1
                            ss.nod_counter = 0
                    else:
                        ss.nod_counter = 0
                ss.prev_nose_y = norm_y

                # ── Fatigue score ─────────────────────────────────────
                fatigue, f_label, f_cls = compute_fatigue(
                    ear, ss.ear_counter, ss.mar_counter,
                    (time.time() - distract_start) * 30 if distract_start else 0
                )
                ss.ear_history.append(ear)
                ss.fatigue_history.append(fatigue)
                ss.peak_fatigue = max(ss.peak_fatigue, fatigue)

                # ── Log row ───────────────────────────────────────────
                ss.log_rows.append({
                    "time":    datetime.datetime.now().isoformat(timespec="seconds"),
                    "ear":     round(ear, 3),
                    "mar":     round(mar, 3),
                    "fatigue": round(fatigue, 1),
                    "state":   state,
                })
            else:
                # No face → treat as looking away
                if distract_start is None:
                    distract_start = time.time()
                elif time.time() - distract_start > DISTRACTION_SECONDS:
                    state = "DISTRACTED!"
                    alert_sound("distract")

            # ── Draw HUD on frame ─────────────────────────────────────
            frame = draw_overlay(frame, ear, mar, state, fatigue)
            frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                           channels="RGB", use_container_width=True)

            # ── Alert banner ──────────────────────────────────────────
            elapsed = int(time.time() - ss.trip_start)
            mins, secs = divmod(elapsed, 60)
            if state == "DROWSY!":
                alert_ph.markdown(
                    '<div class="alert-banner alert-drowsy">😴  DROWSY!  Pull over and rest immediately.</div>',
                    unsafe_allow_html=True)
            elif state == "HEAD NOD!":
                alert_ph.markdown(
                    '<div class="alert-banner alert-nod">⚡  HEAD NOD detected — micro-sleep risk! Stop now.</div>',
                    unsafe_allow_html=True)
            elif state == "YAWNING!":
                alert_ph.markdown(
                    '<div class="alert-banner alert-yawn">🥱  YAWNING!  Take a short break soon.</div>',
                    unsafe_allow_html=True)
            elif state == "DISTRACTED!":
                alert_ph.markdown(
                    '<div class="alert-banner alert-distract">👁️  LOOKING AWAY!  Eyes on the road.</div>',
                    unsafe_allow_html=True)
            else:
                alert_ph.markdown(
                    '<div class="alert-banner alert-safe">✅  All Clear — You are driving safely.</div>',
                    unsafe_allow_html=True)

            # ── Status panel ──────────────────────────────────────────
            eye_badge = "badge-green" if ear >= EAR_THRESHOLD else "badge-red"
            state_badge = "badge-green" if state == "ALERT" else ("badge-red" if state in ["DROWSY!", "HEAD NOD!"] else "badge-yellow")
            status_ph.markdown(f"""
            <div class="status-panel">
                <div class="status-row">
                    <span class="status-key">🎥 Camera</span>
                    <span class="status-val"><span class="badge-green" style="background:var(--green-bg);color:var(--green-txt);border-radius:6px;padding:2px 10px;">LIVE</span></span>
                </div>
                <div class="status-row">
                    <span class="status-key">👁️ Eyes</span>
                    <span class="status-val"><span class="{eye_badge}" style="background:var(--{'green' if ear >= EAR_THRESHOLD else 'red'}-bg);color:var(--{'green' if ear >= EAR_THRESHOLD else 'red'}-txt);border-radius:6px;padding:2px 10px;">{'Open' if ear >= EAR_THRESHOLD else 'Closing'}</span></span>
                </div>
                <div class="status-row">
                    <span class="status-key">🚗 Driver State</span>
                    <span class="status-val"><span class="{state_badge}" style="background:var(--{'green' if state=='ALERT' else ('red' if state in ['DROWSY!','HEAD NOD!'] else 'yellow')}-bg);color:var(--{'green' if state=='ALERT' else ('red' if state in ['DROWSY!','HEAD NOD!'] else 'yellow')}-txt);border-radius:6px;padding:2px 10px;">{state}</span></span>
                </div>
                <div class="status-row">
                    <span class="status-key">⏱️ Trip Duration</span>
                    <span class="status-val">{mins:02d}:{secs:02d}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Fatigue meter ─────────────────────────────────────────
            f_score, f_label, f_cls = compute_fatigue(
                ear, ss.ear_counter, ss.mar_counter,
                (time.time() - distract_start) * 30 if distract_start else 0
            )
            bar_class = f"fatigue-bar-{f_cls}"
            num_colour = {"safe": "var(--green-txt)", "warn": "var(--yellow-txt)", "crit": "var(--red-txt)"}[f_cls]
            fatigue_ph.markdown(f"""
            <div class="fatigue-meter">
                <div class="fatigue-label">Fatigue Level</div>
                <div style="font-size:2rem;font-weight:700;color:{num_colour};line-height:1;">{int(f_score)}<span style="font-size:1rem;font-weight:400;color:var(--muted);">/100</span></div>
                <div style="font-size:0.82rem;color:var(--muted);margin:.2rem 0 .5rem;">{f_label}</div>
                <div class="fatigue-bar-wrap">
                    <div class="fatigue-bar {bar_class}" style="width:{f_score}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Alert counts ──────────────────────────────────────────
            ac = ss.alert_counts
            counts_ph.markdown(f"""
            <div class="alert-grid">
                <div class="alert-item">
                    <div class="alert-item-num">{ac['drowsy']}</div>
                    <div class="alert-item-name">😴 Drowsy</div>
                </div>
                <div class="alert-item">
                    <div class="alert-item-num">{ac['yawn']}</div>
                    <div class="alert-item-name">🥱 Yawn</div>
                </div>
                <div class="alert-item">
                    <div class="alert-item-num">{ac['distract']}</div>
                    <div class="alert-item-name">👁️ Distracted</div>
                </div>
                <div class="alert-item">
                    <div class="alert-item-num">{ac['nod']}</div>
                    <div class="alert-item-name">⚡ Head Nod</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    finally:
        cap.release()
        face_mesh.close()


# ─────────────────────────────────────────────────────────────────────
# TRIP SUMMARY
# ─────────────────────────────────────────────────────────────────────
def show_trip_summary():
    ss   = st.session_state
    ac   = ss.alert_counts
    dur  = int(ss.trip_duration)
    mins, secs = divmod(dur, 60)
    total_alerts = min(99, sum(ac.values()))
    peak  = int(ss.peak_fatigue)

    if peak < 35:
        grade = "🟢 Excellent"
        msg   = "Great drive! You stayed alert the whole time. 🎉"
    elif peak < 65:
        grade = "🟡 Good"
        msg   = "Decent trip, but watch out for fatigue next time. 💛"
    else:
        grade = "🔴 Needs Improvement"
        msg   = "Please rest before driving again. Safety first! ❤️"

    st.markdown(f"""
    <div class="summary-card">
        <div class="summary-title">🏁 Trip Complete</div>
        <div class="summary-msg">{msg}</div>
        <div style="font-size:1.1rem;font-weight:600;margin-bottom:1rem;">Safety Grade: {grade}</div>
        <div class="summary-stats">
            <div class="summary-stat">
                <div class="summary-stat-num">{mins:02d}:{secs:02d}</div>
                <div class="summary-stat-name">⏱️ Duration</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-num">{total_alerts}</div>
                <div class="summary-stat-name">🚨 Total Alerts</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-num">{ac['drowsy']}</div>
                <div class="summary-stat-name">😴 Drowsy Events</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-num">{peak}</div>
                <div class="summary-stat-name">🔥 Peak Fatigue</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # CSV download
    if ss.log_rows:
        df = pd.DataFrame(ss.log_rows)
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Trip Report (CSV)",
            data=csv,
            file_name=f"trip_{datetime.date.today()}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────
# DASHBOARD (HOME SCREEN)
# ─────────────────────────────────────────────────────────────────────
def render_dashboard():
    ss = st.session_state

    # ── Nav bar ───────────────────────────────────────────────────────
    col_nav, col_toggle = st.columns([6, 1])
    with col_nav:
        st.markdown("""
        <div class="nav-bar">
            <span class="nav-logo">DriveSafe</span>
            <div class="nav-links">
                <span class="nav-link active">Dashboard</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_toggle:
        st.markdown("<br>", unsafe_allow_html=True)
        ss.dark_mode = st.toggle("🌙", value=ss.dark_mode)

    # ── Hero section ──────────────────────────────────────────────────
    quote = get_rotating("quote_idx", SAFETY_QUOTES, interval=8)
    stat  = get_rotating("stat_idx",  INDIA_STATS,   interval=8)

    hero_left, hero_right = st.columns([2, 1])
    with hero_left:
        name_display = ss.driver_name or "Driver"
        st.markdown(f"""
        <div class="hero-section">
            <div class="hero-greeting">Good day 👋</div>
            <div class="hero-title">Welcome, {name_display}</div>
            <div class="hero-sub">Your AI co-pilot is ready to keep you safe on the road.</div>
            <div class="progress-row">
                <span class="pill">Drowsy Alerts <span class="pill-bar pill-dark">0</span></span>
                <span class="pill">Yawn Alerts <span class="pill-bar pill-accent">0</span></span>
                <span class="pill">Distraction <span class="pill-bar pill-outline">0%</span></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with hero_right:
        st.markdown("""
        <div class="hero-section" style="height:100%;display:flex;flex-direction:column;justify-content:center;">
            <div class="big-stats">
                <div class="big-stat">
                    <div class="big-stat-num">1.5L</div>
                    <div class="big-stat-label">Road Deaths/Year</div>
                </div>
                <div class="big-stat">
                    <div class="big-stat-num">40%</div>
                    <div class="big-stat-label">Due to Fatigue</div>
                </div>
                <div class="big-stat">
                    <div class="big-stat-num">3×</div>
                    <div class="big-stat-label">Night Risk</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main content ──────────────────────────────────────────────────
    left_col, right_col = st.columns([1.2, 1], gap="large")

    with left_col:
        # Quote
        st.markdown(f'<div class="quote-box">💬 &nbsp;{quote}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-ticker">📊 &nbsp;{stat}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Name input
        st.markdown('<label>👤 Your Name</label>', unsafe_allow_html=True)
        ss.driver_name = st.text_input(
            "Name",
            value=ss.driver_name,
            placeholder="e.g. Rahul Sharma",
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Start button
        st.markdown('<div class="accent-btn">', unsafe_allow_html=True)
        if st.button("🚗  Start Your Journey", use_container_width=True):
            ss.monitoring   = True
            ss.trip_start   = time.time()
            ss.trip_done    = False
            ss.log_rows     = []
            ss.alert_counts = {"drowsy": 0, "yawn": 0, "distract": 0, "nod": 0}
            ss.ear_history.clear()
            ss.fatigue_history.clear()
            ss.ear_counter  = ss.mar_counter = ss.distract_counter = ss.nod_counter = 0
            ss.prev_nose_y  = None
            ss.cam_error    = ""
            ss.peak_fatigue = 0
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="ds-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">How It Keeps You Safe <span class="card-arrow">↗</span></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="how-item">
            <div class="how-icon">👁️</div>
            <div class="how-text">
                <span class="how-title">Eye Tracking</span>
                Watches your eyes. If they close for too long, an alarm sounds immediately.
            </div>
        </div>
        <div class="how-item">
            <div class="how-icon">🥱</div>
            <div class="how-text">
                <span class="how-title">Yawn Detection</span>
                Spots yawning early and alerts you to take a break before tiredness sets in.
            </div>
        </div>
        <div class="how-item">
            <div class="how-icon">📍</div>
            <div class="how-text">
                <span class="how-title">Distraction Guard</span>
                If you look away from the road for more than 2.5 seconds, it alerts you.
            </div>
        </div>
        <div class="how-item">
            <div class="how-icon">⚡</div>
            <div class="how-text">
                <span class="how-title">Head Nod Alert</span>
                Detects sudden head drops (micro-sleep) and wakes you with a loud alarm.
            </div>
        </div>
        <div class="how-item">
            <div class="how-icon">🔥</div>
            <div class="how-text">
                <span class="how-title">Fatigue Score</span>
                Combines all signals into a live fatigue level: 🟢 Safe · 🟡 Tired · 🔴 Critical.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if ss.cam_error:
        st.error(ss.cam_error)

    if ss.trip_done:
        st.markdown("<br>", unsafe_allow_html=True)
        show_trip_summary()


# ─────────────────────────────────────────────────────────────────────
# MONITORING SCREEN
# ─────────────────────────────────────────────────────────────────────
def render_monitoring():
    ss = st.session_state

    # ── Header ────────────────────────────────────────────────────────
    h_left, h_right = st.columns([4, 1])
    with h_left:
        name = ss.driver_name or "Driver"
        st.markdown(f"""
        <div class="nav-bar">
            <span class="nav-logo">DriveSafe</span>
            <span style="font-size:1rem;font-weight:600;color:var(--text);">🎥 Monitoring — {name}</span>
            <span style="font-size:0.82rem;color:var(--muted);">AI is watching. Stay safe.</span>
        </div>
        """, unsafe_allow_html=True)
    with h_right:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        if st.button("⛔ End Trip", use_container_width=True):
            ss.monitoring    = False
            ss.trip_done     = True
            ss.trip_duration = time.time() - ss.trip_start
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Alert banner placeholder ───────────────────────────────────────
    alert_ph = st.empty()

    # ── Two-column layout ─────────────────────────────────────────────
    cam_col, side_col = st.columns([1.6, 1], gap="medium")

    with cam_col:
        st.markdown('<div class="card-label">📷 Live Camera Feed</div>', unsafe_allow_html=True)
        frame_ph = st.empty()

    with side_col:
        status_ph  = st.empty()
        fatigue_ph = st.empty()
        st.markdown('<div class="card-label">🚨 Today\'s Alerts</div>', unsafe_allow_html=True)
        counts_ph  = st.empty()

    # ── Run monitoring loop ───────────────────────────────────────────
    run_monitoring(frame_ph, alert_ph, status_ph, fatigue_ph, counts_ph)

    ss.monitoring    = False
    ss.trip_done     = True
    ss.trip_duration = time.time() - (ss.trip_start or time.time())
    st.rerun()


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    init_state()
    inject_css(st.session_state.dark_mode)

    if st.session_state.monitoring:
        render_monitoring()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()
