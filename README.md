# 🛡️ Driver Safety Monitoring System

AI-powered real-time drowsiness, yawn, distraction and head-nod detection
using **Streamlit + OpenCV + MediaPipe**.

---

## 🚀 Quick Start

### 1. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🧩 Features

| Feature | Description |
|---|---|
| 👁️ EAR Drowsiness | Eye Aspect Ratio below threshold for N frames → alert |
| 🥱 Yawn Detection | Mouth Aspect Ratio (MAR) threshold → yawn alert |
| ⚡ Head Nod | Sudden nose-tip Y drop → micro-sleep alert |
| 📍 Distraction | Face off-centre > 2.5 s → distraction alert |
| 📊 Fatigue Score | 0-100 composite score (🟢 Safe / 🟡 Warning / 🔴 Critical) |
| 🔊 Audio Alerts | Pygame synthesised beeps per alert type |
| 📁 CSV Logging | Auto-saved trip log with timestamps |
| 🌙 Dark / Light | Toggle in dashboard header |

---

## 📦 Package Notes

- **mediapipe** — requires Python 3.8–3.11 and a 64-bit OS
- **pygame** — used for synthesised beep alerts (no external audio files needed)
- If `pygame` fails to install, the app still works silently

---

## 🗂️ Project Structure

```
drowsiness_detection/
├── app.py            ← single runnable Streamlit app
├── requirements.txt  ← all Python dependencies
└── README.md         ← this file
```

Trip CSV logs are saved to the same directory as `app.py`.

---

## ⚙️ Adjustable Thresholds (top of app.py)

```python
EAR_THRESHOLD       = 0.22   # lower → more sensitive
EAR_CONSEC_FRAMES   = 20     # fewer → faster alert
MAR_THRESHOLD       = 0.65   # yawn sensitivity
DISTRACTION_SECONDS = 2.5    # seconds off-centre before alert
```

---

## 🎓 College Demo Tips

1. Run in a well-lit room for best face detection
2. Sit about **50–80 cm** from the webcam
3. Enter your name on the dashboard before starting
4. Click **Stop & Summary** to see the full trip report and download CSV

---

*Built with ❤️ using Streamlit, OpenCV, and MediaPipe*
