# 🛡️ Driver Safety Monitoring System

AI-powered real-time driver monitoring system using **Streamlit + OpenCV + MediaPipe**.

---

## 🚀 Features

* 👁️ **Drowsiness Detection (EAR)**
* 🥱 **Yawn Detection (MAR)**
* 👁️ **Distraction Detection (Face Off-Center)**
* ⚡ **Head Nod Detection (Micro-sleep)**
* 📊 **Fatigue Score (0–100)**
* 🔊 **Audio Alerts (Windows Beep)**
* 📁 **CSV Trip Logging**
* 🌙 **Dark / Light Mode UI**

---

## 🧠 How It Works

* Uses **MediaPipe Face Mesh** for facial landmarks
* Calculates:

  * Eye Aspect Ratio (EAR)
  * Mouth Aspect Ratio (MAR)
* Detects:

  * Eye closure duration
  * Yawning
  * Head movement
  * Face direction
* Generates alerts when unsafe behavior is detected

---

## 🖥️ Tech Stack

* Python 3.11
* Streamlit
* OpenCV
* MediaPipe
* NumPy
* Pandas

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## ⚙️ Adjustable Parameters

```python
EAR_THRESHOLD = 0.20
EAR_CONSEC_FRAMES = 30
MAR_THRESHOLD = 0.65
DISTRACTION_SECONDS = 2.5
```

---

## 📊 Project Structure

```
drowsiness_detection/
│── app.py
│── requirements.txt
│── README.md
```

---

## 🎯 Use Cases

* Driver safety systems
* Smart vehicles
* Fleet monitoring
* Research & college projects

---

## ⚠️ Notes

* Works best in **good lighting**
* Keep face clearly visible to camera
* Camera permission required

---

## 👩‍💻 Author

Vedika Srivastava

---

## 💡 Future Improvements

* 🎤 Voice alerts
* 📱 Mobile support
* 📊 Analytics dashboard
* ☁️ Cloud deployment

---

## ❤️ Built With

Streamlit + OpenCV + MediaPipe
