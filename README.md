# 🛡️ Driver Safety Monitoring System

AI-powered real-time driver monitoring system using **Streamlit, OpenCV, and MediaPipe**.

---

## 🚀 Features

* 👁️ Drowsiness Detection (Eye Aspect Ratio)
* 🥱 Yawn Detection (Mouth Aspect Ratio)
* 👁️ Distraction Detection (Face off-center)
* ⚡ Head Nod Detection (Micro-sleep)
* 📊 Fatigue Score (0–100)
* 🔊 Audio Alerts
* 📁 CSV Trip Logging
* 🌙 Dark / Light Mode UI

---

## 🧠 How It Works

The system uses **MediaPipe Face Mesh** to track facial landmarks and calculates:

* Eye Aspect Ratio (EAR)
* Mouth Aspect Ratio (MAR)

Based on these values, it detects:

* Drowsiness
* Yawning
* Distraction
* Head nods

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

Open:

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
* College projects

---

## ⚠️ Notes

* Works best in good lighting
* Keep face clearly visible
* Webcam required

---

## 👩‍💻 Author

Vedika Srivastava

---

## 💡 Future Improvements

* 🎤 Voice alerts
* 📊 Advanced analytics
* ☁️ Cloud deployment

---

⭐ If you like this project, give it a star!
