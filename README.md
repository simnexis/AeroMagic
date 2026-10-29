<div align="center">

# ✨ AeroMagic

### *Real-time Computer Vision & Mystical Particle Spheres with MediaPipe & OpenCV*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-00C4B4?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> **Transforms natural webcam hand gestures into responsive, particle-rich mystical energy spheres rendered at 60 FPS.**

</div>

---

## 📸 Demo Preview

<div align="center">
  <img src="assets/aeromagic-demo.png" alt="AeroMagic Demo" width="700">
</div>

---

## ⚡ Key Highlights

- **🖐️ Sub-Millisecond Hand Tracking**: 21 3D hand-landmark topology tracking powered by Google MediaPipe Hands.
- **✨ Layered Radiance Physics**: Multi-stage alpha blending synthesizing concentric amber-gold, pale-gold, and crimson energy particle fields.
- **📏 Dynamic Depth Scaling**: Dynamically recalculates sphere radius and angular velocity based on landmark distance / apparent camera proximity.
- **👐 Bi-Manual Synthesis**: Concurrent dual-hand tracking with independent particle emitter physics.
- **🎯 Geometric Gesture Gating**: Aspect-ratio and fingertip-to-wrist vector checking to ensure spheres only summon on deliberate open-palm poses.

---

## 🏗️ Architecture

```text
Webcam Frame (OpenCV VideoCapture)
       │
       ▼
RGB Conversion & Landmark Inference (MediaPipe Hands)
       │
       ▼
Geometric Palm Verification (Wrist-to-MCP Distance & Finger Extension Ratios)
       │
   [Verified]
       │
       ▼
Particle Physics Simulation (Dynamic Radiance, Angular Velocity, Radial Blend)
       │
       ▼
Sub-pixel Alpha Blending & Particle Overlay
       │
       ▼
Real-time 60 FPS Display (cv2.imshow)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/simnexis/AeroMagic.git
cd AeroMagic
```

### 2. Set Up Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(On macOS / Linux: `source .venv/bin/activate`)*

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run AeroMagic

```bash
python AeroMagic.py
```

*Ensure your webcam is enabled and the room has reasonable lighting. Hold an open palm up to the camera to summon the sphere!*

---

## 🎮 Controls

| Action | Result |
| :--- | :--- |
| **Open Palm** | Summons and anchors mystical energy sphere |
| **Closed Fist / Relaxed Hand** | Dismisses particle field smoothly |
| **Move Hand Closer / Further** | Dynamically scales sphere diameter and particle speed |
| **`Q` or `Esc`** | Exit application cleanly |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **OpenCV (`opencv-python`)**: High-throughput video stream acquisition, matrix math, color space conversion, alpha blending.
- **MediaPipe (`mediapipe==0.10.21`)**: 21 3D hand landmarks detection with GPU/CPU acceleration.
- **NumPy**: Vectorized coordinate calculations and bounding radius computation.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) © 2026 Simran Gupta ([@simnexis](https://github.com/simnexis)).
