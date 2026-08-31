# AI-Powered Gesture Game Controller 🎮✋

A real-time, vision-based virtual game controller that converts bare-hand gestures and dynamic swipes into keyboard and gamepad inputs using OpenCV, MediaPipe Tasks, and PyDirectInput.

---

## 🚀 Features

- **Dual & Single Hand Steering:** Drive using single hand position tracking or virtual dual-hand steering wheel tilt.
- **Dynamic Swipe Detection:** Buffer-based queue optimized for reliable swipe triggers at 30 FPS.
- **Gesture Controls Mapping:**
  - **Gas / Accelerate:** Open palm (3+ fingers extended) → `UP Arrow`
  - **Brake / Reverse:** Closed fist → `DOWN Arrow`
  - **Steering:** Left/Right wrist offset or dual-hand tilt → `LEFT / RIGHT Arrows`
  - **Jump / Nitro:** Thumb + Index pinch or quick Up-Swipe → `SPACE`
- **Real-Time HUD Dashboard:** Live OpenCV overlay displaying active state, steering direction, FPS, and detected swipe logs.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Computer Vision:** MediaPipe Tasks API (Hand Landmarker), OpenCV
- **Input Emulation:** PyDirectInput (DirectX compatible)
- **Math & Utilities:** NumPy

---

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/satvik89076-gif/gesture-game-controller.git](https://github.com/satvik89076-gif/gesture-game-controller.git)
   cd gesture-game-controller