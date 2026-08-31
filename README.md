# AI-Powered Instant Virtual Game Controller 🎮✋

A real-time, low-latency vision-based virtual game controller that translates bare-hand positioning and pinch actions into instantaneous keyboard and gamepad events using OpenCV, MediaPipe Tasks, and PyDirectInput.

---

## 🚀 Key Features

- **Instant Virtual D-Pad Zones:** $O(1)$ constant-time spatial zone triggering for zero perceived input lag.
- **Visual Feedback HUD:** Active grid zones illuminate instantly upon hand entry.
- **Precision Pinch Detection:** Fast squared-Euclidean distance thresholding between thumb and index fingertips for action/jump triggers.
- **Direct Input Emulation:** PyDirectInput integration for direct hardware-level keystroke delivery into native and browser games.

---

## 🎮 Controls & Mechanics

| Movement / Action | Screen Trigger Zone | Emulated Key |
| :--- | :--- | :--- |
| **Throttle / Accelerate** | Hand in **Top Zone** | `UP Arrow` / `W` |
| **Brake / Reverse** | Hand in **Bottom Zone** | `DOWN Arrow` / `S` |
| **Steer / Move Left** | Hand in **Left Zone** | `LEFT Arrow` / `A` |
| **Steer / Move Right** | Hand in **Right Zone** | `RIGHT Arrow` / `D` |
| **Neutral / Coast** | Hand in **Center Box** | Key Release / None |
| **Jump / Nitro / Action** | **Thumb + Index Pinch** | `SPACE` |

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Computer Vision:** MediaPipe Tasks API (`HandLandmarker`), OpenCV (`cv2`)
- **Input Emulation:** PyDirectInput
- **Architecture:** Spatial bounding zones & vectorized Euclidean landmark tracking

---

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/satvik89076-gif/gesture-game-controller.git](https://github.com/satvik89076-gif/gesture-game-controller.git)
   cd gesture-game-controller