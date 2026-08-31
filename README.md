# AI-Powered Gesture Game Controller 🎮✋

A real-time, vision-based virtual game controller that converts bare-hand gestures and swipes into keyboard/gamepad inputs using OpenCV, MediaPipe Tasks, and PyDirectInput.

---

## Features
- **Dual & Single Hand Steering:** Drive using single hand position tracking or virtual dual-hand steering wheel tilt.
- **Dynamic Swipe Detection:** Sliding position buffer queue optimized for smooth swipe actions at 30 FPS.
- **Intuitive Controls:**
  - **Gas / Accelerate:** Open palm (3+ fingers extended) $\rightarrow$ `UP Arrow`
  - **Brake / Reverse:** Closed fist $\rightarrow$ `DOWN Arrow`
  - **Steering:** Left/Right wrist offset or hand tilt $\rightarrow$ `LEFT / RIGHT Arrows`
  - **Jump / Nitro:** Thumb + Index pinch or quick Up-Swipe $\rightarrow$ `SPACE`
- **Real-Time HUD:** OpenCV overlay displaying FPS, active action states, steering angle, and swipe logs.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/satvik89076-gif/gesture-game-controller.git](https://github.com/satvik89076-gif/gesture-game-controller.git)
   cd gesture-game-controller