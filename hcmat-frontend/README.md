# H-CMAT Real-Time Dashboard 

### Culturally-Aware Multimodal Human-AI Interaction Framework

This is the high-performance frontend dashboard for the **H-CMAT (Human-Centric Multimodal Adaptation Tool)**. It is designed to visualize real-time telemetry from edge devices (Raspberry Pi 4 / Laptop) and demonstrate the **Uncertainty-Guided Fusion Layer** through interactive simulation.

---

## 🚀 Key Features

* **Live Multimodal Feed:** Low-latency webcam integration with SVG tracking overlays for `FACE_LOC` and `POSE` metrics.
* **Uncertainty-Guided Weighting:** Dynamic visualization of modality weights (Speech, Prosody, Facial, Gestures) that adapt based on environmental noise.
* **Autonomous Demo Mode:** A "Look, No Hands!" toggle that simulates chaotic environmental stressors to prove the model's continuous adaptation.
* **Cultural Context Engine:** Hot-swappable cultural profiles that shift the pragmatic inference logic (e.g., High-Context vs. Low-Context).
* **Zustand State Management:** Optimized for high-frequency updates (up to 60fps) without causing global re-renders.

---

## 🛠️ Technical Stack

* **Core:** React 18 + Vite (for ultra-fast HMR)
* **State:** Zustand (Atomic state management for telemetry)
* **Styling:** Native CSS Modules + CSS Variables (Zero-dependency, high-performance)
* **Icons:** Lucide-React
* **Networking:** WebSocket-ready architecture for FastAPI integration

---

## 📂 Project Structure

```text
src/
├── features/
│   ├── MultimodalFeed/      # Camera stream & tracking overlays
│   ├── InferenceMatrix/     # Real-time weights & H-CMAT vs. Baseline comparison
│   ├── EnvironmentalSensors/# Telemetry simulator (Auto/Manual mode)
│   └── ContextLogic/        # Cultural profile selection & logic descriptions
├── layouts/                 # CSS Grid-based dashboard skeleton
├── store/                   # Global Zustand store for live telemetry
└── styles/                  # Global theme variables (Dark Mode)