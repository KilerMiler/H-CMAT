# **H‑CMAT: Hierarchical Culturally‑Aware Multimodal Attention Transformer**

### *Real‑time, locally deployed, multimodal cultural‑pragmatic intelligence engine.*

***

## 🧠 **What is H‑CMAT?**

**H‑CMAT** is a high‑performance, real‑time **multimodal inference engine** that fuses:

*   **Speech (text + audio)**
*   **Facial expressions**
*   **Body posture + gestures**
*   **Hand / sign signals**

…into a single **culturally‑aware pragmatic intent**.

It runs **entirely offline**, in **real time**, with **explainability-by-design**, making it ideal for:

*   Research demonstrations
*   Human–AI interaction studies
*   Cross‑cultural communication AI
*   Edge‑compute multimodal systems

***

# 🚀 **Key Features**

### ✅ **True multimodal AI**

5 parallel encoders running in local unified memory:

| Modality       | Encoder                 |
| -------------- | ----------------------- |
| Speech (text)  | DistilRoBERTa Zero‑shot |
| Speech (audio) | Whisper‑Tiny            |
| Face           | MediaPipe Face Mesh     |
| Body           | MediaPipe Pose          |
| Hand/Sign      | MediaPipe Holistic      |

***

### ✅ **Cultural‑aware fusion**

Each frame is interpreted with respect to a cultural profile:

*   **IN — Indian (High‑Context)**
*   **JP — Japanese (Collectivist)**
*   **US — American (Low‑Context)**

Switch cultures **live during session** and watch the interpretation shift in real time.

***

### ✅ **Glass‑Box Transparency**

The React dashboard shows:

*   Live modality weights
*   Real‑time cross‑modal attention
*   Fusion decision traces
*   Session history (NMS-deduped)
*   Summarized session report

***

### ✅ **Offline, Real-Time**

All models run locally via:

*   **FastAPI**
*   **MPS acceleration** (Apple Silicon)
*   Shared thread‑pool encoder runner
*   WebSocket streaming (50% overlapping windows)

***

### ✅ **Event‑Level NMS**

Sliding windows produce overlapping chunks.

H‑CMAT deduplicates behavioral events via:

    same intent + overlapping clip windows → keep only higher confidence
    different intent → new event

Ensures clean, human-friendly session logs.

***

### 🏛 **Full System Architecture**

     ┌──────────────────────────────────────────────────────────┐
     │                        FRONTEND (React)                  │
     │  • Camera + Microphone capture                           │
     │  • Sliding WebM buffer                                   │
     │  • Worker: base64 audio + JPEG frame                     │
     │  • Live Glass Box matrix                                 │
     │  • Session History (NMS)                                 │
     │  • Cultural Selector (live switch)                       │
     └───────────────▲──────────────────────────────────────────┘
                     │ WS (chunk, heartbeat)
                     ▼
     ┌──────────────────────────────────────────────────────────┐
     │                       BACKEND (FastAPI)                  │
     │ • Session Manager (in‑memory)                            │
     │ • CulturalMapper (JSON profiles)                         │
     │ • ParallelEncoderRunner (shared executor)                │
     │     - Text   (DistilRoBERTa NLI)                         │
     │     - Audio  (Whisper-Tiny ASR)                          │
     │     - Face   (FaceMesh)                                  │
     │     - Body   (Pose)                                      │
     │     - Sign   (Holistic Hands)                            │
     │ • Fusion Layer (AttentionMath + cultural weights)        │
     │ • Window-NMS Event Ledger                                │
     │ • SQLite Persistence                                     │
     └───────────────▲──────────────────────────────────────────┘
                     │ HTTP (REST)
                     ▼
     ┌──────────────────────────────────────────────────────────┐
     │                     SESSION SUMMARY                      │
     │ • Dominant intent                                        │
     │ • Affective state                                        │
     │ • Cultural modifiers used                                │
     │ • Deduped event ledger                                   │
     └──────────────────────────────────────────────────────────┘

***

# 📦 **Tech Stack**

### **Backend**

*   FastAPI
*   Python
*   PyTorch (MPS acceleration)
*   Transformers (Text + Whisper ASR)
*   MediaPipe (Face + Pose + Hands)
*   PyAV (WebM decoding)
*   SQLite (session logs)

### **Frontend**

*   React (+ Vite)
*   Zustand state management
*   Web Workers (media encoding)
*   MediaRecorder (sliding window capture)
*   Custom Glass‑Box visualizations

***

# 🛠 **Installation**

### 1. Clone Repo

```bash
git clone https://github.com/<your-repo>/hcmat.git
cd hcmat
```

### 2. Backend Setup

Requires Python 3.10+ and Mac M1/M2/M3/M4 for MPS.

```bash
cd hcmat-backend
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Start Backend (IMPORTANT)

Only **1 worker** allowed because models are loaded in single process memory.

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

***

### 4. Frontend Setup

```bash
cd hcmat-frontend
npm install
npm run dev
```

Access dashboard at:

    http://127.0.0.1:5173

***

# 🧩 **How the Sliding Window Pipeline Works**

### Frontend:

*   MediaRecorder captures **1-second WebM chunks**
*   Worker assembles **4-second overlapping windows**
*   Worker extracts:
    *   **audio:** stitched WebM (audio track preserved)
    *   **video:** **single JPEG frame** from live camera
*   Payload sent every **2 seconds** via WebSocket

### Backend:

1.  Validate WS chunk
2.  Decode:
    *   PyAV audio → Whisper text
    *   JPEG frame → MediaPipe (Face/Pose/Hands)
3.  Run 5 encoders in parallel
4.  Fuse via cultural weights
5.  Apply NMS → event-level dedup
6.  Stream back:
    *   matrix\_update
    *   attention\_tick
    *   fusion\_result

***

# 🧪 **API Reference**

### **1. Health**

    GET /api/v1/health

### **2. Cultural Profiles**

    GET /api/v1/cultures
    GET /api/v1/cultures/{id}

### **3. Session Lifecycle**

    POST   /api/v1/inference/session
    POST   /api/v1/inference/session/{id}/summarize
    DELETE /api/v1/inference/session/{id}

### **4. WebSocket Stream**

    ws://127.0.0.1:8000/api/v1/stream/{session_id}

***

# 📊 **The Glass‑Box Dashboard**

The dashboard displays:

*   **Speech channel**
*   **Facial expressions**
*   **Body/gesture signal**
*   **Live attention weights**
*   **Real-time cultural shifts**
*   **Session history (deduplicated)**
*   **Live latency**
*   **Session summary modal**

This gives a **real-time peek into H‑CMAT’s internal workings**.

***

# 📚 **Session Summary Report**

Generated at end of session:

*   Dominant pragmatic intent
*   Affective state
*   Confidence score
*   Cultural modifiers applied
*   Deduped ledger
*   Cross-modal breakdown

***

# 🧩 **Cultural Profiles (JSON-driven)**

Profiles live in:

    hcmat-backend/data/cultures/*.json

Each profile defines:

```json
{
  "id": 1,
  "code": "IN",
  "name": "Indian (High-Context)",
  "weights": { "speech": 0.15, "face": 0.55, "body": 0.30 },
  "politeness_bias": 0.85,
  "indirect_threshold": 0.70
}
```

Add new cultures by dropping a new `.json` — no code changes required.

***

# 🛑 **Troubleshooting**

### Video decoding fails (`Invalid data found`)

Cause:

*   Rolling WebM chunks are not frame-perfect.

Fix:

*   Frontend now sends JPEG frame + WebM audio separately.

***

### Incorrect padding errors

Cause:

*   Frontend sending raw base64.

Fix:

*   Worker sends full Data URL
*   Backend aggressively cleans base64

***

### WebSocket closes early

Fix:

*   Final clip must be awaited before summarize
*   Race conditions resolved in latest version

***

# 🗺 **Roadmap**

*   [ ] Add autonomous cultural inference
*   [ ] Multi-head learned fusion layer (true transformer)
*   [ ] Prosody extraction (pitch / rate / intensity)
*   [ ] Temporal gesture recognition via full video sequence
*   [ ] Multi-user cultural modeling

***

# ©️ **License**

MIT or custom — your choice.

***

# 🎤 **Author**

Suryaansh Awasthi  
H-CMAT Creator & Engineer