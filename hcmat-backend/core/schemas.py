"""
core/schemas.py

Single source of truth for ALL Pydantic models used across the project.
Replaces the old api_requests.py + api_responses.py.

Layout:
  ── Inbound (React → FastAPI) ──────────────────────────────────────────
  TemporalContext       timestamp anchors on every clip
  FuseRequest           the WebSocket 'chunk' payload
  SessionInitRequest    POST /api/v1/inference/session body
  SessionSummarizeRequest  (no body needed — session_id in URL)

  ── Outbound (FastAPI → React) ─────────────────────────────────────────
  ModalityDetail        one row in the Glass Box matrix
  HolisticFusion        the final fused interpretation for one clip
  FuseResponse          full WebSocket 'fusion_result' payload
  MatrixUpdateResponse  WebSocket 'matrix_update' payload (early, pre-fusion)
  AttentionTickResponse WebSocket 'attention_tick' payload (Glass Box stream)
  SessionInitResponse   response to POST /session
  SessionSummaryResponse response to POST /session/{id}/summarize
  HealthResponse        response to GET /health
  CultureProfile        response item for GET /cultures and GET /cultures/{id}

  ── WebSocket message envelope ─────────────────────────────────────────
  WSMessage             typed wrapper — every WS frame uses this
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class WSMessageType(str, enum.Enum):
    """
    All valid WebSocket message types in both directions.
    Having these as an enum means a typo in a type string fails
    at parse time, not silently at runtime.
    """
    # Frontend → Backend
    CHUNK      = "chunk"
    HEARTBEAT  = "heartbeat"

    # Backend → Frontend
    MATRIX_UPDATE  = "matrix_update"
    FUSION_RESULT  = "fusion_result"
    ATTENTION_TICK = "attention_tick"
    ERROR          = "error"


# ═══════════════════════════════════════════════════════════════════════
# INBOUND — React → FastAPI
# ═══════════════════════════════════════════════════════════════════════

class TemporalContext(BaseModel):
    """
    Timestamp anchors that React owns and populates.
    FastAPI uses these for Window-Level NMS deduplication.

    clip_start_ms / clip_end_ms:
        Milliseconds relative to session start (not wall clock).
        Derived from chunk arrival timestamps, not from chunk count,
        so they stay accurate even if the MediaRecorder fires slightly
        late (browser timeslice hint is not guaranteed to be exact).

    is_final_clip:
        React sets this True on the last chunk before the user
        clicks Stop. Tells the NMS ledger to flush and prepare
        for /summarize.
    """
    clip_start_ms: int = Field(..., ge=0)
    clip_end_ms:   int = Field(..., gt=0)
    is_final_clip: bool = False


class FuseRequest(BaseModel):
    """
    The core data payload sent over WebSocket every 2 seconds.
    Corresponds to the 'chunk' WebSocket message type.

    Notes:
    - current_text is intentionally OMITTED. Whisper on FastAPI
      derives the transcript from audio_data_base64 natively.
    - video_data_base64 carries the stitched 4-second WebM blob
      (headerChunk prepended by the Web Worker).
    - text_history is the rolling last-N-turns context window.
      React maintains and trims this array.
    """
    session_id:       str
    seq_id:           int = Field(..., ge=0, description="Monotonically increasing. Used to reorder out-of-sequence chunks.")
    temporal_context: TemporalContext
    culture_id:       int = Field(default=1, ge=1, description="Cultural profile ID from GET /cultures")
    text_history:     List[str] = Field(default_factory=list, max_length=5)
    audio_data_base64: Optional[str] = None
    video_data_base64: Optional[str] = None


class SessionInitRequest(BaseModel):
    """Body for POST /api/v1/inference/session."""
    user_id:    str = Field(default="demo_user")
    culture_id: int = Field(default=1, ge=1)
    modalities: List[str] = Field(default=["speech", "face", "body"])
    config: Dict[str, Any] = Field(default_factory=dict)


class HeartbeatPayload(BaseModel):
    """Payload inside a WebSocket 'heartbeat' message."""
    session_id: str
    ts: int   # wall-clock ms from React (Date.now())


# ═══════════════════════════════════════════════════════════════════════
# OUTBOUND — FastAPI → React
# ═══════════════════════════════════════════════════════════════════════

class ModalityDetail(BaseModel):
    """
    One row in the Glass Box modality matrix.
    Shown live in the React UI as an animated progress bar.
    """
    feature:   str   = Field(..., description="Human-readable signal extracted (e.g. 'AU12 / Micro-Tightening')")
    weight:    float = Field(..., ge=0.0, le=1.0, description="Attention weight for this modality (sums to 1.0 across all)")
    local_tag: str   = Field(..., description="Interpretive label (e.g. 'SOCIAL COMPLIANCE MASK')")


class HolisticFusion(BaseModel):
    """
    The final fused interpretation for one clip.
    This is what gets appended to, suppresses, or replaces one item in the
    Session History log.
    """
    primary_intent: str
    affective_state: str
    confidence: float = Field(..., ge=0.0, le=1.0)

    # Set by nms_ledger.py AFTER comparing against the temporal ledger.
    # True  → React pushes a NEW row to Session History.
    # False → React updates/replaces an existing row or suppresses duplicate.
    # None  → not yet determined.
    is_new_event: Optional[bool] = None

    # If NMS overwrites an older event, this tells React exactly which
    # seq_id should be replaced. If None and is_new_event=False, React can
    # safely treat it as a suppressed duplicate.
    replaces_seq_id: Optional[int] = None


class FuseResponse(BaseModel):
    """
    Full payload for the 'fusion_result' WebSocket message.
    Delivered after the Fusion Layer has processed a clip.
    """
    seq_id:           int
    temporal_context: TemporalContext
    modality_matrix:  Dict[str, ModalityDetail]
    holistic_fusion:  HolisticFusion
    fusion_latency_ms: int
    total_latency_ms:  int


class MatrixUpdateResponse(BaseModel):
    """
    Payload for the 'matrix_update' WebSocket message.
    Pushed IMMEDIATELY after encoders finish — before full fusion.
    This is what makes the Glass Box bars animate in real time.
    """
    seq_id:          int
    modality_matrix: Dict[str, ModalityDetail]


class AttentionTickResponse(BaseModel):
    """
    Payload for the 'attention_tick' WebSocket message.
    Streams cross-modal attention weights continuously during inference.
    Powers the live 'Glass Box' transparency feature.
    """
    ts:      int              # server-side wall clock ms
    weights: Dict[str, float] # e.g. {"speech_to_face": 0.72, "face_to_body": 0.45}


class WSErrorResponse(BaseModel):
    """Payload for the 'error' WebSocket message."""
    code:    str
    message: str
    seq_id:  Optional[int] = None


# ═══════════════════════════════════════════════════════════════════════
# WEBSOCKET ENVELOPE
# ═══════════════════════════════════════════════════════════════════════

class WSMessage(BaseModel):
    """
    Every WebSocket frame — in both directions — is wrapped in this envelope.
    React's onmessage handler switches on `type` to route to the correct
    Zustand store updater.

    Without typed messages, the handler becomes a fragile if-else chain
    and backend errors are indistinguishable from inference results.
    """
    type:    WSMessageType
    payload: Any


# ═══════════════════════════════════════════════════════════════════════
# REST ENDPOINT RESPONSES
# ═══════════════════════════════════════════════════════════════════════

class EncoderStatus(BaseModel):
    name:   str
    status: str   # "loaded" | "error" | "unloaded"


class HealthResponse(BaseModel):
    """Response for GET /api/v1/health."""
    status:          str   # "ok" | "degraded" | "error"
    device:          str
    dtype:           str
    encoders:        List[EncoderStatus]
    active_sessions: int
    mps_available:   bool


class SessionInitResponse(BaseModel):
    """Response for POST /api/v1/inference/session."""
    session_id:  str
    created_at:  int   # Unix ms
    status:      str   # "active"
    culture_id:  int


class SessionTerminateResponse(BaseModel):
    """Response for DELETE /api/v1/inference/session/{id}."""
    session_id: str
    status:     str   # "terminated"
    log_saved:  bool


class LedgerEntry(BaseModel):
    """One deduplicated event in the session macro-ledger."""
    seq_id:          int
    clip_start_ms:   int
    clip_end_ms:     int
    primary_intent:  str
    affective_state: str
    confidence:      float
    modality_matrix: Dict[str, ModalityDetail]


class SessionSummaryResponse(BaseModel):
    """Response for POST /api/v1/inference/session/{id}/summarize."""
    session_id:         str
    duration_ms:        int
    turn_count:         int
    holistic_summary:   str
    dominant_intent:    str
    dominant_affect:    str
    session_confidence: float
    ledger:             List[LedgerEntry]   # full deduplicated history


class CultureWeights(BaseModel):
    speech: float
    face:   float
    body:   float


class CultureProfile(BaseModel):
    """
    Response item for GET /api/v1/cultures and GET /api/v1/cultures/{id}.
    Loaded from data/cultures/*.json at startup.
    """
    id:                  int
    code:                str
    name:                str
    weights:             CultureWeights
    politeness_bias:     float = Field(..., ge=0.0, le=1.0)
    indirect_threshold:  float = Field(..., ge=0.0, le=1.0)