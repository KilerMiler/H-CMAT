"""
api/routes/session.py

REST endpoints for session lifecycle and health check.

Endpoints:
    POST   /api/v1/inference/session                → init
    POST   /api/v1/inference/session/{id}/summarize → finale (before teardown)
    DELETE /api/v1/inference/session/{id}           → cleanup
    GET    /api/v1/health                           → pre-demo check
"""

from __future__ import annotations

import torch
from fastapi import APIRouter, HTTPException, Request

from config.logging import get_logger
from config.settings import settings
from core.schemas import (
    EncoderStatus,
    HealthResponse,
    SessionInitRequest,
    SessionInitResponse,
    SessionSummaryResponse,
    SessionTerminateResponse,
)
from core.summarizer import flush_to_sqlite, summarise

logger = get_logger(__name__)
router = APIRouter()


# ── Health check ──────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request) -> HealthResponse:
    """
    Pre-demo verification endpoint.
    Confirms all 5 encoders are loaded, hardware is correct, and
    reports how many active sessions exist.

    Hit this in Postman before walking on stage.
    A single 'error' in the encoders list means something didn't load —
    check the terminal for the stack trace before presenting.
    """
    system = request.app.state.system

    runner = system.get("runner")
    statuses = runner.encoder_status if runner else [
        {"name": n, "status": "unloaded"}
        for n in ["Text", "Audio", "Face", "Gesture", "Sign"]
    ]

    encoder_list = [
        EncoderStatus(name=s["name"], status=s["status"])
        for s in statuses
    ]

    all_ok = all(e.status == "loaded" for e in encoder_list)

    return HealthResponse(
        status="ok" if all_ok else "degraded",
        device=settings.device_name,
        dtype=str(settings.dtype),
        encoders=encoder_list,
        active_sessions=system["session_manager"].active_count,
        mps_available=torch.backends.mps.is_available(),
    )


# ── Session lifecycle ─────────────────────────────────────────────────────

@router.post(
    "/inference/session",
    response_model=SessionInitResponse,
    status_code=201,
    tags=["Session"],
)
async def create_session(
    body: SessionInitRequest,
    request: Request,
) -> SessionInitResponse:
    """
    Initialises a new H-CMAT inference session.

    Called by React the moment the user clicks "Start", before the
    camera turns on.

    Returns a session_id that must be stored in the Zustand store and
    included in every subsequent WebSocket message and REST call.

    IMPORTANT:
    culture_id validation is STRICT here.
    If the client sends an unknown culture_id, we return 404 instead of
    silently falling back to a default profile. This preserves explainability
    and prevents judges/users from believing one culture is active while
    another is actually being used.
    """
    session_manager = request.app.state.system["session_manager"]
    mapper = request.app.state.system["mapper"]

    # Strict validation for public API creation path
    profiles = {p.id: p for p in mapper.list_all()}
    if body.culture_id not in profiles:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Cultural profile {body.culture_id} not found. "
                f"Available IDs: {sorted(profiles.keys())}"
            ),
        )

    profile = profiles[body.culture_id]

    session = session_manager.create(
        user_id=body.user_id,
        culture_id=profile.id,
        modalities=body.modalities,
    )

    logger.info(
        f"Session created: {session.session_id} "
        f"user={body.user_id} culture={profile.code}"
    )

    return SessionInitResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        status="active",
        culture_id=profile.id,
    )


@router.post(
    "/inference/session/{session_id}/summarize",
    response_model=SessionSummaryResponse,
    tags=["Session"],
)
async def summarize_session(
    session_id: str,
    request: Request,
) -> SessionSummaryResponse:
    """
    Generates the holistic session summary from the NMS-deduplicated ledger.

    MUST be called BEFORE DELETE /session/{id}.
    React shows a loading state while this resolves, then animates the
    closing modal with the returned holistic_summary paragraph.

    The summary is also cached on the session object so that the DELETE
    endpoint can flush it to SQLite without re-computing.
    """
    session_manager = request.app.state.system["session_manager"]
    session = session_manager.get(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or already terminated.",
        )

    summary = summarise(session)

    # Cache on the session so DELETE can persist it without re-computing
    session._cached_summary = summary

    logger.info(
        f"Session summarised: {session_id} "
        f"turns={summary.turn_count} intent='{summary.dominant_intent}'"
    )

    return summary


@router.delete(
    "/inference/session/{session_id}",
    response_model=SessionTerminateResponse,
    tags=["Session"],
)
async def terminate_session(
    session_id: str,
    request: Request,
) -> SessionTerminateResponse:
    """
    Tears down the session:
      1. Retrieves the cached summary (from /summarize call above).
      2. Flushes it to SQLite.
      3. Removes the session from the in-memory store.

    Call this ONLY after the closing modal has rendered.
    The session context is permanently gone after this returns.
    """
    system = request.app.state.system
    session_manager = system["session_manager"]
    session = session_manager.get(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or already terminated.",
        )

    # Flush to SQLite if a summary was generated
    log_saved = False
    cached_summary = getattr(session, "_cached_summary", None)
    if cached_summary:
        log_saved = flush_to_sqlite(session, cached_summary)
    else:
        logger.warning(
            f"[{session_id}] DELETE called without prior /summarize. "
            "Session log will NOT be saved to SQLite."
        )

    terminated = session_manager.terminate(session_id)

    return SessionTerminateResponse(
        session_id=session_id,
        status="terminated" if terminated else "not_found",
        log_saved=log_saved,
    )