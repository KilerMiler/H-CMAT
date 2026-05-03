"""
core/summarizer.py

Two responsibilities:
  1. Synthesise a holistic natural-language summary from the NMS-deduplicated
     session ledger (called by POST /session/{id}/summarize).
  2. Flush the completed session to SQLite for archival
     (called by DELETE /session/{id} — AFTER the summary has been returned).

SQLite write rule (from our architectural decision):
    SQLite is written to EXACTLY ONCE — at session end.
    During the live session everything lives in RAM.
    This prevents file-lock conflicts between the WebSocket writer
    and the REST endpoint writers.
"""

from __future__ import annotations

import json
import sqlite3
import time
from statistics import mean
from typing import TYPE_CHECKING

from config.logging import get_logger
from config.settings import settings
from core.schemas import LedgerEntry, ModalityDetail, SessionSummaryResponse

if TYPE_CHECKING:
    from core.session_manager import SessionState

logger = get_logger(__name__)


# ── Intent → human-readable affect description ─────────────────────────────
# Extend this dict as H-CMAT recognises more pragmatic states.
_INTENT_AFFECT_MAP: dict[str, str] = {
    "POLITE REFUSAL":       "cognitive dissonance between verbal compliance and non-verbal resistance",
    "GENUINE AGREEMENT":    "alignment across all modalities — speech, face, and body congruent",
    "SURFACE AGREEMENT":    "verbal acceptance with subdued facial and postural cues suggesting reservation",
    "INFORMATION SEEKING":  "neutral inquiry affect — open body language, rising intonation",
    "COGNITIVE DISSONANCE": "conflicting signals across modalities indicating internal conflict",
    "DISENGAGEMENT":        "declining attention weights on face and body channels",
    "UNKNOWN":              "insufficient multimodal signal to determine pragmatic state",
}


def _build_summary_text(
    dominant_intent: str,
    dominant_affect: str,
    session_confidence: float,
    turn_count: int,
    duration_ms: int,
) -> str:
    """
    Builds the holistic summary paragraph shown in the closing modal.
    For production, this would be a call to an LLM given the full ledger.
    For the demo, it's a high-quality template — still impressive on screen.
    """
    affect_desc = _INTENT_AFFECT_MAP.get(
        dominant_intent.upper(),
        "a complex pragmatic state requiring further analysis",
    )
    duration_s = duration_ms / 1000
    conf_pct = round(session_confidence * 100, 1)

    return (
        f"Across {turn_count} analysed turns ({duration_s:.1f}s), the H-CMAT system detected "
        f"{affect_desc}. "
        f"The dominant pragmatic state throughout the interaction was classified as "
        f"'{dominant_intent}' with a session-level confidence of {conf_pct}%. "
        f"The dominant affective signal was '{dominant_affect}', sustained across the "
        f"majority of the interaction window. "
        f"Cultural context modifiers were applied throughout, adjusting modality trust "
        f"weights in real time based on the selected cultural profile."
    )


def summarise(session: "SessionState") -> SessionSummaryResponse:
    """
    Generates the holistic session summary from the NMS-deduplicated ledger.
    Called by POST /api/v1/inference/session/{id}/summarize BEFORE teardown.
    """
    session.status = "summarizing"
    ledger = session.ledger

    if not ledger:
        logger.warning(f"[{session.session_id}] Summarising empty ledger.")
        return SessionSummaryResponse(
            session_id=session.session_id,
            duration_ms=0,
            turn_count=0,
            holistic_summary="No multimodal data was captured during this session.",
            dominant_intent="UNKNOWN",
            dominant_affect="UNKNOWN",
            session_confidence=0.0,
            ledger=[],
        )

    # ── Compute aggregate stats ─────────────────────────────────────────
    confidences     = [e["confidence"] for e in ledger]
    session_conf    = round(mean(confidences), 4)

    # Most frequent intent wins
    intent_counts: dict[str, int] = {}
    for e in ledger:
        intent_counts[e["primary_intent"]] = intent_counts.get(e["primary_intent"], 0) + 1
    dominant_intent = max(intent_counts, key=intent_counts.get)

    affect_counts: dict[str, int] = {}
    for e in ledger:
        affect_counts[e["affective_state"]] = affect_counts.get(e["affective_state"], 0) + 1
    dominant_affect = max(affect_counts, key=affect_counts.get)

    duration_ms = (
        ledger[-1]["clip_end_ms"] - ledger[0]["clip_start_ms"]
        if len(ledger) > 1
        else ledger[0]["clip_end_ms"] - ledger[0]["clip_start_ms"]
    )

    summary_text = _build_summary_text(
        dominant_intent, dominant_affect, session_conf,
        turn_count=len(ledger), duration_ms=duration_ms,
    )

    # ── Build typed LedgerEntry list for the response ────────────────
    typed_ledger = []
    for e in ledger:
        typed_ledger.append(LedgerEntry(
            seq_id=e["seq_id"],
            clip_start_ms=e["clip_start_ms"],
            clip_end_ms=e["clip_end_ms"],
            primary_intent=e["primary_intent"],
            affective_state=e["affective_state"],
            confidence=e["confidence"],
            modality_matrix={
                k: ModalityDetail(**v)
                for k, v in e["modality_matrix"].items()
            },
        ))

    logger.info(
        f"[{session.session_id}] Summary: '{dominant_intent}' "
        f"conf={session_conf} turns={len(ledger)}"
    )

    return SessionSummaryResponse(
        session_id=session.session_id,
        duration_ms=duration_ms,
        turn_count=len(ledger),
        holistic_summary=summary_text,
        dominant_intent=dominant_intent,
        dominant_affect=dominant_affect,
        session_confidence=session_conf,
        ledger=typed_ledger,
    )


# ── SQLite persistence ─────────────────────────────────────────────────────

def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Creates the sessions table if it doesn't exist yet."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id       TEXT PRIMARY KEY,
            user_id          TEXT,
            culture_id       INTEGER,
            created_at       INTEGER,
            ended_at         INTEGER,
            duration_ms      INTEGER,
            turn_count       INTEGER,
            dominant_intent  TEXT,
            dominant_affect  TEXT,
            session_conf     REAL,
            holistic_summary TEXT,
            ledger_json      TEXT
        )
    """)
    conn.commit()


def flush_to_sqlite(
    session: "SessionState",
    summary: SessionSummaryResponse,
) -> bool:
    """
    Writes the completed session to SQLite.
    Called ONLY by DELETE /session/{id} — after the summary has been returned.

    Returns True on success, False on any SQLite error.
    The calling route should still return 200 even if this fails
    (don't let a logging error break the demo teardown).
    """
    try:
        db_path = settings.sqlite_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path))
        _ensure_schema(conn)

        ledger_json = json.dumps([e.model_dump() for e in summary.ledger])
        ended_at    = int(time.time() * 1000)

        conn.execute("""
            INSERT OR REPLACE INTO sessions
            (session_id, user_id, culture_id, created_at, ended_at,
             duration_ms, turn_count, dominant_intent, dominant_affect,
             session_conf, holistic_summary, ledger_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.session_id,
            session.user_id,
            session.culture_id,
            session.created_at,
            ended_at,
            summary.duration_ms,
            summary.turn_count,
            summary.dominant_intent,
            summary.dominant_affect,
            summary.session_confidence,
            summary.holistic_summary,
            ledger_json,
        ))
        conn.commit()
        conn.close()

        logger.info(f"[{session.session_id}] Flushed to SQLite at {db_path}")
        return True

    except Exception as exc:
        logger.error(f"[{session.session_id}] SQLite flush failed: {exc}")
        return False