"""
core/nms_ledger.py

Window-Level Non-Maximum Suppression for the H-CMAT session ledger.

This removes duplicate behavioral events caused by overlapping sliding windows.

Frontend rule:
    is_new_event=True
        → push a new row.

    is_new_event=False and replaces_seq_id is not None
        → replace/update the row with that seq_id.

    is_new_event=False and replaces_seq_id is None
        → duplicate was suppressed; no UI row update needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.logging import get_logger
from config.settings import settings

if TYPE_CHECKING:
    from core.session_manager import SessionState
    from core.schemas import FuseResponse

logger = get_logger(__name__)


def _make_ledger_entry(response: "FuseResponse") -> dict:
    """Converts a FuseResponse into a flat dict for the session ledger."""
    return {
        "seq_id": response.seq_id,
        "clip_start_ms": response.temporal_context.clip_start_ms,
        "clip_end_ms": response.temporal_context.clip_end_ms,
        "primary_intent": response.holistic_fusion.primary_intent,
        "affective_state": response.holistic_fusion.affective_state,
        "confidence": response.holistic_fusion.confidence,
        "modality_matrix": {
            k: v.model_dump() for k, v in response.modality_matrix.items()
        },
    }


def _clips_overlap(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> bool:
    """
    Returns True if two clip windows share at least
    settings.nms_overlap_threshold_ms milliseconds.
    """
    overlap_ms = min(end_a, end_b) - max(start_a, start_b)
    return overlap_ms >= settings.nms_overlap_threshold_ms


def apply_nms(
    session: "SessionState",
    response: "FuseResponse",
) -> bool:
    """
    Applies Window-Level NMS to the session ledger.

    Mutates:
        - session.ledger
        - response.holistic_fusion.is_new_event
        - response.holistic_fusion.replaces_seq_id

    Returns:
        True  → new event appended
        False → duplicate suppressed or existing event replaced
    """
    new_start = response.temporal_context.clip_start_ms
    new_end = response.temporal_context.clip_end_ms
    new_intent = response.holistic_fusion.primary_intent
    new_conf = response.holistic_fusion.confidence

    response.holistic_fusion.replaces_seq_id = None

    LOOKBACK = 3
    ledger_window = session.ledger[-LOOKBACK:] if session.ledger else []

    for idx, entry in enumerate(reversed(ledger_window)):
        real_idx = len(session.ledger) - 1 - idx

        entry_start = entry["clip_start_ms"]
        entry_end = entry["clip_end_ms"]
        entry_intent = entry["primary_intent"]
        entry_conf = entry["confidence"]

        if not _clips_overlap(entry_start, entry_end, new_start, new_end):
            continue

        if entry_intent != new_intent:
            logger.debug(
                f"[NMS] Overlap but intent changed: "
                f"'{entry_intent}' → '{new_intent}'. Treating as new event."
            )
            break

        if new_conf > entry_conf:
            old_seq_id = entry["seq_id"]

            logger.debug(
                f"[NMS] Replacing seq={old_seq_id}: "
                f"{entry_intent} conf {entry_conf:.3f} → {new_conf:.3f}"
            )

            session.ledger[real_idx] = _make_ledger_entry(response)
            response.holistic_fusion.is_new_event = False
            response.holistic_fusion.replaces_seq_id = old_seq_id
            return False

        logger.debug(
            f"[NMS] Suppressing lower-confidence duplicate: "
            f"{new_intent} conf {new_conf:.3f} <= {entry_conf:.3f}"
        )
        response.holistic_fusion.is_new_event = False
        response.holistic_fusion.replaces_seq_id = None
        return False

    session.ledger.append(_make_ledger_entry(response))
    response.holistic_fusion.is_new_event = True
    response.holistic_fusion.replaces_seq_id = None

    logger.debug(
        f"[NMS] New event appended: {new_intent} "
        f"(conf={new_conf:.3f}, ledger size={len(session.ledger)})"
    )
    return True