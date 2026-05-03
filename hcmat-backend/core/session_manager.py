"""
core/session_manager.py

In-memory session store.

Responsibilities:
  - Create and terminate sessions
  - Store the rolling text_history context window per session
  - Track session metadata (start time, culture, status)
  - Provide a clean interface for the WebSocket handler and REST routes

Intentionally NOT responsible for:
  - NMS deduplication (that's nms_ledger.py)
  - SQLite persistence (that happens at session end in summarizer.py)
  - AI inference (that's parallel_runner.py)

Threading note:
  The session dict is accessed from both the REST routes (asyncio event loop)
  and the WebSocket handler. Since Python dicts are GIL-protected for
  individual operations, and we never iterate + mutate concurrently,
  this is safe without an asyncio.Lock for the demo scope.
  A production system should add a lock for multi-worker deployments.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)


@dataclass
class SessionState:
    """
    All mutable state for one active session.
    Lives entirely in RAM. Flushed to SQLite only when the session ends.
    """
    session_id:   str
    user_id:      str
    culture_id:   int
    modalities:   List[str]
    created_at:   int          # Unix ms

    # The sliding context window React populates; we trim it here too
    # so both sides stay in sync even if React sends stale history.
    text_history: List[str] = field(default_factory=list)

    # Sequence counter — used to detect and discard stale out-of-order frames
    last_seq_id:  int = -1

# Set to "active" | "summarize_ready" | "summarizing" | "terminated"    
    status: str = "active"

    # Will be populated by nms_ledger.py as clips arrive
    # (imported lazily to avoid circular import)
    ledger: List[dict] = field(default_factory=list)


class SessionManager:
    """
    Thread-safe (GIL-level) in-memory store for all active sessions.
    One instance is created at server startup and lives for the server lifetime.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}
        logger.info("SessionManager initialised (in-memory store).")

    # ── CRUD ──────────────────────────────────────────────────────────

    def create(
        self,
        user_id:    str,
        culture_id: int,
        modalities: List[str],
    ) -> SessionState:
        """
        Creates a new session and returns it.
        Called by POST /api/v1/inference/session.
        """
        session_id = str(uuid.uuid4())
        session = SessionState(
            session_id=session_id,
            user_id=user_id,
            culture_id=culture_id,
            modalities=modalities,
            created_at=int(time.time() * 1000),
        )
        self._sessions[session_id] = session
        logger.info(f"Session created: {session_id} (culture={culture_id})")
        return session

    def get(self, session_id: str) -> Optional[SessionState]:
        """Returns the session or None if not found / already terminated."""
        return self._sessions.get(session_id)

    def require(self, session_id: str) -> SessionState:
        """
        Returns the session or raises KeyError.
        Use this inside WebSocket handlers where a missing session
        should immediately close the connection.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found or already terminated.")
        return session

    def terminate(self, session_id: str) -> bool:
        """
        Marks session as terminated and removes it from the active store.
        Returns True if the session existed, False otherwise.
        Called by DELETE /api/v1/inference/session/{id}.
        """
        session = self._sessions.pop(session_id, None)
        if session is None:
            logger.warning(f"Attempted to terminate unknown session: {session_id}")
            return False
        session.status = "terminated"
        logger.info(f"Session terminated: {session_id}")
        return True

    # ── Context window helpers ─────────────────────────────────────────

    def update_text_history(self, session_id: str, new_history: List[str]) -> None:
        """
        Syncs the text_history from the incoming FuseRequest.
        Trims to max_history_turns to prevent unbounded memory growth.
        """
        session = self.require(session_id)
        trimmed = new_history[-settings.max_history_turns:]
        session.text_history = trimmed

    # ── Sequence guard ─────────────────────────────────────────────────

    def is_seq_valid(self, session_id: str, seq_id: int) -> bool:
        """
        Returns False if seq_id is less than or equal to the last seen seq_id,
        meaning this chunk arrived out of order and should be discarded.
        """
        session = self.require(session_id)
        if seq_id <= session.last_seq_id:
            logger.warning(
                f"[{session_id}] Discarding stale chunk: "
                f"seq_id={seq_id} <= last_seen={session.last_seq_id}"
            )
            return False
        session.last_seq_id = seq_id
        return True

    # ── Metadata ───────────────────────────────────────────────────────

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def all_session_ids(self) -> List[str]:
        return list(self._sessions.keys())