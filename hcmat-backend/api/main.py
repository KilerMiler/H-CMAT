"""
api/main.py

The FastAPI application entry point.

Responsibilities:
  1. Lifespan manager — loads all heavy AI models ONCE at startup,
     stores them on app.state.system so every route and WS handler
     can access them without re-loading.
  2. CORS middleware — allows the React dev server to call the API.
  3. Router registration — mounts all REST and WebSocket routes under
     the /api/v1 prefix.
  4. ThreadPoolExecutor — shared executor that WebSocket handlers use
     via asyncio.run_in_executor() to run blocking inference without
     freezing the event loop.

Run the server:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

For the demo (no reload, maximum stability):
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1

Why --workers 1?
    The AI models are loaded into a single process's memory.
    Multiple Uvicorn workers = multiple processes = each loads its own
    copy of all 5 models. On 16GB unified memory that will OOM.
    Always use --workers 1 for the demo.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.logging import get_logger
from config.settings import settings
from core.cultural_mapper import CulturalMapper
from core.fusion import FusionLayer
from core.session_manager import SessionManager
from inference.parallel_runner import ParallelEncoderRunner
from api.routes import session as session_router
from api.routes import cultures as cultures_router
from api.routes import stream as stream_router

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# LIFESPAN — Model loading and teardown
# ═══════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager.

    Everything BEFORE yield runs at startup (before the first request).
    Everything AFTER yield runs at shutdown (after the last request).

    This is where all heavy models are loaded into unified memory.
    Server startup will take 10-15 seconds. Every subsequent inference
    call will be sub-100ms because models are already in RAM.

    All loaded objects are stored on app.state.system so routes and
    WebSocket handlers can access them via request.app.state.system
    without global variables.
    """
    settings.print_startup_banner()

    # ── 1. Session manager (in-memory, no I/O cost) ──────────────────
    logger.info("Initialising SessionManager...")
    session_manager = SessionManager()

    # ── 2. Cultural profiles (reads JSON files from disk) ────────────
    logger.info("Initialising CulturalMapper...")
    mapper = CulturalMapper()

    # ── 3. Thread pool executor (shared across all WS handlers) ──────
    # max_workers = max_parallel_workers (5) × 2 to accommodate concurrent
    # sessions during the demo without queueing.
    logger.info("ParallelEncoderRunner will manage its own internal executor.")

    # ── 4. Parallel encoder runner (loads all 5 AI models) ───────────
    # This is the slow step (10-15s). Models are pulled into unified memory:
    #   - DistilRoBERTa (Text)      ~280MB
    #   - Whisper-Tiny (Audio)       ~39MB
    #   - MediaPipe Face Mesh       ~10MB
    #   - MediaPipe Pose            ~12MB
    #   - MediaPipe Holistic        ~25MB
    logger.info("Loading AI encoders (this takes 10-15 seconds)...")
    runner = ParallelEncoderRunner()

    # ── 5. Fusion layer (lightweight — just AttentionMath + rules) ───
    logger.info("Initialising Fusion Layer...")
    brain = FusionLayer()

    # ── Store everything on app.state ────────────────────────────────
    app.state.system = {
        "session_manager": session_manager,
        "mapper":          mapper,
        "runner":          runner,
        "brain":           brain,
    }

    logger.info("=" * 52)
    logger.info("  H-CMAT SYSTEM READY — all models loaded.")
    logger.info(f"  Listening on http://{settings.host}:{settings.port}")
    logger.info(f"  Cultural profiles: {mapper.profile_count} loaded")
    logger.info("=" * 52)

    # ── Hand control to FastAPI ───────────────────────────────────────
    yield

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Shutting down H-CMAT engine...")
    runner.shutdown()
    app.state.system.clear()
    logger.info("Shutdown complete.")


# ═══════════════════════════════════════════════════════════════════════
# APPLICATION FACTORY
# ═══════════════════════════════════════════════════════════════════════

def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.
    Separated from module-level instantiation so it can be imported
    in tests without side effects.
    """
    application = FastAPI(
        title="H-CMAT Inference API",
        description=(
            "Hierarchical Culturally-Aware Multimodal Attention Transformer — "
            "Real-time pragmatic intent analysis from speech, face, and body."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",       # Swagger UI — useful for demo live walk-through
        redoc_url="/redoc",
    )

    # ── CORS ─────────────────────────────────────────────────────────
    # Allows the React dev server (Vite :5173 or CRA :3000) to call
    # the API and establish WebSocket connections.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── REST routes ──────────────────────────────────────────────────
    # All endpoints under /api/v1 prefix as per the locked API surface.
    application.include_router(
        session_router.router,
        prefix="/api/v1",
    )
    application.include_router(
        cultures_router.router,
        prefix="/api/v1",
    )

    # ── WebSocket route ───────────────────────────────────────────────
    # WebSockets use ws:// not http:// but FastAPI's include_router handles
    # the prefix correctly for both protocols.
    application.include_router(
        stream_router.router,
        prefix="/api/v1",
    )

    # ── Root health ping (lightweight, no system access needed) ──────
    @application.get("/", tags=["Root"])
    async def root():
        return {
            "service": "H-CMAT Inference API",
            "version": "1.0.0",
            "docs":    "/docs",
            "health":  "/api/v1/health",
        }

    return application


# ═══════════════════════════════════════════════════════════════════════
# MODULE-LEVEL APP INSTANCE
# ═══════════════════════════════════════════════════════════════════════

# Uvicorn imports this object:
#   uvicorn api.main:app --port 8000
app = create_app()