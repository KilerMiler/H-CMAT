"""
config/settings.py

Application-wide configuration.
"""

import pathlib
from pydantic_settings import BaseSettings, SettingsConfigDict
from config.device import DEVICE, DTYPE
import torch


class Settings(BaseSettings):
    """
    All tuneable parameters in one place.
    Pydantic reads from environment / .env automatically.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────────────────
    # Localhost-only by default for demo safety.
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False

    # ── CORS ──────────────────────────────────────────────────────────────
    allowed_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    # ── H-CMAT Architecture ───────────────────────────────────────────────
    cultural_vector_dim: int = 256
    max_parallel_workers: int = 5

    window_size_ms: int = 4000
    stride_ms: int = 2000

    max_history_turns: int = 5

    # ── Payload safety limits ─────────────────────────────────────────────
    # Base64 length limits, not decoded byte limits.
    # These prevent accidental huge browser blobs from freezing the demo.
    max_audio_base64_chars: int = 5_000_000
    max_video_base64_chars: int = 20_000_000

    # ── Session / NMS ─────────────────────────────────────────────────────
    nms_overlap_threshold_ms: int = 2000

    # ── Paths ─────────────────────────────────────────────────────────────
    base_dir: pathlib.Path = pathlib.Path(__file__).parent.parent
    cultures_dir: pathlib.Path = base_dir / "data" / "cultures"
    sqlite_path: pathlib.Path = base_dir / "data" / "sessions.db"

    @property
    def device(self) -> torch.device:
        return DEVICE

    @property
    def dtype(self) -> torch.dtype:
        return DTYPE

    @property
    def device_name(self) -> str:
        names = {"mps": "Apple Silicon (MPS)", "cuda": "NVIDIA CUDA", "cpu": "CPU"}
        return names.get(DEVICE.type, DEVICE.type)

    def print_startup_banner(self) -> None:
        """Prints a clean status banner when the server boots."""
        print("\n" + "=" * 52)
        print("  H-CMAT INFERENCE ENGINE — STARTING UP")
        print("=" * 52)
        print(f"  Host      : {self.host}:{self.port}")
        print(f"  Hardware  : {self.device_name}")
        print(f"  Precision : {self.dtype}")
        print(f"  Workers   : {self.max_parallel_workers} parallel encoders")
        print(f"  Window    : {self.window_size_ms}ms  |  Stride: {self.stride_ms}ms")
        print(f"  Cultures  : {self.cultures_dir}")
        print(f"  SQLite    : {self.sqlite_path}")
        print("=" * 52 + "\n")


settings = Settings()