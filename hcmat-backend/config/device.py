"""
config/device.py

Single source of truth for hardware detection.
Every encoder imports from here — never duplicates MPS logic.

Rule: if you need to force CPU for debugging, change ONE line here.
"""

import torch


def get_device() -> torch.device:
    """
    Returns the best available compute device.

    Priority order:
      1. Apple Silicon MPS  — GPU cores + Neural Engine (M1/M2/M3/M4)
      2. CUDA               — NVIDIA GPU (if somehow running on a different machine)
      3. CPU                — always available fallback

    NOTE: MediaPipe encoders (Face, Gesture, Sign) ignore this entirely.
    They run their own Apple Silicon optimised runtime. Never move MediaPipe
    tensors to MPS manually — it will crash.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """
    Returns the optimal floating-point precision for the detected device.

    MPS (Apple Silicon):
        float16 (FP16) — halves memory footprint with no CPU fallback.
        INT8 quantisation is designed for edge MCUs (Raspberry Pi etc.),
        not for the unified memory architecture of Apple Silicon.

    CUDA / CPU:
        float32 — safest default across vendors.
    """
    if device.type == "mps":
        return torch.float16
    return torch.float32


# ── Module-level singletons ────────────────────────────────────────────────
# Computed once at import time; re-used everywhere via:
#   from config.device import DEVICE, DTYPE

DEVICE: torch.device = get_device()
DTYPE: torch.dtype = get_dtype(DEVICE)