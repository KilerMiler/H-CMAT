"""
inference/parallel_runner.py

Owns all encoder instances and runs them through one persistent executor.

Responsibilities:
  - Load all 5 encoders once at backend startup.
  - Decode browser WebM payloads received over WebSocket.
  - Run Text, Audio, Face, Gesture, and Hand/Sign encoders concurrently.
  - Return safe fallback outputs if any decode/encoder stage fails.

Important:
  The frontend may send either:
    1. Full Data URL:
         data:video/webm;codecs=vp8,opus;base64,AAAA...
    2. Raw base64:
         AAAA...

  This file robustly handles both.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import re
import cv2
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np

from config.logging import get_logger
from config.settings import settings
from inference.encoders import (
    TextEncoder,
    AudioEncoder,
    FaceEncoder,
    GestureEncoder,
    SignEncoder,
)

logger = get_logger(__name__)

try:
    import av

    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
    logger.error(
        "PyAV is not installed. Run: pip install av\n"
        "Without it, audio/video decoding will use safe fallbacks."
    )


# Keeps only normal base64 characters.
_BASE64_RE = re.compile(r"[^A-Za-z0-9+/=]")


# ═══════════════════════════════════════════════════════════════════════
# BASE64 / DATA URL HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _is_too_large(b64_string: str, max_chars: int) -> bool:
    """
    Checks payload size before decoding.

    max_chars is base64/DataURL character count, not decoded byte count.
    """
    return len(b64_string or "") > max_chars


def _clean_base64(b64_string: str) -> str:
    """
    Robustly cleans browser DataURL/base64 payloads.

    Handles:
      - full data URL prefix:
          data:video/webm;codecs=vp8,opus;base64,AAAA...
      - raw base64:
          AAAA...
      - whitespace/newlines
      - URL-safe base64 chars:
          - becomes +
          _ becomes /
      - malformed/excess padding
      - accidental non-base64 characters

    Returns:
      Clean base64 string with correct padding.
    """
    if not b64_string:
        return ""

    if not isinstance(b64_string, str):
        b64_string = str(b64_string)

    # Strip full Data URL prefix if present.
    # Prefer splitting at "base64," because MIME type may contain commas/params.
    if "base64," in b64_string:
        b64_string = b64_string.split("base64,", 1)[1]
    elif "," in b64_string:
        # Fallback for any data URL-like prefix.
        b64_string = b64_string.split(",", 1)[1]

    # Remove whitespace/newlines.
    b64_string = "".join(b64_string.split())

    # Normalize URL-safe base64 variants.
    b64_string = b64_string.replace("-", "+").replace("_", "/")

    # Remove accidental invalid characters.
    b64_string = _BASE64_RE.sub("", b64_string)

    # Normalize padding.
    # Remove existing padding first, then add mathematically correct padding.
    b64_string = b64_string.rstrip("=")
    padding = (-len(b64_string)) % 4
    b64_string += "=" * padding

    return b64_string


def _safe_b64decode(b64_string: str) -> bytes:
    """
    Decodes base64 after aggressive cleanup.

    validate=False is intentional because browser payloads can contain
    harmless formatting differences. The cleanup above removes invalid chars.
    """
    clean = _clean_base64(b64_string)

    if not clean:
        raise binascii.Error("empty base64 payload")

    return base64.b64decode(clean, validate=False)


# ═══════════════════════════════════════════════════════════════════════
# MEDIA DECODING
# ═══════════════════════════════════════════════════════════════════════

def _decode_webm_to_middle_frame(video_base64: str) -> np.ndarray:
    """
    Decodes visual payload into a BGR frame.

    Supports:
      - data:image/jpeg;base64,...
      - data:image/png;base64,...
      - data:video/webm;base64,...

    Returns blank frame on failure.
    """
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    if not video_base64:
        return blank

    if _is_too_large(video_base64, settings.max_video_base64_chars):
        logger.warning(
            "Video payload too large "
            f"({len(video_base64)} chars). Returning blank frame."
        )
        return blank

    try:
        raw_bytes = _safe_b64decode(video_base64)

        logger.debug(f"[VisualDecoder] decoded bytes={len(raw_bytes)}")

        # First try image decode. This handles JPEG/PNG camera frames.
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if image is not None:
            logger.debug(f"[VisualDecoder] decoded image shape={image.shape}")
            return image

        # If image decode failed, fallback to WebM video decode.
        if not AV_AVAILABLE:
            logger.warning("Video decode skipped because PyAV is unavailable.")
            return blank

        container = av.open(io.BytesIO(raw_bytes))
        frames: List[np.ndarray] = []

        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="bgr24"))

        container.close()

        if not frames:
            logger.warning("PyAV decoded 0 video frames — returning blank frame.")
            return blank

        middle_index = len(frames) // 2
        middle_frame = frames[middle_index]

        logger.debug(
            f"[VideoDecoder] frames={len(frames)}, using index={middle_index}"
        )

        return middle_frame

    except Exception as exc:
        logger.error(f"Visual decode failed: {exc}")
        return blank

def _decode_audio_to_waveform(audio_base64: str) -> Optional[dict[str, Any]]:
    """
    Decodes browser WebM audio into a Whisper-friendly waveform dict.

    Returns:
      {
        "array": np.ndarray(float32),
        "sampling_rate": 16000
      }

    Returns None if decode fails.
    """
    if not audio_base64:
        return None

    if _is_too_large(audio_base64, settings.max_audio_base64_chars):
        logger.warning(
            "Audio payload too large "
            f"({len(audio_base64)} chars). Ignoring audio."
        )
        return None

    if not AV_AVAILABLE:
        logger.warning("Audio decode skipped because PyAV is unavailable.")
        return None

    try:
        raw_bytes = _safe_b64decode(audio_base64)

        logger.debug(f"[AudioDecoder] decoded bytes={len(raw_bytes)}")

        container = av.open(io.BytesIO(raw_bytes))

        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=16000,
        )

        chunks: list[np.ndarray] = []

        for frame in container.decode(audio=0):
            resampled_frames = resampler.resample(frame)

            if resampled_frames is None:
                continue

            if not isinstance(resampled_frames, list):
                resampled_frames = [resampled_frames]

            for resampled in resampled_frames:
                if resampled is None:
                    continue

                arr = resampled.to_ndarray()

                # Usually shape is (channels, samples). We requested mono.
                if arr.ndim == 2:
                    arr = arr[0]

                # Convert int16 PCM to float32 waveform in [-1, 1].
                chunks.append(arr.astype(np.float32) / 32768.0)

        container.close()

        if not chunks:
            logger.warning("PyAV decoded 0 audio frames.")
            return None

        waveform = np.concatenate(chunks).astype(np.float32)

        logger.debug(
            f"[AudioDecoder] samples={len(waveform)}, sampling_rate=16000"
        )

        return {
            "array": waveform,
            "sampling_rate": 16000,
        }

    except Exception as exc:
        logger.error(f"Audio decode failed: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# PARALLEL ENCODER RUNNER
# ═══════════════════════════════════════════════════════════════════════

class ParallelEncoderRunner:
    """
    Loads and owns all encoder instances.

    Design:
      - Encoder objects are created once at startup.
      - One persistent ThreadPoolExecutor is reused for all chunks.
      - Per-encoder locks prevent unsafe concurrent access to shared model
        instances across multiple chunks/sessions.
    """

    def __init__(self) -> None:
        logger.info("─── Booting H-CMAT Encoders ───────────────────────────")

        self.text_encoder = TextEncoder(device=settings.device)
        self.audio_encoder = AudioEncoder(device=settings.device)
        self.face_encoder = FaceEncoder()
        self.gesture_encoder = GestureEncoder()
        self.sign_encoder = SignEncoder()

        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_parallel_workers,
            thread_name_prefix="hcmat_encoder",
        )

        self._text_lock = threading.Lock()
        self._audio_lock = threading.Lock()
        self._face_lock = threading.Lock()
        self._gesture_lock = threading.Lock()
        self._sign_lock = threading.Lock()

        logger.info("─── All 5 Encoders Loaded and Ready ───────────────────")

        if not AV_AVAILABLE:
            logger.error(
                "═══════════════════════════════════════════════\n"
                "  CRITICAL: PyAV not installed!\n"
                "  Audio/video decoding will use fallbacks.\n"
                "  Run:  pip install av\n"
                "  Then restart the server.\n"
                "═══════════════════════════════════════════════"
            )

    @staticmethod
    def _safe_process(lock: threading.Lock, fn, *args, **kwargs) -> Dict[str, Any]:
        """
        Serializes access to a shared encoder instance.
        """
        with lock:
            return fn(*args, **kwargs)

    @staticmethod
    def _error_output(modality: str, feature: str = "error") -> Dict[str, Any]:
        """
        Standard fallback if an encoder crashes.
        Fusion will mostly ignore it because uncertainty=1.0.
        """
        return {
            "modality": modality,
            "feature": feature,
            "uncertainty": 1.0,
            "time_ms": 0,
        }

    async def run_all_async(
        self,
        text_input: str,
        audio_base64: Optional[str] = None,
        video_base64: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs all five encoders asynchronously through the persistent executor.
        """
        start_time = time.time()

        # Decode media payloads once per chunk before sending frames/waveforms
        # into the encoder threads.
        audio_input = (
            _decode_audio_to_waveform(audio_base64)
            if audio_base64
            else None
        )

        image_input = (
            _decode_webm_to_middle_frame(video_base64)
            if video_base64
            else np.zeros((480, 640, 3), dtype=np.uint8)
        )

        safe_text = text_input.strip() if text_input and text_input.strip() else ""

        logger.debug(
            "[ParallelRunner] decoded inputs: "
            f"text_len={len(safe_text)}, "
            f"audio={'yes' if audio_input is not None else 'no'}, "
            f"image_shape={getattr(image_input, 'shape', None)}"
        )

        loop = asyncio.get_running_loop()

        tasks = [
            loop.run_in_executor(
                self._executor,
                self._safe_process,
                self._text_lock,
                self.text_encoder.process,
                safe_text,
            ),
            loop.run_in_executor(
                self._executor,
                self._safe_process,
                self._audio_lock,
                self.audio_encoder.process,
                audio_input,
            ),
            loop.run_in_executor(
                self._executor,
                self._safe_process,
                self._face_lock,
                self.face_encoder.process,
                image_input,
            ),
            loop.run_in_executor(
                self._executor,
                self._safe_process,
                self._gesture_lock,
                self.gesture_encoder.process,
                image_input,
            ),
            loop.run_in_executor(
                self._executor,
                self._safe_process,
                self._sign_lock,
                self.sign_encoder.process,
                image_input,
            ),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        normalized: List[Dict[str, Any]] = []
        modality_order = ["Text", "Audio", "Face", "Gesture", "SignLang"]

        for modality, result in zip(modality_order, results):
            if isinstance(result, Exception):
                logger.error(f"[ParallelRunner] {modality} encoder crashed: {result}")
                normalized.append(self._error_output(modality))
            else:
                normalized.append(result)

        latency_ms = int((time.time() - start_time) * 1000)
        logger.debug(f"[ParallelRunner] 5 modalities in {latency_ms}ms")

        return normalized

    def shutdown(self) -> None:
        """
        Shuts down the runner's persistent executor.
        Called from FastAPI lifespan shutdown.
        """
        logger.info("Shutting down ParallelEncoderRunner executor...")
        self._executor.shutdown(wait=True)
        logger.info("ParallelEncoderRunner executor shutdown complete.")

    @property
    def encoder_status(self) -> List[Dict[str, str]]:
        return [
            {"name": "Text", "status": "loaded" if self.text_encoder else "error"},
            {"name": "Audio", "status": "loaded" if self.audio_encoder else "error"},
            {"name": "Face", "status": "loaded" if self.face_encoder else "error"},
            {"name": "Gesture", "status": "loaded" if self.gesture_encoder else "error"},
            {"name": "Sign", "status": "loaded" if self.sign_encoder else "error"},
        ]