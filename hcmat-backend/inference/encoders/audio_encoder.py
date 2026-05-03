from __future__ import annotations

import time
from typing import Any

import torch
from transformers import pipeline

from config.logging import get_logger
from .base_encoder import BaseEncoder

logger = get_logger(__name__)


class AudioEncoder(BaseEncoder):
    def __init__(self, device: torch.device):
        logger.info(f"Loading Audio Encoder (Whisper-Tiny) on {device}... 🎧")
        self.device = device
        self.audio_processor = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=self.device,
        )

    def process(self, audio_input: Any) -> dict:
        """
        audio_input may be:
          - None
          - file path
          - numpy waveform
          - dict {"array": np.ndarray, "sampling_rate": int}

        The ParallelEncoderRunner now attempts to decode browser WebM/Opus
        into a Whisper-friendly waveform dict before calling this encoder.
        """
        start_time = time.time()

        if audio_input is None:
            raw_output = {
                "modality": "Audio",
                "feature": "no_audio_provided",
                "uncertainty": 0.85,
                "time_ms": int((time.time() - start_time) * 1000),
            }
            return self.validate_and_format(raw_output)

        try:
            result = self.audio_processor(audio_input)
            transcribed_text = result.get("text", "").strip()

            if transcribed_text:
                uncertainty = 0.15
            else:
                transcribed_text = "no_speech_detected"
                uncertainty = 0.90

        except Exception as exc:
            logger.error(f"Audio Encoder Error: {exc}")
            transcribed_text = "error"
            uncertainty = 1.0

        raw_output = {
            "modality": "Audio",
            "feature": transcribed_text,
            "uncertainty": uncertainty,
            "time_ms": int((time.time() - start_time) * 1000),
        }
        return self.validate_and_format(raw_output)

#     The Logic: Hum Isme Asliyat Mein Kya Kar Rahe Hain?
# 1. __init__ method (The Setup):

# Whisper-Tiny Kyun?: OpenAI ne Whisper ke kai versions banaye hain (Large, Base, Small, Tiny). H-CMAT framework ki "sub-100ms latency" constraint ko dhyan mein rakhte hue humne Tiny choose kiya hai. Iska footprint chhota hai (approx 39M parameters), isliye jab Text Encoder apna kaam kar raha hoga, tab Whisper Mac ke resources par overlap/jam create nahi karega.

# Device Mapping: Phir se, self.device ensure karta hai ki model Apple GPU (mps) par chale.

# 2. process method (The Execution):

# Flexibility of Input: audio_input bahut flexible hai. Demo ke waqt aap ise ek local file ka path (e.g., "test_recording.wav") de sakte hain, ya frontend se aaya hua raw audio byte array. Hugging Face ka pipeline ise apne aap handle kar leta hai.

# The "Uncertainty" Workaround (Candor Check): Text model (DistilRoBERTa) ke paas exact math probability aati hai (jaise 0.85). Par Speech-to-Text models (jaise Whisper) by default pure sentence ka ek single math score nahi dete. Isliye demo ke liye humne ek logical "Proxy" lagaya hai:

# Agar words extract ho gaye (Signal acha hai) -> Uncertainty = 0.15 (Bharosa kiya ja sakta hai).

# Agar sirf shhhh-shhh (Noise) aayi aur words nahi pakde -> Uncertainty = 0.90 (Is channel ko final H-CMAT brain ignore kar dega).