from __future__ import annotations

import time

import torch
from transformers import pipeline

from config.logging import get_logger
from .base_encoder import BaseEncoder

logger = get_logger(__name__)


class TextEncoder(BaseEncoder):
    def __init__(self, device: torch.device):
        logger.info(f"Loading Text Encoder (Zero-Shot Pragmatics) on {device}... 📚")
        self.device = device
        self.classifier = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-distilroberta-base",
            device=self.device,
        )

        # Aligned with the H-CMAT pragmatic intent layer.
        self.candidate_labels = [
            "agreement",
            "disagreement",
            "polite refusal",
            "hesitation",
            "information seeking",
            "greeting",
            "uncertain response",
            "out of scope",
        ]

    def process(self, user_text: str) -> dict:
        start_time = time.time()

        if not user_text or not user_text.strip():
            raw_output = {
                "modality": "Text",
                "feature": "no_speech_detected",
                "uncertainty": 0.90,
                "time_ms": int((time.time() - start_time) * 1000),
            }
            return self.validate_and_format(raw_output)

        try:
            result = self.classifier(user_text.strip(), self.candidate_labels)
            top_intent = result["labels"][0]
            top_score = float(result["scores"][0])
            uncertainty = round(1.0 - top_score, 4)

        except Exception as exc:
            logger.error(f"Text Encoder Error: {exc}")
            top_intent = "error"
            uncertainty = 1.0

        raw_output = {
            "modality": "Text",
            "feature": top_intent,
            "uncertainty": uncertainty,
            "time_ms": int((time.time() - start_time) * 1000),
        }
        return self.validate_and_format(raw_output)
    

#     The Logic: Hum Isme Asliyat Mein Kya Kar Rahe Hain?
# Aaiye is code ke peechhe ka H-CMAT logic Hinglish mein samajhte hain:

# 1. __init__ method (The Setup):

# One-Time Loading: Jab aapka FastAPI server start hoga, yeh function sirf ek baar chalega. Yeh lagbhag 300MB ka model memory mein load karke rakh lega. Agar hum ise har request par load karte, toh har baar 2-3 second ka lag aata.

# Hardware Mapping: self.device variable ensure karta hai ki model Apple Silicon (MPS) par hi run kare taaki ultra-fast speed mile.

# The "Generalist" Labels: Humne candidate_labels ki ek list define ki hai. Yeh wahi "Librarian" wali approach hai. Model inhi categories mein user ki baat ko sort karega.

# 2. process method (The Execution):

# The Pipeline Call: self.classifier(user_text, self.candidate_labels) wo main line hai jahan AI apna dimaag lagata hai. Yeh user text ko padhta hai aur har label ko ek math score deta hai.

# The Critical Requirement: Aapke research paper ki sabse strict condition thi ki har Leg ko ek Feature Vector aur Uncertainty Metric dena hoga.

# Feature Vector: Yahan humara feature vector wo sabse top category (top_intent) hai jo AI ne pehchani hai.

# Uncertainty Metric: Model ki jo confidence thi (e.g., 0.85 ya 85%), humne use 1.0 se minus kar diya. Toh uncertainty 0.15 (ya 15% doubt) ban gayi. Yeh number aage chalkar "Fusion Brain" ko batayega ki is Text output par kitna bharosa karna hai.

# Safe Fallback (try-except): Demo ke waqt agar kisi wajah se API crash hoti hai ya blank text pass ho jata hai, toh yeh poore server ko band nahi hone dega. Yeh sidha uncertainty = 1.0 (100% doubt) return kar dega, jisse Fusion Brain isko reject kar dega.

# Yeh thi aapke Text Encoder ki poori kahani!