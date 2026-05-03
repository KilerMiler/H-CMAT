from __future__ import annotations

import time

import cv2
import mediapipe as mp
import numpy as np

from config.logging import get_logger
from .base_encoder import BaseEncoder

logger = get_logger(__name__)


class SignEncoder(BaseEncoder):
    def __init__(self):
        logger.info("Loading Hand/Sign Signal Encoder (MediaPipe Holistic)... 🤟")
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, image_input: np.ndarray | None = None) -> dict:
        start_time = time.time()
        feature = "no_hand_signal_data"
        uncertainty = 1.0

        try:
            if image_input is not None:
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(image_rgb)

                left_visible = bool(results.left_hand_landmarks)
                right_visible = bool(results.right_hand_landmarks)
                hands_visible = left_visible or right_visible

                if hands_visible:
                    hand_count = int(left_visible) + int(right_visible)
                    feature = f"hand_signal_extracted|hands_visible={hand_count}"
                    uncertainty = 0.10
                elif results.pose_landmarks:
                    feature = "no_hands_detected"
                    uncertainty = 0.85
                else:
                    feature = "no_user_detected"
                    uncertainty = 0.95
            else:
                feature = "dummy_hand_signal_greeting"
                uncertainty = 0.20

        except Exception as exc:
            logger.error(f"Hand/Sign Signal Encoder Error: {exc}")
            feature = "error"
            uncertainty = 1.0

        raw_output = {
            "modality": "SignLang",
            "feature": feature,
            "uncertainty": uncertainty,
            "time_ms": int((time.time() - start_time) * 1000),
        }
        return self.validate_and_format(raw_output)

#     The Logic: Hum Isme Asliyat Mein Kya Kar Rahe Hain?
# 1. __init__ method (The Setup):

# The "Heavy Lifter": Yeh model aapke pipeline ka sabse complex visual model hai. Phir bhi, model_complexity=1 rakhne se yeh M4 Mac ke CPU/Neural Engine par 30-50ms ke andar process ho jayega.

# 2. process method (The Execution):

# The "Hands" Priority Logic: Sign language mein body dikhna kaafi nahi hai. Agar user ke haath frame se bahar hain, ya jeb (pockets) mein hain, toh sign language processing zero ho jani chahiye. Isliye maine explicitly hands_visible check lagaya hai.

# The Uncertainty Matrix (Very Important for H-CMAT Brain):

# Scenario A (Hands + Face clear): Uncertainty = 0.10. Fusion Transformer is channel ki baat maanega.

# Scenario B (Only Body, No Hands): User frame mein hai par sign nahi kar raha. Uncertainty = 0.85. Iska matlab Fusion Brain kahega, "Bhai Sign Language encoder pe trust mat karo, Audio aur Text ki suno."

# Scenario C (Demo Mode): Agar frame hi nahi aaya, toh pipeline toote na isliye ek dummy string bhej do.