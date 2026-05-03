from __future__ import annotations

import time

import cv2
import mediapipe as mp
import numpy as np

from config.logging import get_logger
from .base_encoder import BaseEncoder

logger = get_logger(__name__)


class GestureEncoder(BaseEncoder):
    def __init__(self):
        logger.info("Loading Gesture Encoder (MediaPipe Pose)... 🧍‍♂️")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
        )

    def _extract_posture_feature(self, landmarks) -> str:
        """
        Lightweight posture heuristic using pose landmarks.
        Detects open/closed/leaning posture from shoulder and wrist positions.
        """
        lm = landmarks.landmark
        P = self.mp_pose.PoseLandmark

        left_shoulder = lm[P.LEFT_SHOULDER]
        right_shoulder = lm[P.RIGHT_SHOULDER]
        left_wrist = lm[P.LEFT_WRIST]
        right_wrist = lm[P.RIGHT_WRIST]
        left_hip = lm[P.LEFT_HIP]
        right_hip = lm[P.RIGHT_HIP]

        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2

        torso_lean = shoulder_center_x - hip_center_x

        wrists_between_shoulders = (
            min(left_shoulder.x, right_shoulder.x)
            <= left_wrist.x
            <= max(left_shoulder.x, right_shoulder.x)
            and min(left_shoulder.x, right_shoulder.x)
            <= right_wrist.x
            <= max(left_shoulder.x, right_shoulder.x)
        )

        wrists_high = (
            left_wrist.y < left_shoulder.y + 0.15
            or right_wrist.y < right_shoulder.y + 0.15
        )

        if wrists_between_shoulders:
            posture = "closed_posture"
        elif abs(torso_lean) > 0.08:
            posture = "leaning_away"
        elif wrists_high:
            posture = "active_gesture"
        else:
            posture = "neutral_posture"

        return (
            f"body_pose_extracted|posture={posture}|"
            f"torso_lean={torso_lean:.3f}|wrists_high={wrists_high}"
        )

    def process(self, image_input: np.ndarray | None = None) -> dict:
        start_time = time.time()
        feature = "no_gesture_data"
        uncertainty = 1.0

        try:
            if image_input is not None:
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)

                if results.pose_landmarks:
                    feature = self._extract_posture_feature(results.pose_landmarks)
                    uncertainty = 0.10
                else:
                    feature = "no_body_detected"
                    uncertainty = 0.90
            else:
                feature = "dummy_neutral_posture"
                uncertainty = 0.15

        except Exception as exc:
            logger.error(f"Gesture Encoder Error: {exc}")
            feature = "error"
            uncertainty = 1.0

        raw_output = {
            "modality": "Gesture",
            "feature": feature,
            "uncertainty": uncertainty,
            "time_ms": int((time.time() - start_time) * 1000),
        }
        return self.validate_and_format(raw_output)

#     The Logic: Hum Isme Asliyat Mein Kya Kar Rahe Hain?
# 1. __init__ method (The Setup):

# model_complexity=1: MediaPipe Pose 3 alag-alag level par aata hai (0, 1, 2). Level 0 sabse fast par thoda kam accurate hota hai. Level 2 super accurate hota hai par latency badha deta hai. Aapke M4 Mac ke liye Level 1 (Balanced) ekdum perfect hai. Yeh accuracy bhi dega aur aapki latency ko 20-30ms ke andar bhi rakhega.

# smooth_landmarks=True: Jab user camera ke samne move karta hai, toh points thode kaanpte (jitter) hain. Yeh setting un mathematical points ko smooth kar deti hai, jisse aapke H-CMAT Fusion Brain ko clear signal milta hai.

# 2. process method (The Execution):

# The 33 Keypoints: Jab results.pose_landmarks true hota hai, toh iska matlab AI ne user ke 33 joints dhoondh liye hain. Asli production environment mein (paper ke hisaab se), hum inhi points ki math calculations se nikalenge ki user "Crossed Arms" (defensive culture) mein khada hai ya "Open Arms" (welcoming culture) mein.

# The Dummy Mode: Face Encoder ki tarah hi, agar aap API testing ke waqt camera feed nahi bhejte (image_input = None), toh yeh script crash nahi hogi. Yeh ek dummy feature bhej degi taaki pipeline test fail na ho.

# Uncertainty Assignment: Yahan bhi uncertainty (doubt) is baat par depend karti hai ki AI ko user ki body kitni clear dikh rahi hai. Agar point extract hue hain, toh error margin kam hai (0.10), warna ignore karne ke liye score high hai (0.90).