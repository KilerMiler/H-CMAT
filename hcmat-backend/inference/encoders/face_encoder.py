from __future__ import annotations

import time

import cv2
import mediapipe as mp
import numpy as np

from config.logging import get_logger
from .base_encoder import BaseEncoder

logger = get_logger(__name__)


class FaceEncoder(BaseEncoder):
    def __init__(self):
        logger.info("Loading Face Encoder (MediaPipe Face Mesh)... 🧑‍💻")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    @staticmethod
    def _dist(a, b) -> float:
        return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5)

    def _extract_expression_feature(self, landmarks) -> str:
        """
        Lightweight geometric expression heuristic.
        This is not a full emotion classifier, but it is stronger than only
        checking whether a face exists.
        """
        lm = landmarks.landmark

        # MediaPipe Face Mesh approximate indices.
        left_mouth = lm[61]
        right_mouth = lm[291]
        upper_lip = lm[13]
        lower_lip = lm[14]

        left_eye_top = lm[159]
        left_eye_bottom = lm[145]
        right_eye_top = lm[386]
        right_eye_bottom = lm[374]

        mouth_width = self._dist(left_mouth, right_mouth)
        mouth_open = self._dist(upper_lip, lower_lip)
        eye_open = (
            self._dist(left_eye_top, left_eye_bottom)
            + self._dist(right_eye_top, right_eye_bottom)
        ) / 2

        mouth_open_ratio = mouth_open / max(mouth_width, 1e-6)

        if mouth_open_ratio > 0.20 and eye_open > 0.010:
            expression = "genuine_expression"
            uncertainty = 0.08
        elif mouth_open_ratio < 0.08:
            expression = "subdued_expression"
            uncertainty = 0.12
        else:
            expression = "neutral_expression"
            uncertainty = 0.10

        return (
            f"face_landmarks_extracted|expression={expression}|"
            f"mouth_open_ratio={mouth_open_ratio:.3f}|eye_open={eye_open:.3f}|"
            f"uncertainty_hint={uncertainty:.2f}"
        )

    def process(self, image_input: np.ndarray | None = None) -> dict:
        start_time = time.time()
        feature = "no_face_data"
        uncertainty = 1.0

        try:
            if image_input is not None:
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    feature = self._extract_expression_feature(
                        results.multi_face_landmarks[0]
                    )

                    if "genuine_expression" in feature:
                        uncertainty = 0.08
                    elif "subdued_expression" in feature:
                        uncertainty = 0.12
                    else:
                        uncertainty = 0.10
                else:
                    feature = "no_face_detected"
                    uncertainty = 0.95
            else:
                feature = "dummy_attentive_face"
                uncertainty = 0.10

        except Exception as exc:
            logger.error(f"Face Encoder Error: {exc}")
            feature = "error"
            uncertainty = 1.0

        raw_output = {
            "modality": "Face",
            "feature": feature,
            "uncertainty": uncertainty,
            "time_ms": int((time.time() - start_time) * 1000),
        }
        return self.validate_and_format(raw_output)
    

#     The Logic: Hum Isme Asliyat Mein Kya Kar Rahe Hain?
# 1. __init__ method (The Setup):

# refine_landmarks=True: Yeh line H-CMAT framework ke liye bahut zaroori hai. Sirf chehre ka dabba (bounding box) banana kaafi nahi hai. Culturally-aware model ko yeh pata hona chahiye ki user smile kar raha hai, frown kar raha hai, ya confuse hai. Yeh setting aankhon ki putliyon (irises) aur hothon ki shape ke extra detail nikal kar deti hai.

# max_num_faces=1: Hum nahi chahte ki background mein khade kisi doosre insaan ka chehra humare main user ke data ko corrupt kare. Yeh us noise ko filter out kar deta hai.

# 2. process method (The Execution):

# Format Conversion: Camera se aane wali pictures mostly BGR (Blue-Green-Red) format mein hoti hain (OpenCV standard). Lekin AI models ko RGB chahiye hota hai. cv2.cvtColor is choti si problem ko fix karta hai warna model galat data padh lega.

# The "Dummy Mode" Fix: Mujhe pata hai demo test karte waqt aap har baar camera on nahi karna chahenge. Isliye maine code mein ek if image_input is not None: condition laga di hai. Agar aap API ko video nahi denge, toh yeh crash nahi hoga, balki ek "dummy feature" bhej dega taaki pipeline smoothly chalti rahe.


# The Uncertainty Metric: Agar chehra frame mein ekdum clear hai, toh humara uncertainty score 0.05 (sirf 5% doubt) hoga. Agar frame andhera hai ya chehra screen se bahar hai, toh doubt 0.95 ho jayega, aur "Fusion Brain" is visual data ko ignore karke poora focus Audio aur Text par shift kar dega.