from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from config.logging import get_logger

logger = get_logger(__name__)


class EncoderOutput(BaseModel):
    """
    Strict runtime schema for every encoder output.
    """
    modality: str = Field(..., description="Name of the modality (e.g. Text, Audio, Face)")
    feature: Any = Field(..., description="Extracted signal / label / string")
    uncertainty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Doubt score from 0.0 (certain) to 1.0 (max doubt)",
    )
    time_ms: int = Field(..., ge=0, description="Execution latency in milliseconds")


class BaseEncoder(ABC):
    """
    Abstract base class for all H-CMAT encoders.
    """

    @abstractmethod
    def process(self, input_data: Any) -> dict:
        """
        Runs the encoder on input data and returns a dict in EncoderOutput shape.
        """
        raise NotImplementedError

    def validate_and_format(self, raw_output: dict) -> dict:
        """
        Validates an encoder's raw output against EncoderOutput schema.

        If invalid, logs the error and returns a safe fallback output with
        uncertainty=1.0 so the fusion layer can ignore it safely.
        """
        try:
            validated = EncoderOutput(**raw_output)
            return validated.model_dump()

        except ValidationError as exc:
            logger.error(
                "Encoder output schema validation failed.\n"
                f"Raw output: {raw_output}\n"
                f"Validation error: {exc}"
            )
            return {
                "modality": raw_output.get("modality", "Unknown"),
                "feature": "format_error",
                "uncertainty": 1.0,
                "time_ms": max(0, int(raw_output.get("time_ms", 0) or 0)),
            }
        

#         The Logic: Hum Isme Asliyat Mein Kya Kar Rahe Hain?
# 1. EncoderOutput (The Pydantic Validator):

# FastAPI aur modern Python development mein Pydantic sabse powerful data validator hai.

# Notice kijiye maine uncertainty field mein ge=0.0, le=1.0 (greater than equal to 0, less than equal to 1) lagaya hai. Agar kal ko aap ya koi aur developer galti se uncertainty ko 85 (percentage mein) bhej dega, toh yeh code wahi par error pakad lega aur Fusion layer ko crash hone se bacha lega.

# 2. BaseEncoder (The Abstract Base Class - ABC):

# Object-Oriented Programming (OOP) mein ABC ek contract (shart) ki tarah hota hai.

# Jab aap @abstractmethod lagate hain, toh aap system ko bol rahe hain: "Agar kisi ne naya encoder banaya (jaise kal ko Heart-Rate Encoder), toh usme process naam ka function hona hi chahiye. Agar nahi banaya, toh Mac us encoder ko start hi nahi karega." Yeh aapke code ko future-proof banata hai.

# 3. validate_and_format (The Safety Net):

# Jab paancho encoders apna result nikal lenge, toh hum is function se unke data ko filter karenge. Agar sab sahi hai, toh data Fusion Brain ko pass ho jayega. Agar formatting galat hui, toh yeh automatically uncertainty ko 1.0 kar dega (matlab is data ko ignore kardo), jisse pipeline safe rahegi.

# (Quick Note: Humne jo pichli 4 files—text, audio, face, gesture, sign—banayi hain, unme process function exactly is schema ke hisaab se hi design kiya gaya hai, isliye wahan humein kuch bhi change nahi karna padega!)