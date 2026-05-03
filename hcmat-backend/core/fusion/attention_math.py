"""
core/fusion/attention_math.py

Kept from the original with one interface change:
  - calculate_weights() now accepts a culture_weights dict instead of
    a raw torch.Tensor. This decouples AttentionMath from PyTorch
    entirely, making it independently unit-testable.

The math is unchanged:
  confidence = 1 - uncertainty
  cultural_impact = culture_weights[modality_channel]
  raw_score = confidence * cultural_impact
  normalized_weight = raw_score / sum(all_raw_scores)   (softmax-like)
"""

from __future__ import annotations

from typing import Any, Dict, List

from config.logging import get_logger

logger = get_logger(__name__)

# Maps encoder modality names to the culture_weights channel names
_MODALITY_TO_CHANNEL: dict[str, str] = {
    "Text":     "speech",
    "Audio":    "speech",
    "Face":     "face",
    "Gesture":  "body",
    "SignLang": "body",
}


class AttentionMath:
    """
    Computes normalised attention weights for the 5 encoders given
    their uncertainty scores and the active cultural weight profile.
    """

    def __init__(self) -> None:
        logger.info("AttentionMath initialised.")

    def calculate_weights(
        self,
        encoder_outputs: List[Dict[str, Any]],
        culture_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Args:
            encoder_outputs:  List of dicts with keys 'modality' and 'uncertainty'.
            culture_weights:  {"speech": 0.15, "face": 0.55, "body": 0.30}

        Returns:
            Dict mapping modality name → normalised weight (0.0–1.0).
            All weights sum to 1.0.

        Example:
            Audio has uncertainty=0.95 (noisy room)
              → confidence = 0.05
              → cultural_impact = culture_weights["speech"] = 0.15
              → raw_score = 0.05 * 0.15 = 0.0075
              → after normalisation, Audio weight ≈ 0.01
            Face has uncertainty=0.05 (clear shot)
              → confidence = 0.95
              → cultural_impact = 0.55
              → raw_score = 0.95 * 0.55 = 0.5225
              → after normalisation, Face weight ≈ 0.71
        """
        raw_scores: Dict[str, float] = {}
        total_score: float = 0.0

        for output in encoder_outputs:
            modality    = output["modality"]
            uncertainty = float(output.get("uncertainty", 1.0))

            # Clamp uncertainty to [0.0, 1.0] defensively
            uncertainty = max(0.0, min(1.0, uncertainty))

            # Invert doubt → confidence
            # max(0.01, ...) prevents exact-zero division in edge cases
            confidence = max(0.01, 1.0 - uncertainty)

            # Look up which cultural channel this modality belongs to
            channel = _MODALITY_TO_CHANNEL.get(modality, "body")
            cultural_impact = culture_weights.get(channel, 0.33)

            score = confidence * cultural_impact
            raw_scores[modality] = score
            total_score += score

        # Normalise so all weights sum to 1.0
        if total_score == 0.0:
            # All encoders failed — distribute equally (prevents NaN)
            logger.warning("AttentionMath: total_score=0. Distributing weights equally.")
            n = len(encoder_outputs)
            return {o["modality"]: round(1.0 / n, 4) for o in encoder_outputs}

        return {
            modality: round(score / total_score, 4)
            for modality, score in raw_scores.items()
        }