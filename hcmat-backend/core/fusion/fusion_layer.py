"""
core/fusion/fusion_layer.py

H-CMAT fusion layer.

This version keeps the current demo-friendly 3-channel fusion:
    speech = Text + Audio
    face   = Face
    body   = Gesture + Hand/Sign signal

It improves intent inference by considering cross-modal combinations instead
of relying only on the single highest-weight modality.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from core.fusion.attention_math import AttentionMath
from core.schemas import (
    FuseResponse,
    HolisticFusion,
    ModalityDetail,
    TemporalContext,
)
from config.logging import get_logger

logger = get_logger(__name__)

_MODALITY_TO_CHANNEL: dict[str, str] = {
    "Text": "speech",
    "Audio": "speech",
    "Face": "face",
    "Gesture": "body",
    "SignLang": "body",
}


def _canonical_channel(modality: str) -> str:
    return _MODALITY_TO_CHANNEL.get(modality, modality.lower())


_DEFAULT_INTENT = ("INFORMATION SEEKING", "NEUTRAL")


def _derive_local_tag(modality: str, feature: str, weight: float) -> str:
    """
    Generates a human-readable interpretive label for a modality.
    """
    feature_upper = feature.upper()

    if modality == "Text":
        if weight < 0.15:
            return "LOW TRUST — DEFERRED TO VISUAL"
        if "ERROR" in feature_upper or "NO_SPEECH" in feature_upper:
            return "NO SPEECH DETECTED"
        if "POLITE REFUSAL" in feature_upper or "DISAGREEMENT" in feature_upper:
            return "VERBAL DISAGREEMENT"
        if "AGREEMENT" in feature_upper:
            return "VERBAL AGREEMENT"
        if "HESITATION" in feature_upper or "UNCERTAIN" in feature_upper:
            return "AMBIGUOUS AFFIRMATION"
        if "INFORMATION SEEKING" in feature_upper:
            return "INFORMATION SEEKING"
        return "INTENT EXTRACTED"

    if modality == "Audio":
        if weight < 0.10:
            return "NOISE — IGNORED BY FUSION"
        if "ERROR" in feature_upper or "NO_SPEECH" in feature_upper:
            return "NO SPEECH DETECTED"

        hesitation_markers = ["UM", "UH", "MAYBE", "I GUESS", "FINE", "OKAY", "SURE"]
        if any(marker in feature_upper for marker in hesitation_markers):
            return "AMBIGUOUS AFFIRMATION"

        return "SPEECH SIGNAL CLEAR"

    if modality == "Face":
        if "SUBDUED_EXPRESSION" in feature_upper and weight > 0.25:
            return "SOCIAL COMPLIANCE MASK"
        if "GENUINE_EXPRESSION" in feature_upper:
            return "GENUINE EXPRESSION DETECTED"
        if weight > 0.50:
            return "FACE-DOMINANT SOCIAL CUE"
        return "FACE SIGNAL WEAK"

    if modality == "Gesture":
        if "CLOSED_POSTURE" in feature_upper or "LEANING_AWAY" in feature_upper:
            return "SOFT DISAGREEMENT"
        if "ACTIVE_GESTURE" in feature_upper:
            return "ACTIVE BODY SIGNAL"
        if "BODY_POSE_EXTRACTED" in feature_upper:
            return "NEUTRAL POSTURE"
        return "NO BODY DETECTED"

    if modality == "SignLang":
        if "HAND_SIGNAL_EXTRACTED" in feature_upper:
            return "EXPLICIT HAND SIGNAL DETECTED"
        return "NO HAND SIGNAL DATA"

    return "UNKNOWN"


def _infer_cross_modal_intent(
    modality_matrix: Dict[str, ModalityDetail],
) -> tuple[str, str]:
    """
    Infers final intent from the combined speech/face/body state.

    This is still deterministic and explainable, but better than relying only
    on the single highest-weight modality.
    """
    speech = modality_matrix.get("speech")
    face = modality_matrix.get("face")
    body = modality_matrix.get("body")

    speech_tag = speech.local_tag.upper() if speech else ""
    face_tag = face.local_tag.upper() if face else ""
    body_tag = body.local_tag.upper() if body else ""

    speech_weight = speech.weight if speech else 0.0
    face_weight = face.weight if face else 0.0
    body_weight = body.weight if body else 0.0

    visual_weight = face_weight + body_weight

    speech_agree = (
        "VERBAL AGREEMENT" in speech_tag
        or "SPEECH SIGNAL CLEAR" in speech_tag
        or "AMBIGUOUS AFFIRMATION" in speech_tag
    )

    visual_resistance = (
        "SOCIAL COMPLIANCE MASK" in face_tag
        or "SOFT DISAGREEMENT" in body_tag
    )

    visual_positive = (
        "GENUINE EXPRESSION DETECTED" in face_tag
        and "SOFT DISAGREEMENT" not in body_tag
    )

    explicit_body_signal = "EXPLICIT HAND SIGNAL DETECTED" in body_tag

    # Core H-CMAT culturally relevant case:
    # Words look acceptable, but face/body show resistance.
    if speech_agree and visual_resistance and visual_weight >= speech_weight * 0.75:
        return "SURFACE AGREEMENT", "COGNITIVE DISSONANCE"

    if visual_resistance and visual_weight > speech_weight:
        return "POLITE REFUSAL", "COGNITIVE DISSONANCE"

    if speech_agree and visual_positive:
        return "GENUINE AGREEMENT", "POSITIVE AFFECT"

    if explicit_body_signal and body_weight >= 0.25:
        return "EXPLICIT SIGNAL", "INTENTIONAL"

    if "INFORMATION SEEKING" in speech_tag:
        return "INFORMATION SEEKING", "NEUTRAL"

    if "VERBAL DISAGREEMENT" in speech_tag:
        return "DIRECT DISAGREEMENT", "ASSERTIVE"

    return _DEFAULT_INTENT


class FusionLayer:
    """
    H-CMAT fusion layer.

    Stateless between calls.
    Session memory and NMS are handled outside this class.
    """

    def __init__(self) -> None:
        logger.info("Loading H-CMAT Fusion Layer...")
        self.attention = AttentionMath()
        logger.info("Fusion Layer ready.")

    def fuse(
        self,
        encoder_outputs: List[Dict[str, Any]],
        culture_weights: Dict[str, float],
        seq_id: int,
        temporal_context: TemporalContext,
        pipeline_start: float,
    ) -> FuseResponse:
        fusion_start = time.time()

        raw_weights = self.attention.calculate_weights(
            encoder_outputs,
            culture_weights,
        )

        modality_matrix: Dict[str, ModalityDetail] = {}

        dominant_modality = "Text"
        dominant_channel = "speech"
        highest_weight = -1.0

        for output in encoder_outputs:
            modality = output["modality"]
            feature = str(output["feature"])
            weight = raw_weights.get(modality, 0.0)

            local_tag = _derive_local_tag(modality, feature, weight)
            channel = _canonical_channel(modality)

            detail = ModalityDetail(
                feature=feature,
                weight=weight,
                local_tag=local_tag,
            )

            # Collapse five raw encoders into three human-interpretable channels.
            # If two encoders map to the same channel, keep the more confident /
            # higher-weight representative.
            if channel in modality_matrix:
                if weight > modality_matrix[channel].weight:
                    modality_matrix[channel] = detail
            else:
                modality_matrix[channel] = detail

            if weight > highest_weight:
                highest_weight = weight
                dominant_modality = modality
                dominant_channel = channel

        primary_intent, affective_state = _infer_cross_modal_intent(
            modality_matrix
        )

        fusion_latency_ms = int((time.time() - fusion_start) * 1000)
        total_latency_ms = int((time.time() - pipeline_start) * 1000)

        logger.debug(
            f"[Fusion] seq={seq_id} intent={primary_intent} "
            f"dominant={dominant_modality}/{dominant_channel}({highest_weight:.3f}) "
            f"fusion={fusion_latency_ms}ms total={total_latency_ms}ms"
        )

        return FuseResponse(
            seq_id=seq_id,
            temporal_context=temporal_context,
            modality_matrix=modality_matrix,
            holistic_fusion=HolisticFusion(
                primary_intent=primary_intent,
                affective_state=affective_state,
                confidence=round(max(0.0, highest_weight), 4),
                is_new_event=None,
                replaces_seq_id=None,
            ),
            fusion_latency_ms=fusion_latency_ms,
            total_latency_ms=total_latency_ms,
        )