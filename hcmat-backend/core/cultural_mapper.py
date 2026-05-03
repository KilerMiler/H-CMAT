"""
core/cultural_mapper.py

Loads cultural profiles from data/cultures/*.json at startup.

Key design:
- Adding a new culture = drop a new JSON file. No Python changes required.
- REST routes expose these profiles directly.
- Inference uses culture weights dynamically on every chunk, allowing
  live frontend switching during the demo.

Important:
- Unknown culture IDs are now treated as errors instead of silently falling
  back to a default profile. This preserves explainability.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, Optional

from config.logging import get_logger
from config.settings import settings
from core.schemas import CultureProfile

logger = get_logger(__name__)


class CulturalMapper:
    """
    Loads all cultural profiles from data/cultures/*.json at startup
    and provides fast O(1) lookup by ID during inference.
    """

    def __init__(self, cultures_dir: Optional[pathlib.Path] = None):
        self._cultures_dir = cultures_dir or settings.cultures_dir
        self._profiles: Dict[int, CultureProfile] = {}

        self._load_all()

    def _load_all(self) -> None:
        """
        Scans the cultures directory and loads every valid *.json file.
        Invalid files are skipped with a warning so one bad file does not
        break the full demo startup.
        """
        logger.info(f"Loading cultural profiles from {self._cultures_dir}")

        all_files = sorted(self._cultures_dir.glob("*.json"))
        json_files = [f for f in all_files if not f.name.startswith((".", "_"))]

        if not json_files:
            logger.warning(
                f"No culture JSON files found in {self._cultures_dir}. "
                "CulturalMapper will have zero profiles."
            )
            return

        for path in json_files:
            try:
                text = path.read_text(encoding="utf-8-sig").strip()
                if not text:
                    logger.warning(f"  - Skipping empty file: {path.name}")
                    continue

                raw = json.loads(text)
                profile = CultureProfile(**raw)
                self._profiles[profile.id] = profile

                logger.info(
                    f"  ✓ Loaded culture [{profile.id}] "
                    f"{profile.code} — {profile.name}"
                )

            except Exception as exc:
                logger.warning(f"  ✗ Skipped {path.name}: {exc}")

        logger.info(
            f"Cultural Mapper ready — {len(self._profiles)} profiles loaded."
        )

    def has_id(self, culture_id: int) -> bool:
        """Returns True if the culture ID exists."""
        return culture_id in self._profiles

    def available_ids(self) -> list[int]:
        """Returns sorted list of valid culture IDs."""
        return sorted(self._profiles.keys())

    def get_by_id(self, culture_id: int) -> CultureProfile:
        """
        Returns the CultureProfile for the given ID.

        Raises:
            KeyError if the ID does not exist.
        """
        if culture_id not in self._profiles:
            raise KeyError(
                f"Cultural profile {culture_id} not found. "
                f"Available IDs: {self.available_ids()}"
            )

        return self._profiles[culture_id]

    def list_all(self) -> list[CultureProfile]:
        """Returns all loaded profiles sorted by ID."""
        return sorted(self._profiles.values(), key=lambda p: p.id)

    def get_weights_for_fusion(self, culture_id: int) -> dict[str, float]:
        """
        Returns modality weight dict ready for FusionLayer.

        Example:
            {"speech": 0.15, "face": 0.55, "body": 0.30}
        """
        profile = self.get_by_id(culture_id)
        return {
            "speech": profile.weights.speech,
            "face": profile.weights.face,
            "body": profile.weights.body,
        }

    @property
    def profile_count(self) -> int:
        return len(self._profiles)