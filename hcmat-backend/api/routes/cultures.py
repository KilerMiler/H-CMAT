"""
api/routes/cultures.py

Cultural profile endpoints.

Endpoints:
    GET /api/v1/cultures          → list all profiles
    GET /api/v1/cultures/{id}     → fetch one profile with weights

Current demo profiles:
    culture_id=1 → Indian (High-Context)
    culture_id=2 → Japanese (Collectivist)
    culture_id=3 → American (Low-Context)
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, Request

from config.logging import get_logger
from core.schemas import CultureProfile

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/cultures",
    response_model=List[CultureProfile],
    tags=["Cultures"],
)
async def list_cultures(request: Request) -> List[CultureProfile]:
    """
    Returns all available cultural profiles, sorted by ID.

    React calls this on app load to build the culture selector dropdown.
    """
    mapper = request.app.state.system["mapper"]
    return mapper.list_all()


@router.get(
    "/cultures/{culture_id}",
    response_model=CultureProfile,
    tags=["Cultures"],
)
async def get_culture(culture_id: int, request: Request) -> CultureProfile:
    """
    Returns the full weight configuration for a specific cultural profile.
    """
    mapper = request.app.state.system["mapper"]
    profiles = {p.id: p for p in mapper.list_all()}

    if culture_id not in profiles:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Cultural profile {culture_id} not found. "
                f"Available IDs: {sorted(profiles.keys())}"
            ),
        )

    profile = profiles[culture_id]
    logger.debug(f"Culture profile requested: [{culture_id}] {profile.code}")
    return profile