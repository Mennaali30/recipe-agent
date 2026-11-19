"""Pydantic models shared by the agents."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class UserRequest(BaseModel):
    """What the user asked for, extracted from free text (Arabic or English)."""

    ingredients: list[str] = Field(
        default_factory=list,
        description="Ingredients the user HAS, as short singular English names (e.g. 'chicken', 'tomato').",
    )
    exclude_ingredients: list[str] = Field(
        default_factory=list,
        description="Ingredients the user wants to AVOID (allergy, dislike, diet), in English.",
    )
    max_minutes: Optional[int] = Field(
        default=None,
        description="Maximum total cooking time in minutes, or null if not mentioned.",
    )
    language: Literal["ar", "en"] = Field(
        default="en",
        description="Language the user wrote in: 'ar' for Arabic, 'en' for English.",
    )
