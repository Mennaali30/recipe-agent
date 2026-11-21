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
    remember: bool = Field(
        default=False,
        description=(
            "True ONLY if the user explicitly asks to remember / save the allergies or avoided "
            "ingredients for next time (e.g. 'remember that I am allergic to...', 'افتكري إني...')."
        ),
    )
    forget_memory: bool = Field(
        default=False,
        description=(
            "True ONLY if the user explicitly asks to forget / clear saved allergies or "
            "preferences (e.g. 'forget my allergies', 'انسي الحساسيات')."
        ),
    )
    meals: Optional[int] = Field(
        default=None,
        description=(
            "How many separate meals / days the user wants planned (for example 'meals for 3 "
            "days' = 3). null for a normal single-recipe request."
        ),
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description=(
            "ONLY when the user gave no ingredients they have: one short reply in the user's "
            "language that asks which ingredients they have. Otherwise null."
        ),
    )


class SubstitutionChoice(BaseModel):
    """One chosen substitute for one missing ingredient of one recipe."""

    recipe_id: int = Field(description="recipe_id copied exactly from the input.")
    missing: str = Field(description="The missing ingredient, copied exactly from the input.")
    substitute: str = Field(description="One candidate substitute, copied exactly from the candidates.")


class SubstitutionPlan(BaseModel):
    """All substitutions chosen for the recipes (empty if none is sensible)."""

    choices: list[SubstitutionChoice] = Field(default_factory=list)


class MealPlan(BaseModel):
    """Recipes chosen for a multi-day meal plan, in the order of the days."""

    recipe_ids: list[int] = Field(
        default_factory=list,
        description="recipe_id values chosen from the shortlist, copied exactly, one per day.",
    )
