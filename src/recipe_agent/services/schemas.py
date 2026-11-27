"""Pydantic schemas for the deterministic services (Phase 4). No LLM, no network.

Status words used everywhere:
  verified   exact metric quantity, or a price taken from a sourced row (WFP / current web page)
  estimated  derived from a household unit (cup, piece...), a density, or a rough placeholder price
  unknown    could not be worked out; the value is None and is never guessed
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Status = Literal["verified", "estimated", "unknown"]
ExpenseBasis = Literal["fixed", "per_meal"]


def worst_status(statuses) -> Status:
    """unknown beats estimated beats verified. Empty -> unknown."""
    s = set(statuses)
    if not s or "unknown" in s:
        return "unknown"
    return "estimated" if "estimated" in s else "verified"


class Measure(BaseModel):
    """A quantity in a base dimension: 'g', 'ml', or a count unit ('piece', 'clove', 'head', 'can')."""
    value: Optional[float] = None
    unit: Optional[str] = None
    status: Status = "unknown"
    note: str = ""

    def scaled(self, factor: float) -> "Measure":
        if self.value is None:
            return self.model_copy()
        return self.model_copy(update={"value": self.value * factor})


class IngredientInput(BaseModel):
    name: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    amount_text: Optional[str] = None   # e.g. "2 كوب"; used when quantity is not given
    to_taste: bool = False

    @field_validator("quantity")
    @classmethod
    def _non_negative(cls, v):
        if v is not None and v < 0:
            raise ValueError("quantity cannot be negative")
        return v


class ExtraExpense(BaseModel):
    """Cost that is not an ingredient: packaging, gas, labour... Amount is always given by the user."""
    name: str
    amount_egp: float = Field(ge=0)
    basis: ExpenseBasis = "fixed"   # fixed = once for the whole run; per_meal = for every meal


class RecipeInput(BaseModel):
    name: str
    servings: Optional[float] = Field(default=None, gt=0)
    ingredients: list[IngredientInput] = Field(default_factory=list)
    extra_expenses: list[ExtraExpense] = Field(default_factory=list)


class CostLine(BaseModel):
    name: str
    key: Optional[str] = None
    identity: str
    amount_text: Optional[str] = None
    to_taste: bool = False
    measure: Measure = Field(default_factory=Measure)
    cost_egp: Optional[float] = None
    price_egp_per_kg: Optional[float] = None
    price_source: Optional[str] = None      # wfp / current_web / estimated
    price_status: Status = "unknown"
    status: Status = "unknown"               # quantity status combined with price status
    note: str = ""


class RecipeCost(BaseModel):
    recipe: str
    currency: str = "EGP"
    servings: Optional[float] = None
    lines: list[CostLine] = Field(default_factory=list)
    ingredient_cost_egp: Optional[float] = None     # sum of the priced lines (None if none priced)
    extras_egp: float = 0.0
    total_egp: Optional[float] = None                # ingredients + extras; None if nothing priced
    per_serving_egp: Optional[float] = None
    complete: bool = False                           # True only when every counted line is priced
    coverage: Optional[float] = None
    unpriced: list[str] = Field(default_factory=list)
    status: Status = "unknown"
    notes: list[str] = Field(default_factory=list)
    approximate: bool = True


class ShoppingItem(BaseModel):
    name: str
    identity: str
    category: str
    unit: Optional[str] = None
    needed: Optional[float] = None
    in_pantry: float = 0.0
    to_buy: Optional[float] = None
    est_cost_egp: Optional[float] = None
    price_status: Status = "unknown"
    price_source: Optional[str] = None
    quantity_status: Status = "unknown"
    to_taste: bool = False
    note: str = ""


class ShoppingList(BaseModel):
    items: list[ShoppingItem] = Field(default_factory=list)
    estimated_total_egp: Optional[float] = None     # sum of the priced items only
    total_complete: bool = False                     # True only when every item to buy is priced
    unpriced: list[str] = Field(default_factory=list)
    unknown_quantities: list[str] = Field(default_factory=list)
    price_status: Status = "unknown"
    currency: str = "EGP"


class ProductionPlan(BaseModel):
    recipe: str
    meals: int
    recipe_servings: Optional[float] = None
    scale_factor: Optional[float] = None
    scaled_lines: list[CostLine] = Field(default_factory=list)
    missing_quantities: list[str] = Field(default_factory=list)   # to-taste or unconvertible lines
    ingredient_cost_egp: Optional[float] = None
    extras_egp: float = 0.0
    total_cost_egp: Optional[float] = None
    cost_complete: bool = False
    unpriced: list[str] = Field(default_factory=list)
    cost_per_meal_egp: Optional[float] = None
    selling_price_per_meal_egp: Optional[float] = None
    revenue_egp: Optional[float] = None
    gross_profit_egp: Optional[float] = None
    margin_pct: Optional[float] = None
    status: Status = "unknown"
    shopping: Optional[ShoppingList] = None
    notes: list[str] = Field(default_factory=list)
    currency: str = "EGP"


class BudgetEntry(BaseModel):
    recipe: str
    meals: float
    cost_egp: Optional[float] = None
    complete: bool = False


class BudgetCheck(BaseModel):
    budget_egp: float
    basis: Literal["full_cost", "to_buy"] = "full_cost"
    known_total_egp: Optional[float] = None
    complete: bool = False
    verdict: Literal["within_budget", "over_budget", "cannot_tell"] = "cannot_tell"
    remaining_egp: Optional[float] = None
    over_by_egp: Optional[float] = None
    entries: list[BudgetEntry] = Field(default_factory=list)
    unpriced: list[str] = Field(default_factory=list)
    status: Status = "unknown"
    note: str = ""
    currency: str = "EGP"
