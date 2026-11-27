"""Typed tools over the Phase 4 services.

Each tool has a name, a description, a Pydantic argument model and a function. Calling a tool
validates the arguments, checks that the calling agent may use it, and never raises: a failure comes
back as ToolResult(ok=False, error_code=..., error=...). The tools do no LLM calls and no network.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

import json
import os

from recipe_agent.services import (ApprovalService, ExtraExpense, HistoryService, IngredientInput, InsufficientStock, PantryService,
                                   ProfileService, ProfileUpdate,
                                   RecipeInput, UnitError, build_shopping_list, check_budget,
                                   convert, plan_production, recipe_cost, recipe_from_fifi,
                                   resolve_ingredient)
from recipe_agent.services.costing import cost_line


# ---------------------------------------------------------------- argument models
class RecipeRef(BaseModel):
    """A recipe: an Egyptian recipe by name or id, or ingredients given by the user."""
    name: Optional[str] = Field(default=None, description="Dish name as the user wrote it.")
    recipe_id: Optional[str] = Field(default=None, description="Only if given exactly (egy:...).")
    servings: Optional[float] = Field(default=None, gt=0, description="Servings, only if the user says so.")
    ingredients: list[IngredientInput] = Field(
        default_factory=list, description="Only when the user lists the ingredients and amounts.")


class ConvertArgs(BaseModel):
    quantity: float = Field(ge=0)
    unit: Optional[str] = None
    ingredient: str


class CostArgs(BaseModel):
    recipe: RecipeRef
    extra_expenses: list[ExtraExpense] = Field(default_factory=list)


class ShoppingRecipe(BaseModel):
    recipe: RecipeRef
    meals: Optional[float] = Field(default=None, gt=0, description="Meals wanted; null = as written.")


class ShoppingArgs(BaseModel):
    recipes: list[ShoppingRecipe] = Field(min_length=1)
    use_pantry: bool = True


class ProductionArgs(BaseModel):
    recipe: RecipeRef
    meals: int = Field(gt=0)
    selling_price_per_meal: Optional[float] = Field(default=None, ge=0)
    extra_expenses: list[ExtraExpense] = Field(default_factory=list)
    target_margin_pct: Optional[float] = Field(default=None, ge=0, lt=100)
    use_pantry: bool = False


class BudgetItem(BaseModel):
    recipe: RecipeRef
    meals: float = Field(gt=0)


class BudgetArgs(BaseModel):
    items: list[BudgetItem] = Field(min_length=1)
    budget_egp: float = Field(ge=0)
    use_pantry: bool = False


class PantryEntry(BaseModel):
    name: str
    quantity: Optional[float] = Field(default=None, ge=0)
    unit: Optional[str] = None
    expiry: Optional[date] = None
    low_stock_threshold: Optional[float] = Field(default=None, ge=0)


class PantryChangeArgs(BaseModel):
    items: list[PantryEntry] = Field(min_length=1)


class PantryUseArgs(BaseModel):
    items: list[PantryEntry] = Field(min_length=1)


class PantryDiscardArgs(BaseModel):
    items: list[PantryEntry] = Field(min_length=1, description="Items to throw away completely (name, optional unit).")


class PantryAdjustArgs(BaseModel):
    items: list[PantryEntry] = Field(min_length=1, description="Items with the exact quantity now on the shelf.")


class PantryReadArgs(BaseModel):
    within_days: int = Field(default=3, ge=0, le=60)


class EmptyArgs(BaseModel):
    pass


class ProfileChangeArgs(BaseModel):
    update: Optional[ProfileUpdate] = None
    allergies: list[str] = Field(default_factory=list, description="Allergies to SAVE (only on an explicit request).")


class ProfileClearArgs(BaseModel):
    what: Literal["allergies", "profile", "history", "all"]


# ---------------------------------------------------------------- plumbing
@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str = ""
    error_code: str = ""          # bad_args / not_allowed / not_found / ambiguous / insufficient_stock / unit / failed
    candidates: list = field(default_factory=list)


@dataclass
class ToolContext:
    user_id: str = "default"
    session_factory: Optional[Callable] = None    # () -> SQLAlchemy Session
    store_loader: Optional[Callable] = None       # () -> egypt_store-like {"entries": {...}}
    today: Optional[date] = None
    message: str = ""                              # the raw user message being handled
    history: list = field(default_factory=list)    # recent HistoryEntry items, oldest first
    memory: Any = None                             # allergy memory module (default recipe_agent.memory)
    approved: bool = False                         # True only when the user just confirmed this exact action


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type
    fn: Callable
    writes: bool = False
    approval: bool = False                         # needs the user's yes first (see services/approvals.py)
    describe: Optional[Callable] = None            # (ctx, parsed args) -> {"ar": ..., "en": ...}


class ToolError(Exception):
    def __init__(self, code, message, candidates=None):
        super().__init__(message)
        self.code, self.candidates = code, candidates or []


class ToolRegistry:
    def __init__(self, specs):
        self.specs = {s.name: s for s in specs}

    def names(self):
        return sorted(self.specs)

    def describe(self, names=None):
        return [{"name": s.name, "description": s.description,
                 "args": s.args_model.model_json_schema(), "writes": s.writes}
                for n, s in sorted(self.specs.items()) if names is None or n in names]

    def call(self, name: str, args, ctx: ToolContext, allowed=None) -> ToolResult:
        spec = self.specs.get(name)
        if spec is None:
            return ToolResult(False, error=f"unknown tool {name}", error_code="not_found")
        if allowed is not None and name not in allowed:
            return ToolResult(False, error=f"this agent may not use tool {name}", error_code="not_allowed")
        try:
            parsed = args if isinstance(args, spec.args_model) else spec.args_model.model_validate(args)
        except ValidationError as exc:
            return ToolResult(False, error=str(exc.errors()[0]["msg"]) if exc.errors() else "bad arguments",
                              error_code="bad_args")
        if spec.approval and not ctx.approved:
            return self._stage(spec, parsed, ctx)
        try:
            return ToolResult(True, data=spec.fn(ctx, parsed))
        except ToolError as exc:
            return ToolResult(False, error=str(exc), error_code=exc.code, candidates=exc.candidates)
        except InsufficientStock as exc:
            return ToolResult(False, error=str(exc), error_code="insufficient_stock")
        except (UnitError, ValueError) as exc:
            return ToolResult(False, error=str(exc), error_code="unit" if isinstance(exc, UnitError) else "bad_args")
        except Exception as exc:  # a tool bug must not crash the whole graph
            return ToolResult(False, error=f"{type(exc).__name__}: {exc}", error_code="failed")


    def _stage(self, spec, parsed, ctx) -> ToolResult:
        """Do not run a sensitive tool: save it and ask the user. Fails closed (nothing runs) on any problem."""
        if ctx.session_factory is None:
            return ToolResult(False, error="approval needs a database", error_code="failed")
        try:
            try:
                summary = spec.describe(ctx, parsed) if spec.describe else {}
            except Exception:
                summary = {}
            ar = summary.get("ar") or f"تنفيذ: {spec.name}"
            en = summary.get("en") or f"Run: {spec.name}"
            with _Session(ctx) as s:
                ApprovalService(s).stage(ctx.user_id, spec.name, parsed.model_dump(mode="json"), ar, en)
            return ToolResult(False, data={"ar": ar, "en": en}, error="waiting for the user's approval",
                              error_code="needs_approval")
        except Exception as exc:
            return ToolResult(False, error=f"{type(exc).__name__}: {exc}", error_code="failed")


# ---------------------------------------------------------------- recipe lookup
_AR_MARKS = re.compile("[\u064b-\u065f\u0670\u0640]")


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = _AR_MARKS.sub("", s)
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه")):
        s = s.replace(a, b)
    return re.sub(r"[^\w\s]", " ", s).strip()


def _entries(ctx: ToolContext) -> dict:
    if ctx.store_loader:
        return ctx.store_loader()["entries"]
    from recipe_agent import egypt_store
    return egypt_store.load_store()["entries"]


def find_egyptian_recipe(ctx: ToolContext, query: str) -> dict:
    """The one curated entry that best matches `query`. Raises ToolError not_found / ambiguous."""
    q = norm_text(query)
    if not q:
        raise ToolError("not_found", "no recipe name given")
    tiers = {0: [], 1: [], 2: []}
    qtokens = set(q.split())
    for e in _entries(ctx).values():
        names = [norm_text(e.get("name", "")), norm_text(e.get("name_en", ""))]
        names = [n for n in names if n]
        if q in names:
            tiers[0].append(e)
        elif any(q in n or (n and n in q) for n in names):
            tiers[1].append(e)
        elif any(qtokens and qtokens <= set(n.split()) for n in names):
            tiers[2].append(e)
    for t in (0, 1, 2):
        if len(tiers[t]) == 1:
            return tiers[t][0]
        if len(tiers[t]) > 1:
            raise ToolError("ambiguous", f"more than one recipe matches '{query}'",
                            [{"recipe_id": e["recipe_id"], "name": e["name"]} for e in tiers[t][:5]])
    raise ToolError("not_found", f"no Egyptian recipe named '{query}'")


def resolve_recipe(ctx: ToolContext, ref: RecipeRef) -> RecipeInput:
    """Ingredients given by the user win; else an exact recipe_id; else a lookup by name."""
    if ref.ingredients:
        return RecipeInput(name=ref.name or "recipe", servings=ref.servings, ingredients=ref.ingredients)
    if ref.recipe_id:
        entries = _entries(ctx)
        if ref.recipe_id not in entries:
            raise ToolError("not_found", f"unknown recipe id {ref.recipe_id}")
        entry = entries[ref.recipe_id]
    elif ref.name:
        entry = find_egyptian_recipe(ctx, ref.name)
    else:
        raise ToolError("not_found", "no recipe given")
    rec = recipe_from_fifi(entry)
    if ref.servings:
        rec = rec.model_copy(update={"servings": ref.servings})
    return rec


# ---------------------------------------------------------------- tool functions
class _Session:
    def __init__(self, ctx):
        if ctx.session_factory is None:
            raise ToolError("failed", "no database configured")
        self.s = ctx.session_factory()

    def __enter__(self):
        return self.s

    def __exit__(self, *a):
        self.s.close()


def _stock(ctx, use_pantry):
    if not use_pantry:
        return None
    with _Session(ctx) as s:
        return PantryService(s).stock_map(ctx.user_id)


def t_convert(ctx, a: ConvertArgs):
    return convert(a.quantity, a.unit, resolve_ingredient(a.ingredient))


def t_cost(ctx, a: CostArgs):
    rec = resolve_recipe(ctx, a.recipe)
    rec = rec.model_copy(update={"extra_expenses": [*rec.extra_expenses, *a.extra_expenses]})
    return recipe_cost(rec)


def t_shopping(ctx, a: ShoppingArgs):
    lines = []
    for item in a.recipes:
        rec = resolve_recipe(ctx, item.recipe)
        if item.meals is not None and not rec.servings:
            raise ToolError("bad_args", f"servings of '{rec.name}' unknown; cannot scale to {item.meals:g} meals")
        factor = item.meals / rec.servings if item.meals is not None else 1.0
        lines += [cost_line(i, factor)[0] for i in rec.ingredients]
    return build_shopping_list(lines, _stock(ctx, a.use_pantry))


def t_production(ctx, a: ProductionArgs):
    from recipe_agent.services import min_price_for_margin
    rec = resolve_recipe(ctx, a.recipe)
    plan = plan_production(rec, a.meals, a.selling_price_per_meal, a.extra_expenses,
                           stock=_stock(ctx, a.use_pantry))
    suggested = None
    if a.target_margin_pct is not None and plan.cost_complete and plan.cost_per_meal_egp is not None:
        suggested = min_price_for_margin(plan.cost_per_meal_egp, a.target_margin_pct)
    return {"plan": plan, "suggested_price_egp": suggested, "target_margin_pct": a.target_margin_pct}


def t_budget(ctx, a: BudgetArgs):
    items = [(resolve_recipe(ctx, i.recipe), i.meals) for i in a.items]
    return check_budget(items, a.budget_egp, stock=_stock(ctx, a.use_pantry))


def t_pantry_add(ctx, a: PantryChangeArgs):
    out = []
    with _Session(ctx) as s:
        svc = PantryService(s)
        for e in a.items:
            if not e.quantity:
                raise ToolError("bad_args", f"quantity of {e.name} is missing")
            out.append(svc.add_item(ctx.user_id, e.name, e.quantity, e.unit, expiry=e.expiry,
                                    low_stock_threshold=e.low_stock_threshold))
    return out


def _stock_text(ctx, name) -> str:
    """What is in the pantry for one item now, e.g. '2000 g' (for the confirmation question)."""
    try:
        with _Session(ctx) as s:
            ms = PantryService(s).get_stock(ctx.user_id, name)
        parts = [f"{m.value:g} {m.unit}" for m in ms if m.value]
        return " + ".join(parts) if parts else "0"
    except Exception:
        return "?"


def d_pantry_discard(ctx, a: PantryDiscardArgs):
    rows = "، ".join(f"{e.name} ({_stock_text(ctx, e.name)})" for e in a.items)
    rows_en = ", ".join(f"{e.name} ({_stock_text(ctx, e.name)})" for e in a.items)
    return {"ar": f"مسح كل الكمية من المخزن: {rows}", "en": f"Throw away all the stock of: {rows_en}"}


def d_pantry_adjust(ctx, a: PantryAdjustArgs):
    rows = "، ".join(f"{e.name}: {_stock_text(ctx, e.name)} ← {e.quantity:g} {e.unit or ''}".rstrip() for e in a.items)
    rows_en = ", ".join(f"{e.name}: {_stock_text(ctx, e.name)} -> {e.quantity:g} {e.unit or ''}".rstrip() for e in a.items)
    return {"ar": f"تعديل كمية المخزن إلى الرقم الجديد: {rows}", "en": f"Set the pantry stock to the new amount: {rows_en}"}


def d_profile_clear(ctx, a):
    ar = {"allergies": "الحساسيات المحفوظة", "profile": "البروفايل", "history": "تاريخ المحادثات",
          "all": "الحساسيات والبروفايل وتاريخ المحادثات"}[a.what]
    en = {"allergies": "your saved allergies", "profile": "your profile", "history": "your chat history",
          "all": "your saved allergies, profile and chat history"}[a.what]
    return {"ar": f"مسح {ar}", "en": f"Delete {en}"}


def t_pantry_discard(ctx, a: PantryDiscardArgs):
    """All or nothing: every item must exist first."""
    with _Session(ctx) as s:
        svc = PantryService(s)
        for e in a.items:
            if not [m for m in svc.get_stock(ctx.user_id, e.name)]:
                raise ToolError("not_found", f"{e.name} is not in the pantry")
        return [svc.discard_item(ctx.user_id, e.name, e.unit) for e in a.items]


def t_pantry_adjust(ctx, a: PantryAdjustArgs):
    for e in a.items:
        if e.quantity is None:
            raise ToolError("bad_args", f"the new quantity of {e.name} is missing")
    with _Session(ctx) as s:
        svc = PantryService(s)
        return [svc.adjust_item(ctx.user_id, e.name, e.quantity, e.unit) for e in a.items]


def t_pantry_use(ctx, a: PantryUseArgs):
    """All or nothing: every item is checked against the stock first, so a failure changes nothing."""
    with _Session(ctx) as s:
        svc = PantryService(s)
        stock = svc.stock_map(ctx.user_id)
        need = {}
        for e in a.items:
            if not e.quantity:
                raise ToolError("bad_args", f"quantity of {e.name} is missing")
            ing, m = svc._measure(e.name, e.quantity, e.unit)
            k = (ing.identity, m.unit)
            need[k] = (need.get(k, (0.0, e.name, m.unit))[0] + m.value, e.name, m.unit)
        for k, (q, name, unit) in need.items():
            have = stock.get(k, 0.0)
            if have + 1e-9 < q:
                raise InsufficientStock(name, have, q, unit)
        return [svc.use_item(ctx.user_id, e.name, e.quantity, e.unit) for e in a.items]


def t_pantry_list(ctx, a: EmptyArgs):
    with _Session(ctx) as s:
        return PantryService(s).list_items(ctx.user_id)


def t_pantry_expiring(ctx, a: PantryReadArgs):
    with _Session(ctx) as s:
        svc = PantryService(s)
        return {"expired": svc.expired(ctx.user_id, ctx.today),
                "expiring_soon": svc.expiring_soon(ctx.user_id, a.within_days, ctx.today),
                "within_days": a.within_days}


def t_pantry_low(ctx, a: EmptyArgs):
    with _Session(ctx) as s:
        return PantryService(s).low_stock(ctx.user_id)


def t_profile_show(ctx, a: EmptyArgs):
    with _Session(ctx) as s:
        return ProfileService(s, ctx.memory).get(ctx.user_id)


def t_profile_update(ctx, a: ProfileChangeArgs):
    has_update = a.update is not None and bool(a.update.model_dump(exclude_none=True))
    if not has_update and not a.allergies:
        raise ToolError("bad_args", "nothing to save")
    with _Session(ctx) as s:
        svc = ProfileService(s, ctx.memory)
        if has_update:
            svc.update(ctx.user_id, a.update)
        if a.allergies:
            svc.add_allergies(ctx.user_id, a.allergies)
        return svc.get(ctx.user_id)


def t_profile_clear(ctx, a: ProfileClearArgs):
    done = {}
    with _Session(ctx) as s:
        svc = ProfileService(s, ctx.memory)
        if a.what in ("allergies", "all"):
            svc.clear_allergies(ctx.user_id)
            done["allergies"] = True
        if a.what in ("profile", "all"):
            done["profile"] = svc.delete_profile(ctx.user_id)
        if a.what in ("history", "all"):
            done["history_messages"] = HistoryService(s).clear(ctx.user_id)
    return done


DEFAULT_APPROVAL_TOOLS = ("pantry_discard", "pantry_adjust", "profile_clear")


def approval_tools_from_env(default=DEFAULT_APPROVAL_TOOLS):
    """RECIPE_APPROVAL_TOOLS="pantry_discard,pantry_add" replaces the list; "none" turns approval off."""
    raw = os.getenv("RECIPE_APPROVAL_TOOLS")
    if raw is None:
        return set(default)
    return set() if raw.strip().lower() in ("", "none") else {x.strip() for x in raw.split(",") if x.strip()}


def build_registry(approval_tools=None) -> ToolRegistry:
    need = approval_tools_from_env() if approval_tools is None else set(approval_tools)
    reg = _base_registry()
    for name in need:
        if name not in reg.specs:
            raise ValueError(f"unknown tool in the approval list: {name}")
        reg.specs[name].approval = True
    return reg


def _base_registry() -> ToolRegistry:
    return ToolRegistry([
        ToolSpec("convert_units", "Convert a quantity of an ingredient to grams / ml / count.", ConvertArgs, t_convert),
        ToolSpec("recipe_cost", "Cost and cost per serving of a recipe, with optional extra expenses.", CostArgs, t_cost),
        ToolSpec("shopping_list", "Merged shopping list, minus the pantry, grouped, with estimated cost.", ShoppingArgs, t_shopping),
        ToolSpec("production_plan", "Scale a recipe to N meals: cost, revenue, profit, margin.", ProductionArgs, t_production),
        ToolSpec("budget_check", "Check recipes against a budget in EGP.", BudgetArgs, t_budget),
        ToolSpec("pantry_add", "Add stock to the user's pantry.", PantryChangeArgs, t_pantry_add, writes=True),
        ToolSpec("pantry_use", "Take stock out of the pantry (fails without changes if not enough).", PantryUseArgs, t_pantry_use, writes=True),
        ToolSpec("pantry_list", "List the pantry.", EmptyArgs, t_pantry_list),
        ToolSpec("pantry_expiring", "Expired and soon-to-expire items.", PantryReadArgs, t_pantry_expiring),
        ToolSpec("profile_show", "Show the user's saved profile and saved allergies.", EmptyArgs, t_profile_show),
        ToolSpec("profile_update", "Save profile preferences and/or allergies the user asked to save.", ProfileChangeArgs, t_profile_update, writes=True),
        ToolSpec("profile_clear", "Delete saved allergies, profile and/or chat history (asks the user first).", ProfileClearArgs, t_profile_clear, writes=True, describe=d_profile_clear),
        ToolSpec("pantry_discard", "Throw away ALL the stock of an item (asks the user first).", PantryDiscardArgs, t_pantry_discard, writes=True, describe=d_pantry_discard),
        ToolSpec("pantry_adjust", "Set the stock of an item to an exact amount (asks the user first).", PantryAdjustArgs, t_pantry_adjust, writes=True, describe=d_pantry_adjust),
        ToolSpec("pantry_low_stock", "Items at or below their low-stock threshold.", EmptyArgs, t_pantry_low),
    ])
