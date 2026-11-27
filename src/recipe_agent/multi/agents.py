"""Specialist agents. Each one: (1) the LLM extracts typed arguments from the message, (2) the agent
calls its allowed tools, (3) code writes the answer (Arabic or English) from the tool results.

If the message lacks something needed (which recipe? how many meals?), the agent returns
needs_input with a question instead of guessing. If a tool fails, the agent says so plainly.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from .tools import (BudgetArgs, BudgetItem, ConvertArgs, CostArgs, PantryChangeArgs, PantryEntry,
                    PantryReadArgs, PantryUseArgs, ProductionArgs, RecipeRef, ShoppingArgs,
                    ShoppingRecipe, ToolContext, ToolRegistry, ToolResult, ProfileChangeArgs, ProfileClearArgs, EmptyArgs)
from recipe_agent.price_range import label_text, range_text
from recipe_agent.services import ExtraExpense, ProfileUpdate

log = logging.getLogger(__name__)

Intent = Literal["recipes", "pantry", "cost", "shopping", "business", "budget", "profile"]


# ---------------------------------------------------------------- routing
class RouteDecision(BaseModel):
    """Which specialist agents the message needs, in order (1 or 2)."""
    intents: list[Intent] = Field(
        min_length=1, max_length=2,
        description=("recipes: find / suggest dishes, meal plans, substitutions, allergies. "
                     "pantry: add / use / list stock, expiry, low stock. "
                     "cost: price or cost per serving of a dish, unit conversion. "
                     "shopping: shopping list. "
                     "business: produce N meals, selling price, profit, margin. "
                     "budget: does it fit a budget in EGP. "
                     "profile: save / show / delete the user's preferences (family size, favorite cuisines, "
                     "dislikes, diet, usual time limit, budget, business name), save or clear allergies on "
                     "request, show or delete the chat history."))
    language: Literal["ar", "en"] = "en"


ROUTER_PROMPT = """You route a message for a cooking assistant for Egyptian families and home food
businesses. The message may be Arabic (Egyptian dialect) or English. Pick the agent(s) needed, in the
order they must run. Use two only when the message clearly asks for two different jobs. Set language
to "ar" if the user wrote Arabic. If unsure, choose recipes."""

_KW = [
    ("profile", r"احفظي|احفظ |افتكري|انسي|امسحي|بروفايل|profile|remember|forget|save my|حجم العيلة|عدد افراد|عدد أفراد|family size|تفضيلات|preferences|clear (my )?history|مسح المحادثة"),
    ("budget", r"ميزاني|budget"),
    ("business", r"انتاج|إنتاج|production|ربح|profit|margin|هامش|سعر البيع|selling price|اجهز \d+|\d+ وجب"),
    ("shopping", r"قايمة شراء|قائمة شراء|قايمه شراء|shopping list|اشتري|المطلوب شراءه|shopping"),
    ("pantry", r"مخزن|pantry|خزين|صلاحي|expir|ضيف|أضيف|اضيف|استهلكت|استخدمت|low stock|stock"),
    ("cost", r"تكلف|كلف|بكام|سعر|cost|price|how much|حول|convert|كام جرام|grams"),
]


def keyword_topics(text: str) -> list:
    """Every topic whose keywords appear in the message, in priority order (no LLM)."""
    low = (text or "").lower()
    return [intent for intent, pat in _KW if re.search(pat, low)]


def keyword_route(text: str) -> RouteDecision:
    """Fallback router (no LLM): the first matching topic, else recipes."""
    lang = "ar" if re.search(r"[\u0600-\u06ff]", text or "") else "en"
    low = (text or "").lower()
    for intent, pat in _KW:
        if re.search(pat, low):
            return RouteDecision(intents=[intent], language=lang)
    return RouteDecision(intents=["recipes"], language=lang)


# ---------------------------------------------------------------- results and wording
@dataclass
class AgentResult:
    agent: str
    ok: bool = True
    answer: str = ""
    needs_input: bool = False
    data: Any = None
    tool_calls: list = field(default_factory=list)   # [(tool name, ok)]


def tr(lang, ar, en):
    return ar if lang == "ar" else en


def egp(v, lang):
    return f"{v:,.2f} " + tr(lang, "جنيه", "EGP")


STATUS_NOTE = {
    "ar": "الأرقام تقريبية (بعض الكميات أو الأسعار مقدّرة).",
    "en": "Numbers are approximate (some quantities or prices are estimated).",
}


def tool_failure_text(res: ToolResult, lang) -> tuple:
    """(text, needs_input) for a failed tool call."""
    code = res.error_code
    if code == "ambiguous":
        names = "، ".join(c["name"] for c in res.candidates) if lang == "ar" else ", ".join(c["name"] for c in res.candidates)
        return tr(lang, f"أي وصفة تقصدي؟ ({names})", f"Which recipe do you mean? ({names})"), True
    if code == "not_found":
        return tr(lang, "ملقتش الوصفة دي في الوصفات المصرية. اكتبي المكونات وكمياتها وأنا أحسبها.",
                  "I could not find that recipe in the Egyptian recipes. Send the ingredients and amounts and I will work it out."), True
    if code == "insufficient_stock":
        return tr(lang, f"المخزون مش كفاية، ومفيش حاجة اتغيرت. ({res.error})", f"Not enough in the pantry, nothing was changed. ({res.error})"), False
    if code == "bad_args":
        return tr(lang, f"ناقصني معلومة: {res.error}", f"I am missing something: {res.error}"), True
    if code == "unit":
        return tr(lang, f"معرفتش أحول الوحدة دي: {res.error}", f"I could not convert that unit: {res.error}"), False
    return tr(lang, "حصلت مشكلة وأنا بحسب. جربي تاني.", "Something went wrong while calculating. Please try again."), False


def with_context(text: str, ctx) -> str:
    """The message, preceded by the last few turns so that "it" / "that recipe" can be resolved."""
    hist = getattr(ctx, "history", None) or []
    if not hist:
        return text
    lines = [f"[{h.role}] {h.text[:300]}" for h in hist[-6:]]
    return ("Earlier messages (oldest first), only to resolve references such as 'it' or 'that recipe'. "
            "Extract values from the CURRENT message only.\n" + "\n".join(lines)
            + "\n\nCURRENT message:\n" + text)


class Specialist:
    name = ""
    tools: set = set()
    schema = None
    prompt = ""

    def __init__(self, registry: ToolRegistry, extract):
        self.registry, self.extract = registry, extract

    def call(self, tool, args, ctx, result: AgentResult) -> ToolResult:
        res = self.registry.call(tool, args, ctx, allowed=self.tools)
        result.tool_calls.append((tool, res.ok))
        return res

    def run(self, text: str, lang: str, ctx: ToolContext) -> AgentResult:
        result = AgentResult(self.name)
        try:
            args = self.extract(self.schema, self.prompt, with_context(text, ctx))
        except Exception as exc:
            log.warning("%s agent could not read the request: %s: %s", self.name, type(exc).__name__, exc)
            result.ok = False
            result.answer = tr(lang, "معرفتش أفهم الطلب دلوقتي (الخدمة مشغولة). جربي تاني بعد شوية.",
                               "I could not read the request right now (the service is busy). Please try again shortly.")
            return result
        try:
            self.handle(args, lang, ctx, result)
        except Exception as exc:
            log.exception("%s agent failed", self.name)
            result.ok = False
            result.answer = tr(lang, "حصلت مشكلة غير متوقعة.", "Something unexpected went wrong.")
        return result

    def fail(self, res, lang, result):
        result.ok = False
        result.answer, result.needs_input = tool_failure_text(res, lang)


# ---------------------------------------------------------------- recipes (existing pipeline)
class RecipesAgent(Specialist):
    """Delegates to the existing, tested recipe pipeline (allergies, memory, Egyptian recipes, verifier)."""
    name = "recipes"

    def __init__(self, registry, extract, pipeline=None):
        super().__init__(registry, extract)
        self.pipeline = pipeline

    def run(self, text, lang, ctx):
        result = AgentResult(self.name)
        pipeline = self.pipeline
        if pipeline is None:
            from recipe_agent.graph import answer as pipeline  # heavy imports only when used
        try:
            result.answer = pipeline(text, ctx.user_id)
        except Exception as exc:
            log.exception("recipe pipeline failed")
            result.ok = False
            from recipe_agent.llm_call import is_transient
            if is_transient(exc):   # quota (429) or overloaded model (503)
                result.answer = tr(lang, "الخدمة مشغولة أو وصلت لحد الاستخدام المجاني مؤقتاً. جرّبي تاني بعد شوية.",
                                   "The service is busy or the free quota is used up for now. Please try again shortly.")
            else:
                result.answer = tr(lang, "معرفتش أجيب وصفات دلوقتي.", "I could not fetch recipes right now.")
        return result


# ---------------------------------------------------------------- pantry
class PantryRequest(BaseModel):
    action: Literal["add", "use", "list", "expiring", "low_stock"]
    items: list[PantryEntry] = Field(default_factory=list, description="For add / use. Quantity and unit as the user said.")
    within_days: int = Field(default=3, description="For expiring: days ahead.")


class PantryAgent(Specialist):
    name = "pantry"
    tools = {"pantry_add", "pantry_use", "pantry_list", "pantry_expiring", "pantry_low_stock"}
    schema = PantryRequest
    prompt = ("Extract the pantry action. add = the user bought / has more; use = the user used / cooked "
              "with it; list = show the pantry; expiring = what expires soon; low_stock = what is running out. "
              "Copy quantities and units exactly as written; never invent a quantity.")

    @staticmethod
    def _rows(views, lang):
        return "\n".join(f"- {v.name}: {v.quantity:g} {v.unit}" + (f" ({tr(lang, 'ينتهي', 'expires')} {v.expiry})" if v.expiry else "")
                         for v in views)

    def handle(self, a: PantryRequest, lang, ctx, result):
        if a.action in ("add", "use"):
            if not a.items:
                result.needs_input = True
                result.answer = tr(lang, "إيه الأصناف والكميات؟", "Which items and how much?")
                return
            tool = "pantry_add" if a.action == "add" else "pantry_use"
            model = PantryChangeArgs if a.action == "add" else PantryUseArgs
            res = self.call(tool, model(items=a.items), ctx, result)
            if not res.ok:
                return self.fail(res, lang, result)
            result.data = res.data
            head = tr(lang, "تمام، ضفت للمخزن:" if a.action == "add" else "تمام، خصمت من المخزن:",
                      "Added to the pantry:" if a.action == "add" else "Taken from the pantry:")
            tail = tr(lang, "الرصيد دلوقتي:", "Now in stock:")
            result.answer = f"{head}\n" + "\n".join(f"- {e.name} {e.quantity:g} {e.unit or ''}".rstrip() for e in a.items) \
                + f"\n{tail}\n{self._rows(res.data, lang)}"
        elif a.action == "list":
            res = self.call("pantry_list", {}, ctx, result)
            if not res.ok:
                return self.fail(res, lang, result)
            result.data = res.data
            result.answer = self._rows(res.data, lang) if res.data else tr(lang, "المخزن فاضي.", "The pantry is empty.")
        elif a.action == "expiring":
            res = self.call("pantry_expiring", PantryReadArgs(within_days=a.within_days), ctx, result)
            if not res.ok:
                return self.fail(res, lang, result)
            result.data = res.data
            d = res.data
            parts = []
            if d["expired"]:
                parts.append(tr(lang, "منتهي الصلاحية:", "Expired:") + "\n" + self._rows(d["expired"], lang))
            if d["expiring_soon"]:
                parts.append(tr(lang, f"هينتهي خلال {d['within_days']} أيام:", f"Expiring within {d['within_days']} days:")
                             + "\n" + self._rows(d["expiring_soon"], lang))
            result.answer = "\n".join(parts) or tr(lang, "مفيش حاجة قربت تنتهي.", "Nothing is close to expiring.")
        else:
            res = self.call("pantry_low_stock", {}, ctx, result)
            if not res.ok:
                return self.fail(res, lang, result)
            result.data = res.data
            result.answer = (tr(lang, "قربت تخلص:", "Running low:") + "\n" + self._rows(res.data, lang)) if res.data \
                else tr(lang, "مفيش حاجة قاربت تخلص.", "Nothing is running low.")


# ---------------------------------------------------------------- cost and units
class CostRequest(BaseModel):
    task: Literal["cost", "convert"]
    recipe: Optional[RecipeRef] = None
    extra_expenses: list[ExtraExpense] = Field(default_factory=list, description="Only extras the user gave with an amount.")
    convert: Optional[ConvertArgs] = None


class CostAgent(Specialist):
    name = "cost"
    tools = {"recipe_cost", "convert_units"}
    schema = CostRequest
    prompt = ("Extract a cost request (task=cost: cost / cost per serving of a dish) or a unit conversion "
              "(task=convert). For a dish give its name as the user wrote it; give ingredients only when the "
              "user listed them with amounts. Never invent servings, prices or extras.")

    def handle(self, a: CostRequest, lang, ctx, result):
        if a.task == "convert":
            if not a.convert:
                result.needs_input = True
                result.answer = tr(lang, "عايزة تحولي كام من إيه؟", "How much of what should I convert?")
                return
            res = self.call("convert_units", a.convert, ctx, result)
            if not res.ok:
                return self.fail(res, lang, result)
            m = res.data
            result.data = m
            if m.value is None:
                result.answer = tr(lang, "معرفتش أحول الوحدة دي بشكل مؤكد.", "I could not convert that reliably.")
            else:
                note = "" if m.status == "verified" else " " + tr(lang, "(تقريبي)", "(approximate)")
                result.answer = f"{a.convert.quantity:g} {a.convert.unit or ''} {a.convert.ingredient} = {m.value:g} {m.unit}{note}"
            return
        if not a.recipe:
            result.needs_input = True
            result.answer = tr(lang, "تكلفة أنهي وصفة؟", "Which recipe should I cost?")
            return
        res = self.call("recipe_cost", CostArgs(recipe=a.recipe, extra_expenses=a.extra_expenses), ctx, result)
        if not res.ok:
            return self.fail(res, lang, result)
        c = res.data
        result.data = c
        result.answer = format_cost(c, lang)


def format_cost(c, lang) -> str:
    if c.total_egp is None:
        out = [tr(lang, f"مقدرتش أحسب تكلفة {c.recipe}: مفيش أسعار للمكونات.", f"I could not cost {c.recipe}: no ingredient prices.")]
    else:
        if c.low_egp is not None:
            rng = {"label_key": c.price_label, "stale": c.stale_prices}
            total = f"{range_text(c.low_egp, c.high_egp, lang)} ({label_text(rng, lang)})"
        else:
            total = egp(c.total_egp, lang)
        out = [tr(lang, f"تكلفة {c.recipe}: {total}", f"Cost of {c.recipe}: {total}")]
        if c.per_serving_egp is not None:
            per = range_text(c.per_serving_low_egp, c.per_serving_high_egp, lang) if c.per_serving_low_egp is not None else egp(c.per_serving_egp, lang)
            out.append(tr(lang, f"للفرد: {per} ({c.servings:g} أفراد)", f"Per serving: {per} ({c.servings:g} servings)"))
        if c.extras_egp:
            out.append(tr(lang, f"منها مصاريف إضافية: {egp(c.extras_egp, lang)}", f"Includes extra expenses: {egp(c.extras_egp, lang)}"))
    if c.unpriced:
        out.append(tr(lang, "من غير سعر (مش محسوبة): ", "No price (not counted): ") + ", ".join(c.unpriced))
    if not c.complete:
        out.append(tr(lang, "الإجمالي ناقص لأن فيه مكونات من غير سعر.", "The total is incomplete because some items have no price."))
    out.append(STATUS_NOTE[lang])
    return "\n".join(out)


# ---------------------------------------------------------------- shopping
class ShoppingRequest(BaseModel):
    recipes: list[ShoppingRecipe] = Field(default_factory=list)
    use_pantry: bool = Field(default=True, description="Subtract what is already in the pantry.")


CATEGORY_AR = {"produce": "خضار وفاكهة", "meat_poultry": "لحوم وفراخ", "seafood": "سمك", "dairy_eggs": "ألبان وبيض",
               "grains_legumes": "حبوب وبقوليات", "bakery": "مخبوزات", "spices": "بهارات", "oils_fats": "زيوت وسمن",
               "condiments_sweets": "صوصات وحلويات", "non_food": "غير غذائي", "other": "أخرى"}


class ShoppingAgent(Specialist):
    name = "shopping"
    tools = {"shopping_list"}
    schema = ShoppingRequest
    prompt = ("Extract the dishes to shop for. Give each dish name as written; meals only if the user states "
              "how many meals. use_pantry is true unless the user says to ignore the pantry.")

    def handle(self, a: ShoppingRequest, lang, ctx, result):
        if not a.recipes:
            result.needs_input = True
            result.answer = tr(lang, "قايمة الشراء لأنهي وصفات؟", "Which recipes is the shopping list for?")
            return
        res = self.call("shopping_list", ShoppingArgs(recipes=a.recipes, use_pantry=a.use_pantry), ctx, result)
        if not res.ok:
            return self.fail(res, lang, result)
        sl = res.data
        result.data = sl
        result.answer = format_shopping(sl, lang)


def format_shopping(sl, lang) -> str:
    out, cat = [], None
    for i in sl.items:
        if i.category != cat:
            cat = i.category
            out.append(f"\n{CATEGORY_AR.get(cat, cat) if lang == 'ar' else cat.replace('_', ' ')}:")
        if i.to_taste:
            out.append(f"- {i.name} ({tr(lang, 'حسب الذوق', 'to taste')})")
        elif i.needed is None:
            out.append(f"- {i.name} ({tr(lang, 'الكمية غير معروفة', 'quantity unknown')})")
        elif not i.to_buy:
            out.append(f"- {i.name}: {tr(lang, 'موجود في المخزن', 'covered by pantry')}")
        else:
            guess = tr(lang, " (تقديري)", " (estimated)") if i.price_source == "estimated" else ""
            cost = f", ~{egp(i.est_cost_egp, lang)}{guess}" if i.est_cost_egp is not None else f", {tr(lang, 'من غير سعر', 'no price')}"
            out.append(f"- {i.name}: {i.to_buy:g} {i.unit}{cost}")
    if sl.low_egp is not None:
        rng = {"label_key": sl.price_label, "stale": sl.stale_prices}
        shown = f"{range_text(sl.low_egp, sl.high_egp, lang)} ({label_text(rng, lang)})"
        out.append("\n" + tr(lang, f"الإجمالي المعروف: {shown}", f"Known total: {shown}"))
    elif sl.estimated_total_egp is not None:
        out.append("\n" + tr(lang, f"الإجمالي المعروف: {egp(sl.estimated_total_egp, lang)}", f"Known total: {egp(sl.estimated_total_egp, lang)}"))
    if not sl.total_complete:
        out.append(tr(lang, "الإجمالي ناقص: فيه أصناف من غير سعر أو كمية.", "The total is incomplete: some items have no price or quantity."))
    out.append(STATUS_NOTE[lang])
    return "\n".join(out).strip()


# ---------------------------------------------------------------- business (production, profit)
class BusinessRequest(BaseModel):
    recipe: Optional[RecipeRef] = None
    meals: Optional[int] = Field(default=None, description="Number of meals to produce.")
    selling_price_per_meal: Optional[float] = None
    extra_expenses: list[ExtraExpense] = Field(default_factory=list)
    target_margin_pct: Optional[float] = Field(default=None, description="Wanted margin in percent, if asked.")
    use_pantry: bool = False


class BusinessAgent(Specialist):
    name = "business"
    tools = {"production_plan"}
    schema = BusinessRequest
    prompt = ("Extract a production request: the dish, how many meals, the selling price per meal, extra "
              "expenses with amounts (packaging, gas...), and a target margin if asked. Leave anything not "
              "stated as null. Never invent a price or an expense.")

    def handle(self, a: BusinessRequest, lang, ctx, result):
        if not a.recipe:
            result.needs_input = True
            result.answer = tr(lang, "هتنتجي أنهي وصفة؟", "Which recipe are you producing?")
            return
        if not a.meals:
            result.needs_input = True
            result.answer = tr(lang, "عايزة تنتجي كام وجبة؟", "How many meals do you want to produce?")
            return
        res = self.call("production_plan", ProductionArgs(
            recipe=a.recipe, meals=a.meals, selling_price_per_meal=a.selling_price_per_meal,
            extra_expenses=a.extra_expenses, target_margin_pct=a.target_margin_pct,
            use_pantry=a.use_pantry), ctx, result)
        if not res.ok:
            return self.fail(res, lang, result)
        result.data = res.data
        result.answer = format_production(res.data, lang)


def format_production(d, lang) -> str:
    p = d["plan"]
    out = [tr(lang, f"خطة إنتاج {p.meals} وجبة من {p.recipe}", f"Production plan: {p.meals} meals of {p.recipe}")]
    if p.scale_factor is None:
        out += p.notes
        return "\n".join(out)
    out.append(tr(lang, f"مضاعفة الوصفة ×{p.scale_factor:g}", f"Recipe scaled ×{p.scale_factor:g}"))
    if p.total_cost_egp is not None:
        out.append(tr(lang, f"التكلفة الكلية: {egp(p.total_cost_egp, lang)}", f"Total cost: {egp(p.total_cost_egp, lang)}"))
        out.append(tr(lang, f"تكلفة الوجبة: {egp(p.cost_per_meal_egp, lang)}", f"Cost per meal: {egp(p.cost_per_meal_egp, lang)}"))
    if p.revenue_egp is not None:
        out.append(tr(lang, f"الإيراد: {egp(p.revenue_egp, lang)}", f"Revenue: {egp(p.revenue_egp, lang)}"))
    if p.gross_profit_egp is not None:
        out.append(tr(lang, f"الربح: {egp(p.gross_profit_egp, lang)} (هامش {p.margin_pct:g}%)",
                      f"Profit: {egp(p.gross_profit_egp, lang)} (margin {p.margin_pct:g}%)"))
    elif p.revenue_egp is not None:
        out.append(tr(lang, "مقدرتش أحسب الربح لأن التكلفة ناقصة.", "Profit not shown because the cost is incomplete."))
    if d.get("suggested_price_egp") is not None:
        out.append(tr(lang, f"أقل سعر للوجبة لهامش {d['target_margin_pct']:g}%: {egp(d['suggested_price_egp'], lang)}",
                      f"Lowest price per meal for a {d['target_margin_pct']:g}% margin: {egp(d['suggested_price_egp'], lang)}"))
    elif d.get("target_margin_pct") is not None:
        out.append(tr(lang, "مقدرتش أقترح سعر لأن التكلفة ناقصة.", "No price suggestion because the cost is incomplete."))
    if p.unpriced:
        out.append(tr(lang, "من غير سعر (مش محسوبة): ", "No price (not counted): ") + ", ".join(p.unpriced))
    if p.missing_quantities:
        out.append(tr(lang, "كمية غير معروفة: ", "Quantity unknown: ") + ", ".join(p.missing_quantities))
    if p.shopping is not None:
        out.append("\n" + tr(lang, "المطلوب شراءه:", "To buy:") + "\n" + format_shopping(p.shopping, lang))
    out.append(STATUS_NOTE[lang])
    return "\n".join(out)


# ---------------------------------------------------------------- budget
class BudgetRequest(BaseModel):
    items: list[BudgetItem] = Field(default_factory=list)
    budget_egp: Optional[float] = None
    use_pantry: bool = False


class BudgetAgent(Specialist):
    name = "budget"
    tools = {"budget_check", "profile_show"}
    schema = BudgetRequest
    prompt = ("Extract the dishes (with number of meals each) and the budget in EGP. use_pantry is true only if "
              "the user says to count what they already have. Leave unstated values null.")

    def handle(self, a: BudgetRequest, lang, ctx, result):
        saved_note = ""
        if a.budget_egp is None:
            prof = self.call("profile_show", EmptyArgs(), ctx, result)
            if prof.ok and prof.data.budget_egp is not None:
                a.budget_egp = prof.data.budget_egp
                per = {"daily": tr(lang, "اليومية", "daily"), "weekly": tr(lang, "الأسبوعية", "weekly"),
                       "monthly": tr(lang, "الشهرية", "monthly")}.get(prof.data.budget_period or "", "")
                saved_note = tr(lang, f"(استخدمت ميزانيتك المحفوظة {per}: {egp(a.budget_egp, lang)})",
                                f"(used your saved {per} budget: {egp(a.budget_egp, lang)})").replace("  ", " ")
        if a.budget_egp is None:
            result.needs_input = True
            result.answer = tr(lang, "ميزانيتك كام جنيه؟", "What is your budget in EGP?")
            return
        if not a.items:
            result.needs_input = True
            result.answer = tr(lang, "الميزانية دي لأنهي وجبات؟", "Which meals is the budget for?")
            return
        res = self.call("budget_check", BudgetArgs(items=a.items, budget_egp=a.budget_egp, use_pantry=a.use_pantry), ctx, result)
        if not res.ok:
            return self.fail(res, lang, result)
        b = res.data
        result.data = b
        known = egp(b.known_total_egp, lang) if b.known_total_egp is not None else tr(lang, "غير معروف", "unknown")
        out = [tr(lang, f"الميزانية: {egp(b.budget_egp, lang)} | المعروف من التكلفة: {known}",
                  f"Budget: {egp(b.budget_egp, lang)} | Known cost: {known}")]
        if b.verdict == "within_budget":
            out.append(tr(lang, f"داخل الميزانية، والباقي {egp(b.remaining_egp, lang)}.", f"Within budget, {egp(b.remaining_egp, lang)} left."))
        elif b.verdict == "over_budget":
            out.append(tr(lang, f"فوق الميزانية بـ {egp(b.over_by_egp, lang)}.", f"Over budget by {egp(b.over_by_egp, lang)}."))
        else:
            out.append(tr(lang, "مقدرش أحكم: فيه أسعار ناقصة، فالتكلفة الحقيقية ممكن تكون أعلى.",
                          "Cannot tell: some prices are missing, so the real cost may be higher."))
        if b.unpriced:
            out.append(tr(lang, "من غير سعر: ", "No price: ") + ", ".join(b.unpriced))
        out.append(STATUS_NOTE[lang])
        if saved_note:
            out.insert(0, saved_note)
        result.answer = "\n".join(out)


# ---------------------------------------------------------------- profile and history
class ProfileRequest(BaseModel):
    action: Literal["show", "update", "clear"]
    update: Optional[ProfileUpdate] = Field(default=None, description="Preferences to save, only what the user stated.")
    allergies: list[str] = Field(default_factory=list,
                                 description="Allergies the user asks to SAVE / remember, as short English food words (milk, egg, peanut...).")
    clear_what: Optional[Literal["allergies", "profile", "history", "all"]] = Field(
        default=None, description="For action=clear: what the user explicitly asked to delete.")


class ProfileAgent(Specialist):
    name = "profile"
    tools = {"profile_show", "profile_update", "profile_clear"}
    schema = ProfileRequest
    prompt = ("Extract a profile request. show = display what is saved. update = the user asks to save "
              "preferences or allergies (family size, favorite cuisines, disliked ingredients, diet, usual "
              "time limit in minutes, budget with its period, business name/type, mode household or business). "
              "clear = the user explicitly asks to delete saved allergies, the profile or the chat history. "
              "Only extract what the user states; never invent a value. Allergies only when asked to save them.")

    @staticmethod
    def _fmt(p, lang) -> str:
        def lst(x):
            return "، ".join(x) if lang == "ar" and x else (", ".join(x) if x else tr(lang, "غير محدد", "not set"))

        def val(x, suffix=""):
            return f"{x:g}{suffix}" if isinstance(x, (int, float)) else (x or tr(lang, "غير محدد", "not set"))
        allergies = (tr(lang, "تعذّرت قراءة الحساسيات المحفوظة", "could not read the saved allergies")
                     if p.allergies is None else lst(p.allergies))
        rows = [
            (tr(lang, "الحساسيات", "Allergies"), allergies),
            (tr(lang, "الوضع", "Mode"), tr(lang, "بيزنس" if p.mode == "business" else "بيت", p.mode)),
            (tr(lang, "عدد أفراد العيلة", "Family size"), val(p.family_size)),
            (tr(lang, "المطابخ المفضلة", "Favorite cuisines"), lst(p.favorite_cuisines)),
            (tr(lang, "مكونات مش بتحبيها", "Disliked ingredients"), lst(p.disliked_ingredients)),
            (tr(lang, "نظام غذائي", "Diet"), lst(p.diet)),
            (tr(lang, "الوقت المفضل (دقيقة)", "Preferred time (min)"), val(p.max_minutes)),
            (tr(lang, "الميزانية", "Budget"),
             (f"{p.budget_egp:g} " + tr(lang, "جنيه", "EGP") + (f" ({p.budget_period})" if p.budget_period else ""))
             if p.budget_egp is not None else tr(lang, "غير محدد", "not set")),
        ]
        if p.business_name or p.business_type or p.mode == "business":
            rows += [(tr(lang, "اسم البيزنس", "Business name"), val(p.business_name)),
                     (tr(lang, "نوع البيزنس", "Business type"), val(p.business_type))]
        return "\n".join(f"- {k}: {v}" for k, v in rows)

    def handle(self, a: ProfileRequest, lang, ctx, result):
        if a.action == "show":
            res = self.call("profile_show", EmptyArgs(), ctx, result)
            if not res.ok:
                return self.fail(res, lang, result)
            result.data = res.data
            result.answer = tr(lang, "اللي محفوظ عندي:", "What I have saved:") + "\n" + self._fmt(res.data, lang)
            return
        if a.action == "update":
            allergies = list(a.allergies)
            try:  # the allergy words in the message are added by code, whatever the LLM extracted
                from recipe_agent.allergy import extract_allergies
                if allergies:
                    allergies = list(dict.fromkeys([*allergies, *extract_allergies(ctx.message)]))
            except Exception:
                log.exception("allergy extraction skipped")
            res = self.call("profile_update", ProfileChangeArgs(update=a.update, allergies=allergies), ctx, result)
            if not res.ok:
                if res.error_code == "bad_args":
                    result.needs_input = True
                    result.answer = tr(lang, "قوليلي إيه اللي أحفظه (مثلاً: عدد أفراد العيلة 5).",
                                       "Tell me what to save (for example: family size 5).")
                    return
                return self.fail(res, lang, result)
            result.data = res.data
            result.answer = tr(lang, "تمام، حفظت. دلوقتي عندي:", "Saved. I now have:") + "\n" + self._fmt(res.data, lang)
            return
        if not a.clear_what:
            result.needs_input = True
            result.answer = tr(lang, "عايزة أمسح إيه بالظبط: الحساسيات، البروفايل، ولا المحادثات؟",
                               "What exactly should I delete: allergies, profile or chat history?")
            return
        res = self.call("profile_clear", ProfileClearArgs(what=a.clear_what), ctx, result)
        if not res.ok:
            return self.fail(res, lang, result)
        result.data = res.data
        names = {"allergies": tr(lang, "الحساسيات", "allergies"), "profile": tr(lang, "البروفايل", "profile"),
                 "history_messages": tr(lang, "المحادثات", "chat history")}
        done = ", ".join(names[k] for k in res.data) if lang != "ar" else "، ".join(names[k] for k in res.data)
        result.answer = tr(lang, f"تمام، مسحت: {done}.", f"Done, deleted: {done}.")
