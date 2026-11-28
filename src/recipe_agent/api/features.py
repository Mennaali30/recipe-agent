"""API feature endpoints: pantry, recipes, cost, shopping, budget, business, pending confirmations.

No new logic: each endpoint calls the same validated tool the agents use (multi/tools.py), so the numbers
and rules are identical to the chat. A tool failure becomes the standard error body with the right status.
The user is always the logged-in user (see server.who). The UI's own buttons are the user's explicit
action, so these endpoints run the action directly (the "confirm first" step is only for chat).
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.encoders import jsonable_encoder

from recipe_agent import egypt_store, tools as recipe_tools
from recipe_agent.multi.tools import (BudgetArgs, ConvertArgs, CostArgs, EmptyArgs, PantryAdjustArgs,
                                      PantryChangeArgs, PantryDiscardArgs, PantryReadArgs, PantryUseArgs,
                                      ProductionArgs, ShoppingArgs, ToolContext, build_registry)
from recipe_agent.services import ApprovalService

log = logging.getLogger(__name__)
STATUS = {"bad_args": 422, "unit": 422, "not_found": 404, "ambiguous": 409, "insufficient_stock": 409,
          "not_allowed": 403, "needs_approval": 409, "failed": 500}


def build_router(who, session_factory, memory=None, registry=None, auth=None, today=None) -> APIRouter:
    """who(requested_user_id, authorization_header) -> the user id for this request."""
    registry = registry or build_registry(approval_tools=())
    router = APIRouter(dependencies=auth or [])
    sf = session_factory

    def user_dep(authorization: Optional[str] = Header(default=None)) -> str:
        """Runs before the body is read: no valid login = 401, whatever the body is."""
        return who(None, authorization)

    def ctx_for(uid):
        return ToolContext(user_id=uid, session_factory=lambda: sf()(), memory=memory, approved=True, today=today)

    def run(tool, args, uid):
        res = registry.call(tool, args, ctx_for(uid))
        if not res.ok:
            exc = HTTPException(STATUS.get(res.error_code, 400), res.error or "request failed")
            if res.candidates:
                exc.candidates = jsonable_encoder(res.candidates)
            raise exc
        return jsonable_encoder(res.data)

    def saved_exclusions(uid):
        try:
            return list(memory.get_exclusions(uid)) if memory is not None else _default_memory().get_exclusions(uid)
        except Exception:
            log.exception("saved allergies could not be read")
            raise HTTPException(503, "saved allergies could not be read, so no recipe is shown")

    def _default_memory():
        from recipe_agent import memory as mem
        return mem

    # ---------------- pantry ----------------
    @router.get("/pantry")
    def pantry_list(uid: str = Depends(user_dep)):
        return run("pantry_list", EmptyArgs(), uid)

    @router.get("/pantry/expiring")
    def pantry_expiring(within_days: int = Query(3, ge=0, le=60), uid: str = Depends(user_dep)):
        return run("pantry_expiring", PantryReadArgs(within_days=within_days), uid)

    @router.get("/pantry/low-stock")
    def pantry_low(uid: str = Depends(user_dep)):
        return run("pantry_low_stock", EmptyArgs(), uid)

    @router.post("/pantry/items")
    def pantry_add(body: PantryChangeArgs, uid: str = Depends(user_dep)):
        return run("pantry_add", body, uid)

    @router.post("/pantry/use")
    def pantry_use(body: PantryUseArgs, uid: str = Depends(user_dep)):
        return run("pantry_use", body, uid)

    @router.put("/pantry/items")
    def pantry_set(body: PantryAdjustArgs, uid: str = Depends(user_dep)):
        return run("pantry_adjust", body, uid)

    @router.post("/pantry/discard")
    def pantry_discard(body: PantryDiscardArgs, uid: str = Depends(user_dep)):
        return run("pantry_discard", body, uid)

    # ---------------- cost, units, shopping, budget, business ----------------
    @router.post("/cost")
    def cost(body: CostArgs, uid: str = Depends(user_dep)):
        return run("recipe_cost", body, uid)

    @router.post("/convert")
    def convert(body: ConvertArgs, uid: str = Depends(user_dep)):
        return run("convert_units", body, uid)

    @router.post("/shopping")
    def shopping(body: ShoppingArgs, uid: str = Depends(user_dep)):
        return run("shopping_list", body, uid)

    @router.post("/budget/check")
    def budget(body: BudgetArgs, uid: str = Depends(user_dep)):
        return run("budget_check", body, uid)

    @router.post("/business/production-plan")
    def production_plan(body: ProductionArgs, uid: str = Depends(user_dep)):
        return run("production_plan", body, uid)

    # ---------------- recipes (the user's saved allergies are ALWAYS applied) ----------------
    @router.get("/recipes/search")
    def recipes_search(ingredients: str = Query(..., min_length=1, description="comma separated"),
                       max_minutes: Optional[int] = Query(None, ge=1, le=600),
                       exclude: str = Query("", description="comma separated, added to the saved allergies"),
                       limit: int = Query(5, ge=1, le=20), lang: str = Query("ar", pattern="^(ar|en)$"),
                       uid: str = Depends(user_dep)):
        ings = [x.strip() for x in ingredients.split(",") if x.strip()]
        excl = list(dict.fromkeys([*saved_exclusions(uid), *[x.strip() for x in exclude.split(",") if x.strip()]]))
        international = True
        try:
            found = recipe_tools.find_recipes(ings, max_minutes=max_minutes, exclude=excl, top_k=limit, lang=lang)
        except Exception:     # the English index is not available: Egyptian recipes only, and say so
            log.exception("international recipe search unavailable")
            international = False
            found = egypt_store.find(ings, max_minutes=max_minutes, exclude=excl, top_k=limit, lang=lang)
        return jsonable_encoder({"recipes": found, "applied_exclusions": excl, "international_available": international})

    @router.get("/recipes/by-name")
    def recipes_by_name(q: str = Query(..., min_length=1, max_length=100), max_minutes: Optional[int] = Query(None, ge=1, le=600),
                        limit: int = Query(3, ge=1, le=10), lang: str = Query("ar", pattern="^(ar|en)$"),
                        uid: str = Depends(user_dep)):
        excl = saved_exclusions(uid)
        found, blocked = egypt_store.find_by_name(q, exclude=excl, max_minutes=max_minutes, top_k=limit, lang=lang)
        out = {"recipes": found, "blocked_by_allergy": blocked, "egyptian": bool(found), "applied_exclusions": excl}
        if not found and not blocked:
            query = egypt_store.dish_query(q)
            try:
                out["recipes"] = recipe_tools.find_international_by_name(query, max_minutes=max_minutes, exclude=excl, top_k=limit)
            except Exception:
                log.exception("international dish lookup unavailable")
            out["note"] = "no_egyptian_recipe" if query else "no_dish_name"
        return jsonable_encoder(out)

    @router.get("/recipes/{recipe_id}")
    def recipe_detail(recipe_id: str, lang: str = Query("ar", pattern="^(ar|en)$"),
                      uid: str = Depends(user_dep)):
        if not egypt_store.is_egypt_id(recipe_id):
            raise HTTPException(404, "only Egyptian recipe ids (egy:...) can be opened here")
        try:
            e = egypt_store.get_recipe(recipe_id)
        except KeyError:
            raise HTTPException(404, f"unknown recipe {recipe_id}")
        blocked = egypt_store.excluded(e, saved_exclusions(uid))
        if blocked:
            raise HTTPException(409, "this recipe contains something in your saved allergies, so it is not shown")
        return jsonable_encoder({
            "recipe_id": e["recipe_id"], "name": e["name"], "name_en": e["name_en"], "category": e["category"],
            "servings": e["servings"], "total_minutes": e["total_minutes"], "nutrition": e["nutrition"],
            "ingredients": [{"name": n, "amount": ln.get("amount_text"), "cost_egp": ln.get("cost_egp")}
                            for n, ln in zip(e["ingredients"], e["cost"]["lines"])],
            "steps": egypt_store.instructions(recipe_id, lang),
            "cost_text": egypt_store.format_cost(e["cost"], lang),
            "cost": {k: e["cost"].get(k) for k in ("low_egp", "high_egp", "price_label", "total_egp", "per_serving_egp",
                                                   "coverage", "unpriced", "approximate", "currency")},
        })

    # ---------------- pending confirmations (the chat asks "yes/no"; these are the buttons) ----------------
    @router.get("/pending")
    def pending(uid: str = Depends(user_dep)):
        with sf()() as s:
            p = ApprovalService(s).peek(uid)
        return {"pending": None if p is None else {"tool": p.tool, "summary_ar": p.summary_ar, "summary_en": p.summary_en}}

    @router.post("/pending/confirm")
    def pending_confirm(uid: str = Depends(user_dep)):
        ctx = ctx_for(uid)
        with sf()() as s:
            p = ApprovalService(s).take(uid)
        if p is None:
            raise HTTPException(404, "nothing is waiting for confirmation (it may have expired)")
        res = registry.call(p.tool, p.args, ctx, allowed={p.tool})
        if not res.ok:
            raise HTTPException(STATUS.get(res.error_code, 400), res.error or "could not do it")
        return {"done": True, "tool": p.tool}

    @router.post("/pending/cancel")
    def pending_cancel(uid: str = Depends(user_dep)):
        with sf()() as s:
            p = ApprovalService(s).take(uid)
        return {"cancelled": p is not None}

    return router
