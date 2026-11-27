"""Entry point used by the Gradio app: one supervisor per process, built on first use."""
import os
from functools import lru_cache

from .supervisor import build_supervisor


@lru_cache(maxsize=1)
def get_supervisor():
    # hybrid router: keywords decide when they can, so most messages cost no extra LLM call
    return build_supervisor(router=os.getenv("RECIPE_AGENT_ROUTER", "hybrid"))


def answer(text: str, user_id: str = "default") -> str:
    """Answer one message through the supervisor (profile, history, pantry, cost... included)."""
    return get_supervisor().invoke({"text": text, "user_id": user_id})["answer"]
