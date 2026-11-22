"""Gradio chat UI for the Recipe Agent (Arabic / English)."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr

from recipe_agent import graph

TITLE = "Recipe Agent | مساعد الوصفات"

DESCRIPTION = """
اكتبي المكونات اللي عندك (بالعربي أو بالإنجليزي) ويمكن تضيفي وقت أو حساسية.
Tell me what ingredients you have, optionally a time limit or an allergy.

لو عايزة أفتكر حساسيتك للمرات الجاية قولي: "افتكري إني عندي حساسية من ..." (ولمسحها: "انسي الحساسيات").
To save an allergy say: "remember that I'm allergic to ..." (to clear: "forget my allergies").

**تنبيه / Note:** فلتر الحساسية بيعتمد على أسماء المكونات وخطوات الوصفة فقط،
راجعي ملصقات المنتجات دايماً. قوايم المكونات في الداتا ممكن تبقى ناقصة أحياناً،
فاقري الخطوات كويس. The allergy filter checks ingredient names and recipe steps only;
always check product labels. Ingredient lists in the dataset can be incomplete, so read
the steps too. Times and nutrition values come from the recipe dataset and are approximate.
"""

EXAMPLES = [
    "عندي فراخ وأرز وثوم وعايزة أكلة تخلص في نص ساعة",
    "عندي بطاطس وبيض وعندي حساسية من اللبن",
    "I have eggs, tomatoes and cheese. Under 20 minutes.",
    "I have salmon and lemon. I'm allergic to nuts.",
]

QUOTA_MESSAGE = (
    "الخدمة وصلت لحد الاستخدام المجاني مؤقتاً. جرّبي تاني بعد شوية.\n\n"
    "The free API quota is exhausted for now. Please try again later."
)

BUSY_MESSAGE = (
    "السيرفر مشغول دلوقتي. جرّبي تاني بعد دقيقة.\n\n"
    "The model server is busy right now. Please try again in a minute."
)


def _is_quota(text: str) -> bool:
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _is_busy(text: str) -> bool:
    return any(key in text for key in ("503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED"))


def respond(message, history, request: gr.Request):
    """Each message is handled on its own; saved allergies come from the memory file.

    Without a login, everyone who opens the app shares the "default" profile.
    """
    user_id = getattr(request, "username", None) or "default"
    for attempt in range(2):
        try:
            return graph.answer(message, user_id=user_id)
        except Exception as exc:
            text = str(exc)
            if _is_quota(text):
                return QUOTA_MESSAGE
            if _is_busy(text):
                if attempt == 0:
                    time.sleep(5)  # one retry only: every retry may use free-tier quota
                    continue
                return BUSY_MESSAGE
            return f"حصلت مشكلة غير متوقعة / Unexpected error: {type(exc).__name__}"
    return BUSY_MESSAGE


demo = gr.ChatInterface(
    fn=respond,
    title=TITLE,
    description=DESCRIPTION,
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.launch()
