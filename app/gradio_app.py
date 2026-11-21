"""Gradio chat UI for the Recipe Agent (Arabic / English)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr

from recipe_agent import graph

TITLE = "Recipe Agent | مساعد الوصفات"

DESCRIPTION = """
اكتبي المكونات اللي عندك (بالعربي أو بالإنجليزي) ويمكن تضيفي وقت أو حساسية.
Tell me what ingredients you have, optionally a time limit or an allergy.

**تنبيه / Note:** فلتر الحساسية بيعتمد على أسماء المكونات وخطوات الوصفة فقط،
راجعي ملصقات المنتجات دايماً. The allergy filter checks ingredient names and recipe
steps only; always check product labels. Times and nutrition values come from the
recipe dataset and are approximate.
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


def respond(message, history):
    """Each message is handled on its own (the chat history is not used)."""
    try:
        return graph.answer(message)
    except Exception as exc:
        text = str(exc)
        if "429" in text or "RESOURCE_EXHAUSTED" in text:
            return QUOTA_MESSAGE
        return f"حصلت مشكلة غير متوقعة / Unexpected error: {type(exc).__name__}"


demo = gr.ChatInterface(
    fn=respond,
    title=TITLE,
    description=DESCRIPTION,
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.launch()
