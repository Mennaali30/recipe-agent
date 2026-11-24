"""Small helpers shared across the project."""
import re


def message_text(message) -> str:
    """Return the plain text of a chat message (handles str or list-of-blocks content)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


_FAHRENHEIT = re.compile(
    r"\b(\d{2,3})\s*(?:°\s*)?(?:degrees?\s*)?(?:F\b|Fahrenheit\b)", re.IGNORECASE
)


def fahrenheit_to_celsius(text: str) -> str:
    """Rewrite Fahrenheit temperatures as Celsius-first: '350 degrees F' -> '175°C (350°F)'.

    Celsius is rounded to the nearest 5 degrees, as ovens are set. Anything without an
    explicit F / Fahrenheit marker is left untouched.
    """
    def _sub(match) -> str:
        f = int(match.group(1))
        c = int(round((f - 32) * 5 / 9 / 5) * 5)
        return f"{c}°C ({f}°F)"

    return _FAHRENHEIT.sub(_sub, str(text))


def clean_name(name) -> str:
    """Tidy a recipe name: collapse whitespace, drop stray quotes, de-capitalize
    ALL-CAPS names ('GRANDMA'S SOUP' -> \"Grandma's Soup\")."""
    s = re.sub(r"\s+", " ", str(name)).strip().strip("\"'").strip()
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)  # no space before punctuation
    if len(s) > 3 and s.isupper():
        s = s.title()
        s = re.sub(r"'S\b", "'s", s)  # title() turns "Grandma's" into "Grandma'S"
    return s
