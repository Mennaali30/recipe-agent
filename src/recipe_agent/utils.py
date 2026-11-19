"""Small helpers shared across the project."""


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
