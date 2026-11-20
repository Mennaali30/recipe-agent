"""Central configuration: picks the LLM(s) based on environment variables."""
import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()


def _model_names() -> list:
    """Primary model first, then the fallbacks (comma separated in GEMINI_FALLBACK_MODELS)."""
    primary = os.getenv("GEMINI_MODEL")
    if not primary:
        raise ValueError("Set GEMINI_MODEL (model code from Google AI Studio).")
    fallbacks = [
        m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()
    ]
    names = []
    for name in [primary, *fallbacks]:
        if name not in names:
            names.append(name)
    return names


def _build(name: str, temperature: float, timeout: int, max_retries: int):
    if LLM_PROVIDER != "gemini":
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=name,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_llm(temperature: float = 0.3, timeout: int = 60, max_retries: int = 5):
    """Return a chat model for the primary model."""
    return _build(_model_names()[0], temperature, timeout, max_retries)


def get_llm_chain(temperature: float = 0.3, schema=None, timeout: int = 60):
    """LLM runnable that falls back to the next model on any error
    (daily quota 429, overload 503, timeout...).

    If `schema` (a Pydantic class) is given, every model returns structured output.
    """
    names = _model_names()
    retries = 1 if len(names) > 1 else 5  # with fallbacks, switch fast instead of retrying
    runnables = []
    for name in names:
        llm = _build(name, temperature, timeout, retries)
        runnables.append(llm.with_structured_output(schema) if schema else llm)
    primary, *rest = runnables
    return primary.with_fallbacks(rest) if rest else primary
