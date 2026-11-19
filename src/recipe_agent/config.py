"""Central configuration: picks the LLM based on environment variables."""
import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL")


def get_llm(temperature: float = 0.3):
    """Return a chat model for the configured provider."""
    if LLM_PROVIDER == "gemini":
        if not GEMINI_MODEL:
            raise ValueError("Set GEMINI_MODEL (model code from Google AI Studio).")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature)

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")
