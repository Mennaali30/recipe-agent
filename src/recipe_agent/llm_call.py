"""Call the LLM with retry + exponential backoff on transient errors.

Transient means the failure is on Google's side and a retry can succeed:
429 RESOURCE_EXHAUSTED (quota / rate limit) and 503 UNAVAILABLE (overloaded model).
Anything else (bad key, bad request) is raised immediately.
"""
import logging
import os
import re
import time

log = logging.getLogger(__name__)

_MARKERS = ("RESOURCE_EXHAUSTED", "UNAVAILABLE")
_CODE = re.compile(r"\b(429|503)\b")


def _attempts() -> int:
    try:
        return max(1, int(os.getenv("LLM_RETRY_ATTEMPTS", "4")))
    except ValueError:
        return 4


def _base_delay() -> float:
    try:
        return max(0.0, float(os.getenv("LLM_RETRY_BASE_DELAY", "2")))
    except ValueError:
        return 2.0


def is_transient(exc: Exception) -> bool:
    """True for quota (429) and overload (503) errors that a retry can fix."""
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        if int(code) in (429, 503):
            return True
    except (TypeError, ValueError):
        pass
    text = str(exc)
    return any(m in text for m in _MARKERS) or bool(_CODE.search(text))


def invoke_with_backoff(runnable, messages, *, attempts=None, base_delay=None, sleep=time.sleep):
    """runnable.invoke(messages) with retries on transient errors.

    Waits base_delay, 2x, 4x... between attempts. After the last attempt the original
    error is raised, so the caller's own fallback (plain answer, code-only plan) runs.
    """
    attempts = attempts or _attempts()
    delay = _base_delay() if base_delay is None else base_delay
    for i in range(attempts):
        try:
            return runnable.invoke(messages)
        except Exception as exc:
            if not is_transient(exc) or i == attempts - 1:
                raise
            wait = delay * (2**i)
            log.warning(
                "LLM call failed with a transient error (%s: %s); retrying in %.1fs (%d/%d)",
                type(exc).__name__, exc, wait, i + 1, attempts,
            )
            sleep(wait)
