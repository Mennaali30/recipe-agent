"""The one place the multi-agent layer talks to an LLM: structured extraction only.

An extractor is `extract(schema, system_prompt, text) -> schema instance`. Tests pass a fake one.
The LLM only turns the user's words into typed arguments; it never writes a number into an answer.
"""
import logging

log = logging.getLogger(__name__)


def llm_extract(schema, system: str, text: str):
    """Real extractor: Gemini with structured output, retries and model fallbacks (as the rest of the app)."""
    from recipe_agent.config import get_llm_chain
    from recipe_agent.llm_call import invoke_with_backoff

    chain = get_llm_chain(temperature=0.0, schema=schema)
    out = invoke_with_backoff(chain, [("system", system), ("human", text)])
    if isinstance(out, dict):
        out = schema.model_validate(out)
    return out
