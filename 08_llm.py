"""
08_llm.py

OncoRAG Clinical Assistant - LLM Communication Module.

Sole responsibility: communicate with an LLM through the OpenRouter API.
This module does NOT perform retrieval, prompt construction, or RAG
orchestration - it only knows how to authenticate, send a prompt, and
return clean text (plus optional generation metadata).

Pipeline position:
    01_documents.py              -> raw Document objects
    02_preprocessing.py          -> cleaned Document objects
    03_chunking.py               -> chunked Document objects
    04_vector_representation.py  -> validated embedding_model
    05_create_chroma_store.py    -> persistent, queryable Chroma index
    06_retrieve_context.py       -> retrieved Document chunks
    07_prompting.py               -> complete prompt string (system + user + sources)
    08_llm.py (this module)      -> raw answer text, given that prompt

Why OpenRouter (not a direct Gemini/OpenAI SDK):
    The project's submission rules mandate reading credentials from
    Streamlit TOML secrets as OPENROUTER_API_KEY / OPENROUTER_MODEL, with
    a default model of "openai/gpt-4o-mini". OpenRouter exposes an
    OpenAI-compatible REST endpoint, so this module talks to it with
    plain HTTP (via the 'requests' library) rather than pulling in an
    extra vendor SDK.

Authentication:
    Reads the API key from the OPENROUTER_API_KEY environment variable
    via os.getenv(). If python-dotenv is installed and a .env file is
    present, it is loaded automatically at import time. In the deployed
    Streamlit app, streamlit_app.py is responsible for copying the key
    from st.secrets into the environment before this module is used -
    never hardcode a real key in any Python file.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Tuple, TypeVar

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional convenience only
    pass

try:
    import requests
except ImportError as import_error:  # pragma: no cover
    raise ImportError(
        "requests could not be imported. Install it with: pip install requests"
    ) from import_error


__all__ = [
    "MissingAPIKeyError",
    "GenerationError",
    "GenerationResult",
    "create_openrouter_client",
    "generate_answer",
    "generate_answer_with_metadata",
    "generate_answer_stream",
    "health_check",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")

# --------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL_NAME = "openai/gpt-4o-mini"
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_MAX_OUTPUT_TOKENS: int = 1024
DEFAULT_TIMEOUT_SECONDS: float = 45.0
DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS: float = 15.0

DEFAULT_MAX_RETRIES: int = 2
DEFAULT_RETRY_INITIAL_DELAY_SECONDS: float = 1.0
DEFAULT_RETRY_BACKOFF_MULTIPLIER: float = 2.0

# Module-level, mutable so streamlit_app.py can inject values loaded from
# Streamlit secrets at runtime (see the project README for the pattern).
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL_NAME)


# --------------------------------------------------------------------------
# Custom exceptions
# --------------------------------------------------------------------------
class MissingAPIKeyError(Exception):
    """Raised when no OpenRouter API key can be found."""


class GenerationError(Exception):
    """Raised when an OpenRouter completion request fails."""


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GenerationResult:
    """Answer text plus metadata describing how it was produced.

    Attributes:
        answer: The cleaned answer text returned by the model.
        model_name: The model identifier used for this generation.
        generation_time_seconds: Wall-clock time spent waiting on the API call.
        prompt_length: Number of characters in the input prompt.
        response_length: Number of characters in the cleaned answer.
        input_tokens: Prompt token count reported by the API, if available.
        output_tokens: Response token count reported by the API, if available.
    """

    answer: str
    model_name: str
    generation_time_seconds: float
    prompt_length: int
    response_length: int
    input_tokens: int
    output_tokens: int


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _run_with_timeout(func: Callable[[], T], timeout_seconds: float) -> T:
    """Run a zero-argument callable with a hard wall-clock timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as exc:
            raise GenerationError(
                f"OpenRouter call timed out after {timeout_seconds}s."
            ) from exc


def _call_with_retry(
    func: Callable[[], T],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_RETRY_INITIAL_DELAY_SECONDS,
    backoff_multiplier: float = DEFAULT_RETRY_BACKOFF_MULTIPLIER,
) -> T:
    """Call func(), retrying on failure with exponential backoff.

    Only network/timeout-style errors are worth retrying; a bad request
    (e.g. an invalid model name or malformed payload) will fail the same
    way every time, so retrying it just makes the user wait longer for
    the same error. Those are raised immediately without retrying.
    """
    delay = initial_delay
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except GenerationError as exc:
            last_exc = exc
            if "HTTP 4" in str(exc):
                # Client errors (bad key, bad model, bad payload) will not
                # be fixed by retrying - fail fast instead of hanging.
                raise
            if attempt >= max_retries:
                raise
            logger.warning(
                "OpenRouter call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt + 1, max_retries + 1, exc, delay,
            )
            time.sleep(delay)
            delay *= backoff_multiplier

    raise last_exc  # pragma: no cover - unreachable, satisfies type checkers


def _resolve_api_key(explicit_key: Optional[str]) -> str:
    """Resolve the API key to use, preferring an explicitly passed value."""
    key = explicit_key or OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise MissingAPIKeyError(
            "No OpenRouter API key found. Set OPENROUTER_API_KEY as an "
            "environment variable, in a local .env file, or via Streamlit "
            "secrets in production."
        )
    return key


def _clean_response_text(text: Optional[str]) -> str:
    """Strip leading/trailing whitespace from a raw response string."""
    return (text or "").strip()


def _post_chat_completion(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    timeout_seconds: float,
) -> dict:
    """Send a single chat-completion request to OpenRouter and return JSON."""

    def _do_request() -> dict:
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_output_tokens,
                },
                timeout=timeout_seconds,
            )
        except requests.exceptions.RequestException as exc:
            raise GenerationError(f"Network error calling OpenRouter: {exc}") from exc

        if not response.ok:
            raise GenerationError(
                f"OpenRouter returned HTTP {response.status_code}: {response.text[:500]}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise GenerationError(
                f"OpenRouter returned a non-JSON response: {response.text[:500]}"
            ) from exc

    return _run_with_timeout(_do_request, timeout_seconds + 5.0)


def _extract_answer_and_usage(payload: dict) -> Tuple[str, int, int]:
    """Pull the answer text and token usage out of an OpenRouter response."""
    choices = payload.get("choices") or []
    if not choices:
        error_info = payload.get("error")
        if error_info:
            raise GenerationError(f"OpenRouter error: {error_info}")
        raise GenerationError("OpenRouter response contained no choices.")

    message = choices[0].get("message", {})
    answer = _clean_response_text(message.get("content"))

    usage = payload.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)

    return answer, input_tokens, output_tokens


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def create_openrouter_client(api_key: Optional[str] = None) -> str:
    """Validate and return the OpenRouter API key to use for requests.

    There is no persistent client object for a plain REST API, so this
    simply resolves and validates the key - kept as a function (rather
    than inlining _resolve_api_key everywhere) so streamlit_app.py has a
    single, obvious call to make at startup.

    Args:
        api_key: An explicit API key to use instead of the environment/
            module-level default.

    Returns:
        The resolved API key string.

    Raises:
        MissingAPIKeyError: If no key is available from any source.
    """
    return _resolve_api_key(api_key)


def generate_answer(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Generate a clean answer string from the LLM for a given prompt.

    Args:
        api_key: A validated OpenRouter API key, from create_openrouter_client().
        model: OpenRouter model identifier (e.g. "openai/gpt-4o-mini").
        prompt: The full prompt text to send (typically the output of
            07_prompting.py::build_complete_prompt()).
        temperature: Sampling temperature. Defaults to 0.2.
        max_output_tokens: Maximum number of tokens to generate.
        timeout_seconds: Hard wall-clock timeout per attempt.

    Returns:
        The generated answer text, with surrounding whitespace removed.

    Raises:
        GenerationError: If prompt is empty, or generation fails after
            all retry attempts.
    """
    result = generate_answer_with_metadata(
        api_key, model, prompt, temperature, max_output_tokens, timeout_seconds
    )
    return result.answer


def generate_answer_with_metadata(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> GenerationResult:
    """Generate an answer from the LLM along with generation metadata.

    Args:
        api_key: A validated OpenRouter API key.
        model: OpenRouter model identifier.
        prompt: The full prompt text to send.
        temperature: Sampling temperature. Defaults to 0.2.
        max_output_tokens: Maximum number of tokens to generate.
        timeout_seconds: Hard wall-clock timeout per attempt.

    Returns:
        A GenerationResult with the cleaned answer, model name,
        generation time, prompt/response character lengths, and
        input/output token counts.

    Raises:
        GenerationError: If prompt is empty, or generation fails after
            all retry attempts.
    """
    if not prompt or not prompt.strip():
        raise GenerationError("prompt must not be empty.")
    if not api_key:
        raise MissingAPIKeyError("api_key must not be empty.")

    logger.info(
        "Generating answer with model '%s' (temperature=%s, max_output_tokens=%d)...",
        model, temperature, max_output_tokens,
    )

    start_time = time.monotonic()

    def _attempt() -> dict:
        return _post_chat_completion(
            api_key, model, prompt, temperature, max_output_tokens, timeout_seconds
        )

    payload = _call_with_retry(_attempt)
    elapsed_seconds = time.monotonic() - start_time

    answer, input_tokens, output_tokens = _extract_answer_and_usage(payload)

    if not input_tokens and not output_tokens:
        # Fall back to a rough estimate if the API omitted usage data.
        input_tokens = round(len(prompt.split()) * 1.3)
        output_tokens = round(len(answer.split()) * 1.3)

    result = GenerationResult(
        answer=answer,
        model_name=model,
        generation_time_seconds=elapsed_seconds,
        prompt_length=len(prompt),
        response_length=len(answer),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    logger.info(
        "Generated answer in %.2fs | prompt_length=%d | response_length=%d | "
        "input_tokens=%d | output_tokens=%d.",
        elapsed_seconds, result.prompt_length, result.response_length,
        input_tokens, output_tokens,
    )
    return result


def generate_answer_stream(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Iterator[str]:
    """Stream the LLM's answer incrementally as it is generated.

    Uses OpenRouter's Server-Sent Events streaming mode (``stream: true``)
    so the UI can render text as it arrives instead of waiting for the
    full response - this is purely a perceived-latency/UX improvement;
    total generation time is unchanged. Falls back to nothing (silently
    yields no chunks) only if the stream contains no content, matching
    the fail-soft posture used elsewhere in this module.

    Args:
        api_key: A validated OpenRouter API key, from create_openrouter_client().
        model: OpenRouter model identifier (e.g. "openai/gpt-4o-mini").
        prompt: The full prompt text to send.
        temperature: Sampling temperature. Defaults to 0.2.
        max_output_tokens: Maximum number of tokens to generate.
        timeout_seconds: Hard wall-clock timeout for the whole stream.

    Yields:
        Successive text fragments (deltas) as they are generated by the
        model. Concatenating all yielded fragments reproduces the full
        answer text.

    Raises:
        GenerationError: If prompt is empty, or the streaming request
            fails (network error, non-2xx response, or malformed stream).
        MissingAPIKeyError: If api_key is empty.
    """
    if not prompt or not prompt.strip():
        raise GenerationError("prompt must not be empty.")
    if not api_key:
        raise MissingAPIKeyError("api_key must not be empty.")

    logger.info(
        "Streaming answer with model '%s' (temperature=%s, max_output_tokens=%d)...",
        model, temperature, max_output_tokens,
    )

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_output_tokens,
                "stream": True,
            },
            timeout=timeout_seconds,
            stream=True,
        )
    except requests.exceptions.RequestException as exc:
        raise GenerationError(f"Network error calling OpenRouter (stream): {exc}") from exc

    if not response.ok:
        raise GenerationError(
            f"OpenRouter returned HTTP {response.status_code}: {response.text[:500]}"
        )

    received_any = False
    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            payload_text = raw_line[len("data:"):].strip()
            if payload_text == "[DONE]":
                break
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue  # skip occasional keep-alive/comment lines

            choices = payload.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {}) or {}
            fragment = delta.get("content")
            if fragment:
                received_any = True
                yield fragment
    except requests.exceptions.RequestException as exc:
        raise GenerationError(f"OpenRouter stream interrupted: {exc}") from exc
    finally:
        response.close()

    if not received_any:
        logger.warning("OpenRouter stream for model '%s' produced no content.", model)


def health_check(api_key: str, model: str = DEFAULT_MODEL_NAME) -> bool:
    """Verify that OpenRouter is reachable and responding correctly.

    Sends a minimal "Hello" prompt and returns True only if a non-empty
    text response is received within a short timeout. Never raises: any
    failure results in False, since a health check must be a safe,
    non-throwing probe.

    Args:
        api_key: A validated OpenRouter API key.
        model: Model identifier to probe.

    Returns:
        True if the model returned a non-empty response, False otherwise.
    """
    logger.info("Running OpenRouter health check (model='%s')...", model)
    try:
        payload = _post_chat_completion(
            api_key, model, "Hello", 0.0, 16, DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS
        )
        answer, _, _ = _extract_answer_and_usage(payload)
    except Exception as exc:
        logger.warning("OpenRouter health check failed: %s", exc)
        return False

    is_healthy = bool(answer)
    logger.info(
        "OpenRouter health check %s.", "passed" if is_healthy else "failed (empty response)"
    )
    return is_healthy
