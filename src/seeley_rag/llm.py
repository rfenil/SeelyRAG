"""Provider-agnostic LLM access.

One module so that "which model answers installer questions" is a config change
rather than a code change, and so nothing above this layer imports a vendor SDK.

⚠ **The build plan names Claude** -- Sonnet for generation, Haiku for the query
router (sections 7.1 and 8). Nothing in the architecture requires it. The plan
recorded a preference before any key existed; this project has an OpenAI key and
no Anthropic one, and OpenAI covers every remaining model need: generation, the
query router, and the Stage 2b vision transcription still owed on 3,459 pages.

So ``openai`` is the default provider and ``anthropic`` remains fully wired. See
ADR 0008.

Latency, measured against this key rather than assumed
------------------------------------------------------
The gpt-5 family reasons by default, which is wrong for a router that the plan
budgets at ~200ms:

===========================  ========  ================
Model                        Latency   Reasoning tokens
===========================  ========  ================
gpt-5-mini (default)           8.3s    yes
gpt-5-nano (default)           4.4s    yes
gpt-5-mini reasoning=minimal   2.1s    0
gpt-5-nano reasoning=minimal   1.5s    0
gpt-4.1-mini                   1.0s    0
gpt-4.1-nano                   0.6s    0
===========================  ========  ================

Hence :data:`REASONING_MODELS` and the ``reasoning_effort`` default: a
reasoning model used as a router is told not to reason, because the routing
decision does not need it and an installer waiting eight seconds does.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from seeley_rag.exceptions import ConfigurationError, SeeleyRagError
from seeley_rag.logging_conf import get_logger
from seeley_rag.settings import get_settings

log = get_logger(__name__)

#: Providers this module can dispatch to.
PROVIDERS = ("openai", "anthropic")

#: Model families that spend reasoning tokens unless told otherwise. Matched as
#: a prefix, so dated snapshots ("gpt-5-mini-2025-08-07") are covered too.
REASONING_MODELS = ("gpt-5", "o1", "o3", "o4")

#: Attempts before giving up.
MAX_ATTEMPTS = 3

#: Base seconds for exponential backoff.
BACKOFF_BASE_SECONDS = 1.5

#: Fenced JSON, for models that wrap their output in markdown despite being told
#: not to. Cheaper to strip than to re-prompt.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class LLMError(SeeleyRagError):
    """A completion could not be obtained."""


def is_reasoning_model(model: str) -> bool:
    """Whether a model spends reasoning tokens by default.

    Args:
        model: Model identifier.

    Returns:
        True for the gpt-5 and o-series families.
    """
    return model.startswith(REASONING_MODELS)


def _retryable(exc: Exception) -> bool:
    """Whether a failure is worth another attempt.

    Args:
        exc: The raised exception.

    Returns:
        True for rate limits, timeouts and 5xx.
    """
    name = type(exc).__name__
    if name in {"AuthenticationError", "PermissionDeniedError", "BadRequestError"}:
        return False
    if name in {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


def _openai_client() -> Any:
    """Build an OpenAI client from configured settings.

    Returns:
        The client.

    Raises:
        ConfigurationError: If the key or the SDK is missing.
    """
    key = get_settings().openai_api_key
    if not key:
        raise ConfigurationError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ConfigurationError(
            'The openai package is not installed. Run: pip install -e ".[downstream]"'
        ) from exc
    return OpenAI(api_key=key)


def _anthropic_client() -> Any:
    """Build an Anthropic client from configured settings.

    Returns:
        The client.

    Raises:
        ConfigurationError: If the key or the SDK is missing.
    """
    key = get_settings().anthropic_api_key
    if not key:
        raise ConfigurationError(
            "ANTHROPIC_API_KEY is not set, but the configured LLM provider is 'anthropic'. "
            "Set the key, or set generate.provider to 'openai' in config/config.yaml."
        )
    try:
        import anthropic
    except ImportError as exc:
        raise ConfigurationError(
            'The anthropic package is not installed. Run: pip install -e ".[downstream]"'
        ) from exc
    return anthropic.Anthropic(api_key=key)


def _call_openai(
    client: Any,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    json_mode: bool,
    reasoning_effort: str | None,
) -> str:
    """Make one OpenAI chat completion.

    Args:
        client: The OpenAI client.
        model: Model identifier.
        system: System instruction.
        user: User message.
        max_tokens: Output cap.
        json_mode: Whether to request a JSON object.
        reasoning_effort: Effort for reasoning models; ignored by others.

    Returns:
        The response text.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if is_reasoning_model(model):
        # Reasoning models reject `max_tokens` in favour of a completion cap,
        # and reason by default -- which costs seconds a router cannot spend.
        payload["max_completion_tokens"] = max_tokens
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
    else:
        payload["max_tokens"] = max_tokens

    response = client.chat.completions.create(**payload)
    return response.choices[0].message.content or ""


def _call_anthropic(client: Any, model: str, system: str, user: str, max_tokens: int) -> str:
    """Make one Anthropic message completion.

    Args:
        client: The Anthropic client.
        model: Model identifier.
        system: System instruction.
        user: User message.
        max_tokens: Output cap.

    Returns:
        The response text.
    """
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text:
            return str(text)
    return ""


def complete(
    system: str,
    user: str,
    model: str | None = None,
    provider: str | None = None,
    client: Any | None = None,
    max_tokens: int = 1024,
    json_mode: bool = False,
    reasoning_effort: str | None = None,
) -> str:
    """Get one completion from the configured provider.

    Args:
        system: System instruction.
        user: User message.
        model: Model identifier. Defaults to the configured generation model.
        provider: ``openai`` or ``anthropic``. Defaults to configured.
        client: Injected SDK client, for tests. Its shape must match
            ``provider``.
        max_tokens: Output cap.
        json_mode: Ask the provider for a JSON object where supported.
        reasoning_effort: ``minimal`` / ``low`` / ``medium`` / ``high`` for
            reasoning models. Defaults to the configured value.

    Returns:
        The response text.

    Raises:
        LLMError: If the call fails and retrying will not help, or attempts run
            out.
        ConfigurationError: If the provider is unknown or unconfigured.
    """
    settings = get_settings().generate
    resolved_provider = (provider or settings.provider).lower()
    resolved_model = model or settings.model
    effort = reasoning_effort if reasoning_effort is not None else settings.reasoning_effort

    if resolved_provider not in PROVIDERS:
        raise ConfigurationError(
            f"Unknown LLM provider {resolved_provider!r}. Expected one of {PROVIDERS}."
        )

    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if resolved_provider == "openai":
                active = client or _openai_client()
                return _call_openai(
                    active, resolved_model, system, user, max_tokens, json_mode, effort
                )
            active = client or _anthropic_client()
            return _call_anthropic(active, resolved_model, system, user, max_tokens)
        except ConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDKs raise many types
            last = exc
            if not _retryable(exc) or attempt == MAX_ATTEMPTS:
                break
            delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            log.warning("llm_retry", extra={"attempt": attempt, "delay": delay, "error": str(exc)})
            time.sleep(delay)

    raise LLMError(f"{resolved_provider} completion failed: {last}")


def complete_json(
    system: str,
    user: str,
    model: str | None = None,
    provider: str | None = None,
    client: Any | None = None,
    max_tokens: int = 1024,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Get one completion and parse it as a JSON object.

    Args:
        system: System instruction.
        user: User message.
        model: Model identifier. Defaults to the configured router model.
        provider: ``openai`` or ``anthropic``. Defaults to configured.
        client: Injected SDK client, for tests.
        max_tokens: Output cap.
        reasoning_effort: Effort for reasoning models.

    Returns:
        The parsed object.

    Raises:
        LLMError: If the call fails, or the response is not a JSON object.
    """
    text = complete(
        system=system,
        user=user,
        model=model or get_settings().generate.router_model,
        provider=provider,
        client=client,
        max_tokens=max_tokens,
        json_mode=True,
        reasoning_effort=reasoning_effort,
    )
    return parse_json_object(text)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a model's response as a JSON object, tolerating a code fence.

    Args:
        text: The response text.

    Returns:
        The parsed object.

    Raises:
        LLMError: If the text is not a JSON object.
    """
    stripped = (text or "").strip()
    fenced = _JSON_FENCE_RE.search(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model did not return JSON: {stripped[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise LLMError(f"Model returned {type(parsed).__name__}, expected a JSON object.")
    return parsed


def active_provider() -> str:
    """Return the configured provider name.

    Returns:
        ``openai`` or ``anthropic``.
    """
    return get_settings().generate.provider.lower()


def is_configured() -> bool:
    """Whether the configured provider has a usable key.

    Lets callers degrade gracefully instead of raising -- query understanding
    falls back to its deterministic pass, and reranking to fusion order.

    Returns:
        True when a key for the active provider is set.
    """
    settings = get_settings()
    if active_provider() == "anthropic":
        return bool(settings.anthropic_api_key)
    return bool(settings.openai_api_key)
