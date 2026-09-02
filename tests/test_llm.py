"""Provider-agnostic LLM access.

No test here touches the network: every SDK client is injected as a fake, and
``conftest.py``'s tripwire fails anything that tries otherwise.

What matters is the dispatch, not the models. The two SDKs have different call
shapes and different parameter rules, and getting either wrong fails at runtime
with a vendor error that says nothing about this project.
"""

from __future__ import annotations

from typing import Any

import pytest

from seeley_rag import llm
from seeley_rag.exceptions import ConfigurationError
from seeley_rag.llm import (
    LLMError,
    active_provider,
    complete,
    complete_json,
    is_reasoning_model,
    parse_json_object,
)


class FakeOpenAI:
    """An OpenAI-shaped client capturing its payloads."""

    def __init__(self, content: str = '{"ok": true}', error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        outer = self

        class Completions:
            @staticmethod
            def create(**payload: Any) -> Any:
                outer.calls.append(payload)
                if error:
                    raise error
                message = type("Message", (), {"content": content})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        self.chat = type("Chat", (), {"completions": Completions()})()


class FakeAnthropic:
    """An Anthropic-shaped client capturing its payloads."""

    def __init__(self, content: str = '{"ok": true}') -> None:
        self.calls: list[dict[str, Any]] = []
        outer = self

        class Messages:
            @staticmethod
            def create(**payload: Any) -> Any:
                outer.calls.append(payload)
                block = type("Block", (), {"text": content})()
                return type("Response", (), {"content": [block]})()

        self.messages = Messages()


class TestReasoningModelDetection:
    """Which models spend reasoning tokens unless told not to."""

    @pytest.mark.parametrize(
        "model", ["gpt-5", "gpt-5-mini", "gpt-5-nano-2025-08-07", "o1", "o3-mini", "o4-mini"]
    )
    def test_reasoning_families_are_detected(self, model: str) -> None:
        """Dated snapshots must match too, hence prefix matching."""
        assert is_reasoning_model(model)

    @pytest.mark.parametrize("model", ["gpt-4.1-mini", "gpt-4o-mini", "claude-sonnet-5"])
    def test_other_models_are_not(self, model: str) -> None:
        """Sending reasoning parameters to these is an API error."""
        assert not is_reasoning_model(model)


class TestOpenAIDispatch:
    """The default provider."""

    def test_system_and_user_are_sent_as_messages(self) -> None:
        """The shape the chat completions API expects."""
        client = FakeOpenAI()
        complete("be helpful", "a question", model="gpt-4.1-mini", provider="openai", client=client)
        messages = client.calls[0]["messages"]
        assert messages[0] == {"role": "system", "content": "be helpful"}
        assert messages[1] == {"role": "user", "content": "a question"}

    def test_json_mode_is_requested_when_asked(self) -> None:
        """Without it the model wanders out of JSON on long inputs."""
        client = FakeOpenAI()
        complete("s", "u", model="gpt-4.1-mini", provider="openai", client=client, json_mode=True)
        assert client.calls[0]["response_format"] == {"type": "json_object"}

    def test_non_reasoning_models_get_max_tokens(self) -> None:
        """The classic parameter."""
        client = FakeOpenAI()
        complete("s", "u", model="gpt-4.1-mini", provider="openai", client=client, max_tokens=99)
        assert client.calls[0]["max_tokens"] == 99
        assert "reasoning_effort" not in client.calls[0]

    def test_reasoning_models_get_a_completion_cap_and_effort(self) -> None:
        """They reject `max_tokens`, and reason by default.

        gpt-5-mini takes 8.3s unprompted and 2.1s at minimal effort -- a router
        cannot spend the difference.
        """
        client = FakeOpenAI()
        complete(
            "s",
            "u",
            model="gpt-5-mini",
            provider="openai",
            client=client,
            max_tokens=99,
            reasoning_effort="minimal",
        )
        payload = client.calls[0]
        assert payload["max_completion_tokens"] == 99
        assert payload["reasoning_effort"] == "minimal"
        assert "max_tokens" not in payload

    def test_empty_content_is_not_an_error(self) -> None:
        """A refusal or an empty completion returns "" rather than raising."""
        client = FakeOpenAI(content="")
        assert complete("s", "u", model="gpt-4.1-mini", provider="openai", client=client) == ""


class TestAnthropicDispatch:
    """Still fully wired; the plan's original choice."""

    def test_system_is_a_top_level_parameter(self) -> None:
        """Anthropic takes `system` beside `messages`, not inside it."""
        client = FakeAnthropic()
        complete("be helpful", "a question", model="claude-x", provider="anthropic", client=client)
        payload = client.calls[0]
        assert payload["system"] == "be helpful"
        assert payload["messages"] == [{"role": "user", "content": "a question"}]

    def test_text_is_read_from_the_first_block(self) -> None:
        """Responses are a list of content blocks."""
        client = FakeAnthropic(content="hello")
        assert complete("s", "u", provider="anthropic", client=client) == "hello"


class TestProviderSelection:
    """Config decides, callers may override."""

    def test_default_provider_is_openai(self) -> None:
        """The plan names Claude; this project has an OpenAI key. ADR 0008."""
        assert active_provider() == "openai"

    def test_an_unknown_provider_is_rejected(self) -> None:
        """A typo in config must fail loudly, not fall through to a default."""
        with pytest.raises(ConfigurationError, match="Unknown LLM provider"):
            complete("s", "u", provider="gemini", client=FakeOpenAI())

    def test_anthropic_without_a_key_names_the_fix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The error has to say what to change.

        The key is cleared explicitly: developer shells often export
        ANTHROPIC_API_KEY, and pydantic-settings reads the ambient environment,
        so this would otherwise pass or fail depending on whose machine ran it.
        """
        from seeley_rag import settings as settings_module

        monkeypatch.setattr(settings_module.get_settings(), "anthropic_api_key", None)
        with pytest.raises(ConfigurationError, match="generate.provider"):
            complete("s", "u", provider="anthropic")


class TestJsonParsing:
    """Models wrap JSON in fences however firmly they are told not to."""

    def test_plain_json(self) -> None:
        """The normal case."""
        assert parse_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_json_is_unwrapped(self) -> None:
        """Cheaper to strip than to re-prompt."""
        assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_bare_fence_is_unwrapped(self) -> None:
        """Some models omit the language tag."""
        assert parse_json_object('```\n{"a": 1}\n```') == {"a": 1}

    def test_non_json_raises(self) -> None:
        """Callers catch LLMError and fall back."""
        with pytest.raises(LLMError, match="did not return JSON"):
            parse_json_object("I'm sorry, I can't help with that.")

    def test_a_json_array_is_rejected(self) -> None:
        """Callers index by key; a list would fail confusingly later."""
        with pytest.raises(LLMError, match="expected a JSON object"):
            parse_json_object("[1, 2, 3]")

    def test_complete_json_returns_the_parsed_object(self) -> None:
        """End to end through the dispatch."""
        client = FakeOpenAI(content='{"intent": "fault_diagnosis"}')
        assert complete_json("s", "u", provider="openai", client=client) == {
            "intent": "fault_diagnosis"
        }


class TestRetries:
    """Transient failures are retried; permanent ones are not."""

    def test_a_transient_failure_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rate limits are normal under load."""
        monkeypatch.setattr(llm.time, "sleep", lambda _: None)
        error = type("RateLimitError", (Exception,), {})()

        calls = {"n": 0}
        client = FakeOpenAI()
        original = client.chat.completions.create

        def flaky(**payload: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise error
            return original(**payload)

        client.chat.completions.create = flaky  # type: ignore[method-assign]
        assert complete("s", "u", provider="openai", client=client)
        assert calls["n"] == 2

    def test_a_permanent_failure_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retrying a bad key just delays the failure."""
        monkeypatch.setattr(llm.time, "sleep", lambda _: None)
        error = type("AuthenticationError", (Exception,), {})()
        client = FakeOpenAI(error=error)
        with pytest.raises(LLMError):
            complete("s", "u", provider="openai", client=client)
        assert len(client.calls) == 1

    def test_exhausted_retries_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Give up rather than loop."""
        monkeypatch.setattr(llm.time, "sleep", lambda _: None)
        error = type("RateLimitError", (Exception,), {})()
        client = FakeOpenAI(error=error)
        with pytest.raises(LLMError, match="completion failed"):
            complete("s", "u", provider="openai", client=client)
        assert len(client.calls) == llm.MAX_ATTEMPTS
