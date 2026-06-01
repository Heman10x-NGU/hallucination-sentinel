"""Tests for providers/base.py and providers/openai_compatible.py.

Unit tests use mocked HTTP responses (no real API calls).
Integration tests are skipped unless environment variables are set.
"""

from __future__ import annotations

import math
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hallucination_sentinel.providers.base import (
    CompletionLogprobs,
    ProviderCapabilityError,
    ProviderConfig,
    TokenLogprob,
    TokenLogprobResult,
)
from hallucination_sentinel.providers.openai_compatible import (
    OpenAICompatibleProvider,
    PROVIDER_SPECS,
    ProviderSpec,
    TokenLogprobProvider,
    _parse_openai_logprobs,
    smoke_provider,
)


# ---------------------------------------------------------------------------
# Fixtures: mock API responses
# ---------------------------------------------------------------------------

def _make_openai_response(
    tokens: list[str],
    logprobs: list[float],
    top_logprobs_per_token: list[list[dict[str, float]]] | None = None,
) -> dict:
    """Build a minimal OpenAI-style chat completion response with logprobs."""
    content = []
    for i, (tok, lp) in enumerate(zip(tokens, logprobs)):
        entry: dict = {"token": tok, "logprob": lp}
        if top_logprobs_per_token is not None:
            entry["top_logprobs"] = top_logprobs_per_token[i]
        content.append(entry)
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": "".join(tokens)},
                "logprobs": {"content": content},
            }
        ]
    }


def _response_with_topk() -> dict:
    """Response with top-k=3 logprobs per token."""
    return _make_openai_response(
        tokens=["Hello", " world"],
        logprobs=[-0.1, -0.5],
        top_logprobs_per_token=[
            [
                {"token": "Hello", "logprob": -0.1},
                {"token": "Hi", "logprob": -1.5},
                {"token": "Hey", "logprob": -2.0},
            ],
            [
                {"token": " world", "logprob": -0.5},
                {"token": " there", "logprob": -1.0},
                {"token": " friend", "logprob": -2.5},
            ],
        ],
    )


def _response_selected_only() -> dict:
    """Response with only selected-token logprobs (top_logprobs=[])."""
    return _make_openai_response(
        tokens=["Yes"],
        logprobs=[-0.05],
        top_logprobs_per_token=[[]],
    )


def _response_no_logprobs() -> dict:
    """Response where logprobs field is missing entirely."""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello"},
                "logprobs": None,
            }
        ]
    }


def _response_empty_content() -> dict:
    """Response where logprobs.content is empty."""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello"},
                "logprobs": {"content": []},
            }
        ]
    }


# ---------------------------------------------------------------------------
# providers/base.py
# ---------------------------------------------------------------------------


class TestProviderCapabilityError:
    """Tests for the ProviderCapabilityError exception."""

    def test_basic_message(self):
        err = ProviderCapabilityError("no logprobs")
        assert str(err) == "no logprobs"
        assert err.capability == ""
        assert err.provider == ""

    def test_with_capability_and_provider(self):
        err = ProviderCapabilityError(
            "top-k not available",
            capability="top_k_logprobs",
            provider="openai",
        )
        assert err.capability == "top_k_logprobs"
        assert err.provider == "openai"
        assert "top-k not available" in str(err)

    def test_is_exception(self):
        assert issubclass(ProviderCapabilityError, Exception)


class TestProviderConfig:
    """Tests for the ProviderConfig dataclass."""

    def test_from_env_with_explicit_args(self):
        cfg = ProviderConfig.from_env(
            api_key="sk-test-123",
            base_url="https://example.com/v1",
            model="test-model",
        )
        assert cfg.api_key == "sk-test-123"
        assert cfg.base_url == "https://example.com/v1"
        assert cfg.model == "test-model"

    def test_from_env_falls_back_to_env_vars(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-env-key",
            "OPENAI_BASE_URL": "https://env.example.com/v1",
        }):
            cfg = ProviderConfig.from_env(model="mymodel")
        assert cfg.api_key == "sk-env-key"
        assert cfg.base_url == "https://env.example.com/v1"
        assert cfg.model == "mymodel"

    def test_from_env_missing_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key not provided"):
                ProviderConfig.from_env()

    def test_from_env_custom_env_var_names(self):
        with patch.dict(os.environ, {"MY_API_KEY": "sk-custom"}):
            cfg = ProviderConfig.from_env(
                api_key_env="MY_API_KEY",
                base_url="https://custom.com/v1",
                model="m",
            )
        assert cfg.api_key == "sk-custom"

    def test_frozen(self):
        cfg = ProviderConfig(api_key="k", base_url="u", model="m")
        with pytest.raises(AttributeError):
            cfg.api_key = "other"  # type: ignore[misc]


class TestTokenLogprobResult:
    """Tests for the TokenLogprobResult dataclass."""

    def test_has_top_k_true(self):
        r = TokenLogprobResult(
            token_logprobs=[],
            entropies=np.array([1.0]),
            perplexity=2.0,
            provider="p",
            model="m",
            top_k=5,
        )
        assert r.has_top_k() is True
        assert r.has_selected_only() is False

    def test_has_selected_only_true(self):
        r = TokenLogprobResult(
            token_logprobs=[],
            entropies=None,
            perplexity=2.0,
            provider="p",
            model="m",
            top_k=0,
        )
        assert r.has_top_k() is False
        assert r.has_selected_only() is True

    def test_warnings_default_empty(self):
        r = TokenLogprobResult(
            token_logprobs=[],
            entropies=None,
            perplexity=None,
            provider="p",
            model="m",
            top_k=0,
        )
        assert r.warnings == []


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


class TestParseOpenAILogprobs:
    """Tests for _parse_openai_logprobs."""

    def test_topk_parsed_correctly(self):
        resp = _response_with_topk()
        result = _parse_openai_logprobs(resp, "test", "model-1")
        assert len(result.tokens) == 2
        assert result.top_k >= 3
        assert result.has_top_k()
        assert result.provider == "test"
        assert result.model == "model-1"

    def test_selected_token_always_in_top_dict(self):
        resp = _make_openai_response(
            tokens=["A"],
            logprobs=[-0.1],
            top_logprobs_per_token=[[{"token": "B", "logprob": -1.0}]],
        )
        result = _parse_openai_logprobs(resp, "p", "m")
        # "A" should be added to top_logprobs even though it wasn't in the list
        assert "A" in result.tokens[0].top_logprobs

    def test_no_logprobs_returns_empty_tokens(self):
        resp = _response_no_logprobs()
        result = _parse_openai_logprobs(resp, "p", "m")
        assert len(result.tokens) == 0

    def test_empty_content(self):
        resp = _response_empty_content()
        result = _parse_openai_logprobs(resp, "p", "m")
        assert len(result.tokens) == 0

    def test_selected_logprobs_property(self):
        resp = _response_with_topk()
        result = _parse_openai_logprobs(resp, "p", "m")
        assert result.selected_logprobs == [-0.1, -0.5]

    def test_echo_used_flag(self):
        resp = _response_with_topk()
        result = _parse_openai_logprobs(resp, "p", "m", echo_used=True)
        assert result.echo_used is True


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider constructors
# ---------------------------------------------------------------------------


class TestOpenAICompatibleProviderConstructors:
    """Tests for from_preset and custom constructors."""

    def test_from_preset_openai(self):
        prov = OpenAICompatibleProvider.from_preset("openai", api_key="sk-test")
        assert prov.spec.name == "OpenAI"
        assert prov.api_key == "sk-test"

    def test_from_preset_together(self):
        prov = OpenAICompatibleProvider.from_preset("together", api_key="tk")
        assert prov.spec.max_top_k == 20
        assert prov.spec.echo_supported is True

    def test_from_preset_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider preset"):
            OpenAICompatibleProvider.from_preset("nonexistent", api_key="k")

    def test_from_preset_model_override(self):
        prov = OpenAICompatibleProvider.from_preset(
            "openai", api_key="k", model="gpt-4o"
        )
        assert prov.spec.model == "gpt-4o"

    def test_from_preset_base_url_override(self):
        prov = OpenAICompatibleProvider.from_preset(
            "openai", api_key="k", base_url="https://other.com/v1"
        )
        assert prov.spec.base_url == "https://other.com/v1"

    def test_from_preset_env_fallback(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"}):
            prov = OpenAICompatibleProvider.from_preset("openai")
        assert prov.api_key == "sk-env"

    def test_custom_constructor(self):
        prov = OpenAICompatibleProvider.custom(
            base_url="http://localhost:8080/v1",
            model="my-model",
            api_key="k",
            max_top_k=10,
        )
        assert prov.spec.name == "custom"
        assert prov.spec.max_top_k == 10
        assert prov.spec.base_url == "http://localhost:8080/v1"


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider.generate
# ---------------------------------------------------------------------------


class TestGenerate:
    """Tests for the generate method (mocked HTTP)."""

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_basic_generation(self, mock_call):
        mock_call.return_value = _response_with_topk()
        prov = OpenAICompatibleProvider.from_preset("openai", api_key="sk-test")
        result = prov.generate("Say hello")
        assert len(result.tokens) == 2
        assert result.has_top_k()
        mock_call.assert_called_once()
        # _call_chat_completions receives top_logprobs as a named arg
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["top_logprobs"] > 0

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_echo_rejected_for_openai(self, mock_call):
        prov = OpenAICompatibleProvider.from_preset("openai", api_key="sk-test")
        with pytest.raises(ValueError, match="does not support echo"):
            prov.generate("hello", echo=True)

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_echo_allowed_for_together(self, mock_call):
        mock_call.return_value = _response_with_topk()
        prov = OpenAICompatibleProvider.from_preset("together", api_key="tk")
        result = prov.generate("hello", echo=True)
        assert result.echo_used is True

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_top_k_clamped_to_provider_max(self, mock_call):
        mock_call.return_value = _response_with_topk()
        prov = OpenAICompatibleProvider.from_preset("fireworks", api_key="fk")
        prov.generate("hello", top_k=100)
        call_kwargs = mock_call.call_args[1]
        # fireworks max is 5
        assert call_kwargs["top_logprobs"] <= 5

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_messages_override(self, mock_call):
        mock_call.return_value = _response_with_topk()
        prov = OpenAICompatibleProvider.from_preset("openai", api_key="sk-test")
        msgs = [{"role": "system", "content": "You are helpful."}]
        prov.generate("hello", messages=msgs)
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["messages"] == msgs


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider.score_generation
# ---------------------------------------------------------------------------


class TestScoreGeneration:
    """Tests for score_generation with mocked HTTP responses."""

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_topk_returns_entropy(self, mock_call):
        mock_call.return_value = _response_with_topk()
        prov = OpenAICompatibleProvider.from_preset("openai", api_key="sk-test")
        result = prov.score_generation("Say hello")
        assert result.has_top_k()
        assert result.entropies is not None
        assert len(result.entropies) == 2
        assert all(e >= 0 for e in result.entropies)
        assert result.perplexity is not None
        assert result.perplexity >= 1.0
        assert result.warnings == []

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_selected_only_returns_perplexity_no_entropy(self, mock_call):
        mock_call.return_value = _response_selected_only()
        prov = OpenAICompatibleProvider.from_preset("openai", api_key="sk-test")
        result = prov.score_generation("Say OK")
        assert result.has_selected_only()
        assert result.entropies is None
        assert result.perplexity is not None
        assert result.top_k == 0
        assert len(result.warnings) > 0
        assert "CES scoring is not possible" in result.warnings[0]

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_no_logprobs_raises_capability_error(self, mock_call):
        mock_call.return_value = _response_no_logprobs()
        prov = OpenAICompatibleProvider.from_preset("openai", api_key="sk-test")
        with pytest.raises(ProviderCapabilityError) as exc_info:
            prov.score_generation("Say OK")
        assert "no logprob data" in str(exc_info.value).lower()
        assert exc_info.value.capability == "logprobs"
        assert exc_info.value.provider == "OpenAI"

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_empty_content_raises_capability_error(self, mock_call):
        mock_call.return_value = _response_empty_content()
        prov = OpenAICompatibleProvider.from_preset("openai", api_key="sk-test")
        with pytest.raises(ProviderCapabilityError):
            prov.score_generation("Say OK")

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_api_failure_raises_capability_error(self, mock_call):
        mock_call.side_effect = RuntimeError("connection refused")
        prov = OpenAICompatibleProvider.from_preset("openai", api_key="sk-test")
        with pytest.raises(ProviderCapabilityError) as exc_info:
            prov.score_generation("Say OK")
        assert "connection refused" in str(exc_info.value)

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_default_top_logprobs_20(self, mock_call):
        mock_call.return_value = _response_with_topk()
        prov = OpenAICompatibleProvider.from_preset("openai", api_key="sk-test")
        prov.score_generation("Say hello")
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["top_logprobs"] == 20

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_provider_max_caps_top_logprobs(self, mock_call):
        mock_call.return_value = _response_with_topk()
        prov = OpenAICompatibleProvider.from_preset("fireworks", api_key="fk")
        prov.score_generation("hello")
        call_kwargs = mock_call.call_args[1]
        # fireworks max_top_k is 5
        assert call_kwargs["top_logprobs"] == 5

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_messages_override(self, mock_call):
        mock_call.return_value = _response_with_topk()
        prov = OpenAICompatibleProvider.from_preset("openai", api_key="sk-test")
        msgs = [{"role": "user", "content": "test"}]
        prov.score_generation("ignored", messages=msgs)
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["messages"] == msgs


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider.check_health
# ---------------------------------------------------------------------------


class TestCheckHealth:
    """Tests for check_health."""

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_healthy(self, mock_call):
        mock_call.return_value = _response_with_topk()
        prov = OpenAICompatibleProvider.from_preset("openai", api_key="sk-test")
        health = prov.check_health()
        assert health["healthy"] is True
        assert health["provider"] == "OpenAI"

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_unhealthy_on_error(self, mock_call):
        mock_call.side_effect = RuntimeError("timeout")
        prov = OpenAICompatibleProvider.from_preset("openai", api_key="sk-test")
        health = prov.check_health()
        assert health["healthy"] is False
        assert "timeout" in health["error"]


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider.score_text
# ---------------------------------------------------------------------------


class TestScoreText:
    """Tests for score_text (echo-based)."""

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_uses_echo(self, mock_call):
        mock_call.return_value = _response_with_topk()
        prov = OpenAICompatibleProvider.from_preset("together", api_key="tk")
        prov.score_text("The capital of France is")
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs.get("echo") is True
        assert call_kwargs.get("max_tokens") == 1

    def test_rejected_for_openai(self):
        prov = OpenAICompatibleProvider.from_preset("openai", api_key="sk-test")
        with pytest.raises(ValueError, match="does not support echo"):
            prov.score_text("some text")


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider.generate_with_logprobs
# ---------------------------------------------------------------------------


class TestGenerateWithLogprobs:
    """Tests for generate_with_logprobs."""

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_delegates_to_generate(self, mock_call):
        mock_call.return_value = _response_with_topk()
        prov = OpenAICompatibleProvider.from_preset("openai", api_key="sk-test")
        result = prov.generate_with_logprobs("Say hello", max_tokens=50, top_k=5)
        assert len(result.tokens) == 2
        assert result.has_top_k()
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["max_tokens"] == 50
        assert call_kwargs["top_logprobs"] == 5

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_no_echo_sent(self, mock_call):
        """generate_with_logprobs must not send echo=True."""
        mock_call.return_value = _response_with_topk()
        prov = OpenAICompatibleProvider.from_preset("together", api_key="tk")
        prov.generate_with_logprobs("Say hello")
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs.get("echo") is not True


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider.score_output
# ---------------------------------------------------------------------------


class TestScoreOutput:
    """Tests for score_output (prompt + output echo scoring)."""

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_sends_prompt_and_output_as_messages(self, mock_call):
        mock_call.return_value = _response_with_topk()
        prov = OpenAICompatibleProvider.from_preset("together", api_key="tk")
        prov.score_output("What is 2+2?", "4")
        call_kwargs = mock_call.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is 2+2?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "4"
        assert call_kwargs.get("echo") is True
        assert call_kwargs.get("max_tokens") == 1

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_returns_completion_logprobs(self, mock_call):
        mock_call.return_value = _response_with_topk()
        prov = OpenAICompatibleProvider.from_preset("together", api_key="tk")
        result = prov.score_output("What is 2+2?", "4")
        assert isinstance(result, CompletionLogprobs)
        assert len(result.tokens) > 0

    def test_raises_capability_error_for_openai(self):
        """score_output must raise ProviderCapabilityError for providers without echo."""
        prov = OpenAICompatibleProvider.from_preset("openai", api_key="sk-test")
        with pytest.raises(ProviderCapabilityError) as exc_info:
            prov.score_output("What is 2+2?", "4")
        assert "echo" in str(exc_info.value).lower()
        assert exc_info.value.capability == "echo"
        assert exc_info.value.provider == "OpenAI"

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_top_k_passed_through(self, mock_call):
        mock_call.return_value = _response_with_topk()
        prov = OpenAICompatibleProvider.from_preset("together", api_key="tk")
        prov.score_output("prompt", "output", top_k=3)
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["top_logprobs"] == 3


# ---------------------------------------------------------------------------
# smoke_provider
# ---------------------------------------------------------------------------


class TestSmokeProvider:
    """Tests for the standalone smoke_provider function."""

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_pass_with_topk(self, mock_call):
        mock_call.return_value = _response_with_topk()
        result = smoke_provider(api_key="sk-test", provider="openai")
        assert result["status"] == "PASS"
        assert result["capabilities"]["logprobs_available"] is True
        assert result["capabilities"]["top_k_logprobs"] is True
        assert result["capabilities"]["max_top_k"] >= 2

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_selected_only_reports_ces_unavailable(self, mock_call):
        mock_call.return_value = _response_selected_only()
        result = smoke_provider(api_key="sk-test", provider="openai")
        assert result["status"] == "PASS"
        assert result["capabilities"]["logprobs_available"] is True
        assert result["capabilities"]["top_k_logprobs"] is False
        assert any("CES" in w for w in result["warnings"])

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_no_logprobs_raises(self, mock_call):
        mock_call.return_value = _response_no_logprobs()
        with pytest.raises(ProviderCapabilityError):
            smoke_provider(api_key="sk-test", provider="openai")

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_api_failure_raises(self, mock_call):
        mock_call.side_effect = RuntimeError("down")
        with pytest.raises(ProviderCapabilityError):
            smoke_provider(api_key="sk-test", provider="openai")


# ---------------------------------------------------------------------------
# TokenLogprobProvider protocol check
# ---------------------------------------------------------------------------


class TestTokenLogprobProviderProtocol:
    """Verify OpenAICompatibleProvider satisfies the TokenLogprobProvider protocol."""

    def test_implements_protocol(self):
        assert isinstance(
            OpenAICompatibleProvider.from_preset("openai", api_key="k"),
            TokenLogprobProvider,
        )

    def test_protocol_has_score_generation(self):
        assert hasattr(TokenLogprobProvider, "score_generation")

    def test_protocol_has_check_health(self):
        assert hasattr(TokenLogprobProvider, "check_health")


# ---------------------------------------------------------------------------
# ProviderSpec registry
# ---------------------------------------------------------------------------


class TestProviderSpecs:
    """Verify the provider registry."""

    def test_openai_spec_exists(self):
        assert "openai" in PROVIDER_SPECS

    def test_mimo_provider_preset_exists(self):
        assert "mimo" in PROVIDER_SPECS
        spec = PROVIDER_SPECS["mimo"]
        assert spec.name == "MIMO"
        assert spec.api_key_env == "MIMO_API_KEY"

    def test_all_specs_have_required_fields(self):
        for name, spec in PROVIDER_SPECS.items():
            assert spec.name, f"{name} missing name"
            assert spec.base_url, f"{name} missing base_url"
            assert spec.api_key_env, f"{name} missing api_key_env"
            assert spec.model, f"{name} missing model"
            assert spec.max_top_k >= 0, f"{name} has negative max_top_k"

    def test_frozen(self):
        spec = PROVIDER_SPECS["openai"]
        with pytest.raises(AttributeError):
            spec.name = "changed"  # type: ignore[misc]


class TestMimoProvider:
    """Tests for the MIMO provider preset."""

    def test_missing_mimo_key_has_clear_error(self, monkeypatch):
        monkeypatch.delenv("MIMO_API_KEY", raising=False)
        prov = OpenAICompatibleProvider.from_preset("mimo")
        assert prov.api_key == ""
        with pytest.raises(ValueError, match="MIMO_API_KEY"):
            prov.check_health()

    def test_mimo_from_preset_with_explicit_key(self):
        prov = OpenAICompatibleProvider.from_preset("mimo", api_key="mk-test")
        assert prov.spec.name == "MIMO"
        assert prov.api_key == "mk-test"
        assert prov.spec.api_key_env == "MIMO_API_KEY"

    def test_mimo_model_override(self):
        prov = OpenAICompatibleProvider.from_preset(
            "mimo", api_key="k", model="custom-model"
        )
        assert prov.spec.model == "custom-model"

    def test_mimo_base_url_override(self):
        prov = OpenAICompatibleProvider.from_preset(
            "mimo", api_key="k", base_url="https://mimo.example.com/v1"
        )
        assert prov.spec.base_url == "https://mimo.example.com/v1"


class TestEmptyApiKeyHandling:
    """Tests that empty API keys don't produce illegal Bearer headers."""

    @patch("hallucination_sentinel.providers.openai_compatible.httpx")
    def test_empty_api_key_does_not_send_illegal_bearer_header(self, mock_httpx):
        """When api_key is empty, no Authorization header should be sent."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _response_with_topk()
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        from hallucination_sentinel.providers.openai_compatible import _call_chat_completions

        _call_chat_completions(
            base_url="http://localhost:8080/v1",
            api_key="",
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        call_kwargs = mock_httpx.post.call_args
        headers = call_kwargs[1]["headers"] if "headers" in call_kwargs[1] else call_kwargs[0][2] if len(call_kwargs[0]) > 2 else {}
        # Must not contain Authorization with empty Bearer
        assert "Authorization" not in headers

    @patch("hallucination_sentinel.providers.openai_compatible.httpx")
    def test_nonempty_api_key_sends_bearer_header(self, mock_httpx):
        """When api_key is present, Authorization: Bearer should be sent."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _response_with_topk()
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        from hallucination_sentinel.providers.openai_compatible import _call_chat_completions

        _call_chat_completions(
            base_url="http://localhost:8080/v1",
            api_key="sk-test-key",
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        call_kwargs = mock_httpx.post.call_args
        headers = call_kwargs[1].get("headers", {})
        assert headers.get("Authorization") == "Bearer sk-test-key"


# ---------------------------------------------------------------------------
# Integration tests (skipped unless env vars are set)
# ---------------------------------------------------------------------------


requires_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping live integration test",
)


@requires_openai_key
class TestIntegrationOpenAI:
    """Live integration tests against the OpenAI API.

    Skipped unless OPENAI_API_KEY is set in the environment.
    """

    def test_generate_returns_logprobs(self):
        prov = OpenAICompatibleProvider.from_preset("openai")
        result = prov.generate("Say hello world", max_tokens=5)
        assert len(result.tokens) > 0
        assert all(t.logprob <= 0 for t in result.tokens)

    def test_score_generation_topk(self):
        prov = OpenAICompatibleProvider.from_preset("openai")
        result = prov.score_generation("Say hello", max_tokens=3)
        assert result.has_top_k()
        assert result.entropies is not None
        assert len(result.entropies) > 0

    def test_check_health(self):
        prov = OpenAICompatibleProvider.from_preset("openai")
        health = prov.check_health()
        assert health["healthy"] is True

    def test_smoke_provider(self):
        result = smoke_provider(provider="openai")
        assert result["status"] == "PASS"
        assert result["capabilities"]["logprobs_available"] is True


requires_together_key = pytest.mark.skipif(
    not os.environ.get("TOGETHER_API_KEY"),
    reason="TOGETHER_API_KEY not set; skipping live integration test",
)


@requires_together_key
class TestIntegrationTogether:
    """Live integration tests against Together AI."""

    def test_score_text(self):
        prov = OpenAICompatibleProvider.from_preset("together")
        result = prov.score_text("The capital of France is")
        assert len(result.tokens) > 0
        assert result.echo_used is True

    def test_score_generation_topk(self):
        prov = OpenAICompatibleProvider.from_preset("together")
        result = prov.score_generation("Say hello", max_tokens=3)
        assert result.has_top_k()
        assert result.perplexity is not None
