"""
OpenAI-compatible chat completions provider.

Supports any API that implements the OpenAI /v1/chat/completions spec
with logprobs and top_logprobs parameters.

Tested providers:
    - OpenAI (top_logprobs up to 20)
    - Together AI (top_logprobs up to 20, echo for prompt scoring)
    - Fireworks AI (top_logprobs up to 5, echo/echo_last for prompt scoring)
    - DeepSeek (top_logprobs up to 20)
    - vLLM (top_logprobs up to 20, echo for prompt scoring)
    - Ollama (limited, evolving)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from .base import (
    BaseProvider,
    CompletionLogprobs,
    ProviderCapabilityError,
    ProviderConfig,
    TokenLogprob,
    TokenLogprobResult,
)

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TokenLogprobProvider(Protocol):
    """Protocol for providers that extract token logprobs from generations.

    Implementations must call the underlying API with logprobs enabled,
    extract per-token logprob data, and compute entropy where possible.
    """

    def score_generation(
        self,
        prompt: str,
        *,
        max_tokens: int = 100,
        messages: Optional[list[dict[str, str]]] = None,
    ) -> TokenLogprobResult:
        """Generate text and return logprobs with entropy metrics.

        Args:
            prompt: User prompt (ignored if messages is provided).
            max_tokens: Max tokens to generate.
            messages: Full message list (overrides prompt).

        Returns:
            TokenLogprobResult with per-token data and entropy where available.

        Raises:
            ProviderCapabilityError: If logprobs are completely unavailable.
        """
        ...

    def check_health(self) -> dict[str, Any]:
        """Quick health check: can we reach the API and get logprobs?"""
        ...


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderSpec:
    """Static description of a provider's logprob capabilities."""

    name: str
    base_url: str
    api_key_env: str
    model: str
    max_top_k: int
    echo_supported: bool = False
    notes: str = ""


PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        model="gpt-4o-mini",
        max_top_k=20,
        echo_supported=False,
        notes="logprobs=True + top_logprobs (0-20). No echo, no arbitrary text scoring.",
    ),
    "together": ProviderSpec(
        name="Together AI",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        max_top_k=20,
        echo_supported=True,
        notes="logprobs (int 0-20) + echo=True for prompt token scoring.",
    ),
    "fireworks": ProviderSpec(
        name="Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        max_top_k=5,
        echo_supported=True,
        notes="logprobs (bool or int 0-5) + echo/echo_last for prompt scoring.",
    ),
    "deepseek": ProviderSpec(
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        model="deepseek-chat",
        max_top_k=20,
        echo_supported=False,
        notes="OpenAI-compatible. logprobs=True + top_logprobs (0-20).",
    ),
    "vllm": ProviderSpec(
        name="vLLM",
        base_url="http://localhost:8000/v1",
        api_key_env="VLLM_API_KEY",
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_top_k=20,
        echo_supported=True,
        notes="Reference OpenAI-compatible impl. Supports echo for prompt scoring.",
    ),
    "ollama": ProviderSpec(
        name="Ollama",
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        model="llama3.1",
        max_top_k=5,
        echo_supported=False,
        notes="Native /api/generate has logprobs. OpenAI-compat support evolving.",
    ),
    "groq": ProviderSpec(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model="llama-3.1-8b-instant",
        max_top_k=0,
        echo_supported=False,
        notes="Schema accepts logprobs but NO models support them yet.",
    ),
}


# ---------------------------------------------------------------------------
# API caller
# ---------------------------------------------------------------------------


def _call_chat_completions(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 10,
    top_logprobs: int = 0,
    echo: bool = False,
    extra_body: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Call the /v1/chat/completions endpoint and return the raw JSON response."""
    if httpx is None:
        raise ImportError("httpx is required: pip install httpx")

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    if echo:
        body["echo"] = True
    if extra_body:
        body.update(extra_body)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = httpx.post(
        f"{base_url}/chat/completions",
        json=body,
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _parse_openai_logprobs(
    response: dict[str, Any],
    provider_name: str,
    model: str,
    echo_used: bool = False,
) -> CompletionLogprobs:
    """Parse an OpenAI-style chat completion response with logprobs."""
    choice = response["choices"][0]
    logprobs_obj = choice.get("logprobs") or {}
    content = logprobs_obj.get("content", [])

    tokens: list[TokenLogprob] = []
    max_k = 0

    for entry in content:
        top = entry.get("top_logprobs", [])
        top_dict = {}
        for alt in top:
            tok = alt.get("token", "")
            lp = alt.get("logprob", 0.0)
            top_dict[tok] = lp

        # The selected token may or may not be in top_logprobs; always include it
        selected_tok = entry.get("token", "")
        selected_lp = entry.get("logprob", 0.0)
        if selected_tok and selected_tok not in top_dict:
            top_dict[selected_tok] = selected_lp

        max_k = max(max_k, len(top_dict))

        tokens.append(TokenLogprob(
            token=selected_tok,
            logprob=selected_lp,
            token_id=entry.get("token_id"),
            top_logprobs=top_dict,
        ))

    return CompletionLogprobs(
        tokens=tokens,
        provider=provider_name,
        model=model,
        top_k=max_k,
        echo_used=echo_used,
    )


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    """
    High-level provider for OpenAI-compatible chat completions with logprobs.

    Reads api_key, base_url, model from constructor args or environment
    variables (OPENAI_API_KEY, OPENAI_BASE_URL).  Default top_logprobs=20.

    Usage:
        provider = OpenAICompatibleProvider.from_preset("together")
        result = provider.generate("What is the capital of France?")
        print(result.selected_logprobs)
        print(result.topk_logprobs)

        # score_generation returns entropy + perplexity
        sg = provider.score_generation("Say hello")
        if sg.has_top_k():
            print(sg.entropies)
    """

    spec: ProviderSpec
    api_key: str
    client: Any = None  # httpx.Client, lazily created

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(
        cls,
        preset: str,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> OpenAICompatibleProvider:
        """Create from a named preset (e.g. 'openai', 'together', 'fireworks').

        api_key falls back to the env var named in the spec's api_key_env.
        base_url and model fall back to the spec defaults.
        """
        spec = PROVIDER_SPECS.get(preset)
        if spec is None:
            raise ValueError(
                f"Unknown provider preset '{preset}'. "
                f"Available: {', '.join(PROVIDER_SPECS)}"
            )
        if model or base_url:
            spec = ProviderSpec(
                name=spec.name,
                base_url=base_url or spec.base_url,
                api_key_env=spec.api_key_env,
                model=model or spec.model,
                max_top_k=spec.max_top_k,
                echo_supported=spec.echo_supported,
                notes=spec.notes,
            )
        key = api_key or os.environ.get(spec.api_key_env, "")
        return cls(spec=spec, api_key=key)

    @classmethod
    def custom(
        cls,
        base_url: str,
        model: str,
        api_key: str,
        *,
        max_top_k: int = 20,
        echo_supported: bool = False,
    ) -> OpenAICompatibleProvider:
        """Create a custom provider (e.g. for MIMO or other OpenAI-compatible APIs)."""
        spec = ProviderSpec(
            name="custom",
            base_url=base_url,
            api_key_env="",
            model=model,
            max_top_k=max_top_k,
            echo_supported=echo_supported,
        )
        return cls(spec=spec, api_key=api_key)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 100,
        top_k: Optional[int] = None,
        echo: bool = False,
        messages: Optional[list[dict[str, str]]] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> CompletionLogprobs:
        """
        Generate text and return normalized logprob data.

        Args:
            prompt: User prompt (ignored if messages is provided).
            max_tokens: Max tokens to generate.
            top_k: Number of top alternatives to request (default: provider max).
            echo: Whether to include prompt token logprobs (requires echo_supported).
            messages: Full message list (overrides prompt).
            extra_body: Additional fields to pass in the API request body.

        Returns:
            CompletionLogprobs with normalized token data.
        """
        if echo and not self.spec.echo_supported:
            raise ValueError(
                f"Provider '{self.spec.name}' does not support echo-based prompt scoring. "
                f"Use a provider with echo_supported=True (e.g. together, vllm, fireworks)."
            )

        k = min(top_k or self.spec.max_top_k, self.spec.max_top_k)

        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        raw = _call_chat_completions(
            base_url=self.spec.base_url,
            api_key=self.api_key,
            model=self.spec.model,
            messages=messages,
            max_tokens=max_tokens,
            top_logprobs=k,
            echo=echo,
            extra_body=extra_body,
        )

        return _parse_openai_logprobs(
            raw,
            provider_name=self.spec.name,
            model=self.spec.model,
            echo_used=echo,
        )

    def score_text(
        self,
        text: str,
        *,
        top_k: Optional[int] = None,
    ) -> CompletionLogprobs:
        """
        Score arbitrary text by using echo to get prompt token logprobs.

        Only works with providers that support echo (together, vllm, fireworks).

        Args:
            text: The text to score.
            top_k: Number of top alternatives per position.

        Returns:
            CompletionLogprobs for the input text tokens.
        """
        return self.generate(
            text,
            max_tokens=1,
            top_k=top_k,
            echo=True,
        )

    def score_generation(
        self,
        prompt: str,
        *,
        max_tokens: int = 100,
        messages: Optional[list[dict[str, str]]] = None,
    ) -> TokenLogprobResult:
        """Generate text and return logprobs with entropy metrics.

        Requests top_logprobs=20 by default.  Handles three capability levels:

        1. **Full top-k available** -- entropy is computed per token.
        2. **Only selected-token logprobs** -- perplexity is computed; entropy
           is unavailable and a warning is emitted.  CES cannot be computed.
        3. **No logprobs at all** -- raises ProviderCapabilityError.

        Args:
            prompt: User prompt (ignored if messages is provided).
            max_tokens: Max tokens to generate.
            messages: Full message list (overrides prompt).

        Returns:
            TokenLogprobResult with per-token data and metrics.

        Raises:
            ProviderCapabilityError: If the API returns no logprob data at all.
        """
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        # Request top_logprobs=20 (or provider max, whichever is smaller)
        requested_top_k = min(20, self.spec.max_top_k)

        try:
            raw = _call_chat_completions(
                base_url=self.spec.base_url,
                api_key=self.api_key,
                model=self.spec.model,
                messages=messages,
                max_tokens=max_tokens,
                top_logprobs=requested_top_k,
            )
        except Exception as exc:
            raise ProviderCapabilityError(
                f"API call failed for provider '{self.spec.name}': {exc}",
                provider=self.spec.name,
            ) from exc

        # Parse response
        choice = raw["choices"][0]
        logprobs_obj = choice.get("logprobs")
        content = (logprobs_obj or {}).get("content", [])

        if not content:
            raise ProviderCapabilityError(
                f"Provider '{self.spec.name}' returned no logprob data. "
                "The model or endpoint may not support logprobs. "
                "Cannot perform any uncertainty scoring.",
                capability="logprobs",
                provider=self.spec.name,
            )

        # Extract token logprobs
        tokens: list[TokenLogprob] = []
        warnings: list[str] = []
        has_top_k_data = False

        for entry in content:
            top = entry.get("top_logprobs", [])
            top_dict: dict[str, float] = {}
            for alt in top:
                tok = alt.get("token", "")
                lp = alt.get("logprob", 0.0)
                top_dict[tok] = lp

            selected_tok = entry.get("token", "")
            selected_lp = entry.get("logprob", 0.0)
            if selected_tok and selected_tok not in top_dict:
                top_dict[selected_tok] = selected_lp

            if len(top_dict) > 1:
                has_top_k_data = True

            tokens.append(TokenLogprob(
                token=selected_tok,
                logprob=selected_lp,
                token_id=entry.get("token_id"),
                top_logprobs=top_dict,
            ))

        # Determine capability level and compute metrics
        entropies: Optional[np.ndarray] = None
        perplexity: Optional[float] = None
        effective_top_k = 0

        if has_top_k_data:
            # Full top-k: compute per-token entropy from top-k logprobs
            effective_top_k = max(len(t.top_logprobs) for t in tokens)
            per_token_entropies = []
            for t in tokens:
                if t.top_logprobs:
                    probs = np.array(list(t.top_logprobs.values()), dtype=np.float64)
                    probs = np.exp(probs)
                    probs = np.clip(probs, 1e-10, None)
                    probs = probs / probs.sum()
                    per_token_entropies.append(-float(np.sum(probs * np.log(probs))))
                else:
                    per_token_entropies.append(0.0)
            entropies = np.array(per_token_entropies, dtype=np.float64)
        else:
            # Only selected-token logprobs: can compute perplexity, not entropy
            warnings.append(
                f"Provider '{self.spec.name}' returned only selected-token logprobs "
                "(no top-k alternatives).  Entropy cannot be computed; only "
                "perplexity baseline is available.  CES scoring is not possible."
            )

        # Always compute perplexity when we have selected-token logprobs
        if tokens:
            sel_logprobs = np.array(
                [t.logprob for t in tokens], dtype=np.float64
            )
            perplexity = float(np.exp(-np.mean(sel_logprobs)))

        return TokenLogprobResult(
            token_logprobs=tokens,
            entropies=entropies,
            perplexity=perplexity,
            provider=self.spec.name,
            model=self.spec.model,
            top_k=effective_top_k,
            warnings=warnings,
        )

    def check_health(self) -> dict[str, Any]:
        """Quick health check: can we reach the API and get logprobs?"""
        try:
            result = self.generate("Say OK", max_tokens=2, top_k=1)
            return {
                "healthy": True,
                "provider": self.spec.name,
                "model": self.spec.model,
                "top_k_available": result.has_top_k(),
                "token_count": len(result.tokens),
            }
        except Exception as e:
            return {
                "healthy": False,
                "provider": self.spec.name,
                "error": str(e),
            }


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------


def smoke_provider(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    provider: str = "openai",
) -> dict[str, Any]:
    """Run a smoke test on a provider to check logprob capabilities.

    Makes a minimal generation and reports what capabilities are available.

    Args:
        api_key: API key (falls back to env var for the preset).
        base_url: Base URL override.
        model: Model override.
        provider: Provider preset name.

    Returns:
        Dict with status, capabilities, and any warnings or errors.

    Raises:
        ProviderCapabilityError: If logprobs are completely unavailable.
    """
    prov = OpenAICompatibleProvider.from_preset(
        provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    result: dict[str, Any] = {
        "provider": prov.spec.name,
        "preset": provider,
        "base_url": base_url or prov.spec.base_url,
        "model": model or prov.spec.model,
        "status": "PASS",
        "capabilities": {
            "logprobs_available": False,
            "top_k_logprobs": False,
            "max_top_k": 0,
            "echo_supported": prov.spec.echo_supported,
        },
        "warnings": [],
    }

    # Step 1: Basic health check
    health = prov.check_health()
    if not health.get("healthy"):
        result["status"] = "FAIL"
        result["error"] = health.get("error", "Unknown error")
        raise ProviderCapabilityError(
            f"Provider '{prov.spec.name}' health check failed: {result['error']}",
            capability="logprobs",
            provider=prov.spec.name,
        )

    # Step 2: score_generation to check full capability
    try:
        sg = prov.score_generation("Say OK", max_tokens=2)
        result["capabilities"]["logprobs_available"] = True

        if sg.has_top_k():
            result["capabilities"]["top_k_logprobs"] = True
            result["capabilities"]["max_top_k"] = sg.top_k
        else:
            result["capabilities"]["top_k_logprobs"] = False
            result["capabilities"]["max_top_k"] = 0
            result["warnings"].extend(sg.warnings)
            result["warnings"].append(
                "CES scoring not available: only selected-token logprobs returned."
            )

    except ProviderCapabilityError:
        raise
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = str(exc)
        raise ProviderCapabilityError(
            f"Smoke test failed for '{prov.spec.name}': {exc}",
            provider=prov.spec.name,
        ) from exc

    return result
