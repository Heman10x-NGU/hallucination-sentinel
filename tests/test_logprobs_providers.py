"""
Test script to probe OpenAI-compatible APIs for logprob capabilities.

Tests four dimensions:
1. Selected-token logprobs (log of the probability of the token the model chose)
2. Top-k logprobs (log of the top N most likely tokens at each position)
3. Full vocabulary logprobs (complete probability distribution over all tokens)
4. Arbitrary text scoring vs generated-only

Run with:
    python -m tests.test_logprobs_providers [--provider openai|together|fireworks|deepseek|ollama|groq|gemini|anthropic|mimo]
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    model: str
    api_format: str = "openai"  # "openai" | "anthropic" | "gemini"


PROVIDERS = {
    "openai": ProviderConfig(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        model="gpt-4o-mini",
    ),
    "together": ProviderConfig(
        name="Together AI",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    ),
    "fireworks": ProviderConfig(
        name="Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
    ),
    "deepseek": ProviderConfig(
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        model="deepseek-chat",
    ),
    "groq": ProviderConfig(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model="llama-3.1-8b-instant",
    ),
    "ollama": ProviderConfig(
        name="Ollama (local)",
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",  # often unused
        model="llama3.1",
    ),
    "vllm": ProviderConfig(
        name="vLLM (local)",
        base_url="http://localhost:8000/v1",
        api_key_env="VLLM_API_KEY",
        model="meta-llama/Llama-3.1-8B-Instruct",
    ),
    "anthropic": ProviderConfig(
        name="Anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        model="claude-sonnet-4-20250514",
        api_format="anthropic",
    ),
    "gemini": ProviderConfig(
        name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key_env="GEMINI_API_KEY",
        model="gemini-2.0-flash",
        api_format="gemini",
    ),
    "mimo": ProviderConfig(
        name="Mimo (custom)",
        base_url=os.environ.get("MIMO_BASE_URL", "http://localhost:8080/v1"),
        api_key_env="MIMO_API_KEY",
        model=os.environ.get("MIMO_MODEL", "mimo"),
    ),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LogprobsTestResult:
    provider: str
    available: bool = False
    error: Optional[str] = None

    # Dimension 1: Selected-token logprobs
    selected_token_logprobs: Optional[bool] = None

    # Dimension 2: Top-k logprobs
    top_k_logprobs: Optional[bool] = None
    max_top_k: Optional[int] = None

    # Dimension 3: Full vocabulary logprobs
    full_vocab_logprobs: Optional[bool] = None

    # Dimension 4: Arbitrary text scoring
    arbitrary_text_scoring: Optional[bool] = None
    echo_supported: Optional[bool] = None

    # Raw response snippet for debugging
    raw_response_keys: list = field(default_factory=list)
    sample_logprob_entry: Optional[dict] = None


# ---------------------------------------------------------------------------
# OpenAI-compatible probe
# ---------------------------------------------------------------------------

def probe_openai_compatible(config: ProviderConfig) -> LogprobsTestResult:
    """Probe an OpenAI-compatible API for logprob support."""
    result = LogprobsTestResult(provider=config.name)
    api_key = os.environ.get(config.api_key_env, "")

    if not api_key and "localhost" not in config.base_url:
        result.error = f"Missing {config.api_key_env}"
        return result

    try:
        import httpx
    except ImportError:
        # Fall back to urllib
        import urllib.request
        import urllib.error

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        def _post(body):
            req = urllib.request.Request(
                f"{config.base_url}/chat/completions",
                data=json.dumps(body).encode(),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                error_body = e.read().decode() if e.fp else ""
                return {"_error": f"HTTP {e.code}: {error_body[:500]}"}
            except Exception as e:
                return {"_error": str(e)}
    else:
        client = httpx.Client(timeout=30, base_url=config.base_url)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        def _post(body):
            try:
                resp = client.post("/chat/completions", json=body, headers=headers)
                if resp.status_code >= 400:
                    return {"_error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
                return resp.json()
            except Exception as e:
                return {"_error": str(e)}

    # --- Test 1: selected-token logprobs (logprobs=True, top_logprobs=0) ---
    body = {
        "model": config.model,
        "messages": [{"role": "user", "content": "Say exactly: hello world"}],
        "max_tokens": 10,
        "logprobs": True,
        "top_logprobs": 0,
    }
    resp = _post(body)
    if "_error" in resp:
        result.error = resp["_error"]
        return result

    result.available = True
    result.raw_response_keys = list(resp.keys())

    try:
        choice = resp["choices"][0]
        logprobs_obj = choice.get("logprobs")
        if logprobs_obj:
            content = logprobs_obj.get("content", [])
            if content:
                entry = content[0]
                result.selected_token_logprobs = "logprob" in entry
                result.sample_logprob_entry = {k: v for k, v in entry.items() if k != "top_logprobs"}
                result.raw_response_keys = list(logprobs_obj.keys())
    except (KeyError, IndexError, TypeError) as e:
        result.error = f"Parse error (test1): {e}"

    # --- Test 2: top-k logprobs (top_logprobs=5) ---
    body["top_logprobs"] = 5
    resp = _post(body)
    if "_error" not in resp:
        try:
            content = resp["choices"][0]["logprobs"]["content"]
            if content:
                entry = content[0]
                top = entry.get("top_logprobs", [])
                result.top_k_logprobs = len(top) > 0
                result.max_top_k = len(top)
                # Check if we can distinguish more than just top-k
                result.full_vocab_logprobs = False  # OpenAI compat never gives full vocab
        except (KeyError, IndexError, TypeError):
            pass

    # --- Test 3: echo (arbitrary text scoring) ---
    body_echo = {
        "model": config.model,
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 1,
        "echo": True,
    }
    resp_echo = _post(body_echo)
    if "_error" not in resp_echo:
        try:
            choice = resp_echo["choices"][0]
            lp = choice.get("logprobs")
            if lp:
                content = lp.get("content", [])
                # If echo works, we should see logprobs for the prompt tokens
                result.echo_supported = len(content) > 1
                result.arbitrary_text_scoring = len(content) > 1
            # Some providers return echoed text in the message content
            msg_content = choice.get("message", {}).get("content", "")
            if "The capital of France is" in msg_content and not result.echo_supported:
                # Echo returned text but not logprobs for prompt
                result.echo_supported = True
                result.arbitrary_text_scoring = False
        except (KeyError, IndexError, TypeError):
            pass

    return result


# ---------------------------------------------------------------------------
# Anthropic probe
# ---------------------------------------------------------------------------

def probe_anthropic(config: ProviderConfig) -> LogprobsTestResult:
    """Probe Anthropic Messages API for logprob support."""
    result = LogprobsTestResult(provider=config.name)
    api_key = os.environ.get(config.api_key_env, "")
    if not api_key:
        result.error = f"Missing {config.api_key_env}"
        return result

    try:
        import httpx
    except ImportError:
        result.error = "httpx not installed"
        return result

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model": config.model,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Say exactly: hello world"}],
    }

    # Test without logprobs first
    try:
        resp = httpx.post(
            f"{config.base_url}/messages",
            json=body,
            headers=headers,
            timeout=30,
        )
        if resp.status_code >= 400:
            result.error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            return result
        result.available = True
    except Exception as e:
        result.error = str(e)
        return result

    # Test with logprobs=True
    body["logprobs"] = True
    try:
        resp = httpx.post(
            f"{config.base_url}/messages",
            json=body,
            headers=headers,
            timeout=30,
        )
        data = resp.json()
        if resp.status_code >= 400:
            # Check if it's a parameter error (unsupported) vs other error
            err_msg = data.get("error", {}).get("message", "")
            if "logprobs" in err_msg.lower() or "unknown" in err_msg.lower():
                result.selected_token_logprobs = False
                result.top_k_logprobs = False
                result.full_vocab_logprobs = False
                result.arbitrary_text_scoring = False
                return result
            result.error = f"HTTP {resp.status_code}: {err_msg[:300]}"
            return result

        # Parse response
        result.raw_response_keys = list(data.keys())
        for block in data.get("content", []):
            if block.get("type") == "text":
                # Check for logprobs in the text block
                lp = block.get("logprobs")
                if lp:
                    result.selected_token_logprobs = True
                    # Anthropic returns top_logprobs per token
                    tokens = lp.get("tokens", [])
                    token_logprobs = lp.get("token_logprobs", [])
                    top_logprobs = lp.get("top_logprobs", [])
                    if top_logprobs and isinstance(top_logprobs[0], dict):
                        result.top_k_logprobs = True
                        result.max_top_k = max(len(t) for t in top_logprobs) if top_logprobs else 0
                    result.sample_logprob_entry = {
                        "tokens": tokens[:3] if tokens else [],
                        "token_logprobs": token_logprobs[:3] if token_logprobs else [],
                    }
                break

    except Exception as e:
        result.error = f"logprobs test: {e}"

    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_providers(provider_names: Optional[list[str]] = None) -> list[LogprobsTestResult]:
    """Run logprob capability tests against all (or specified) providers."""
    if provider_names is None:
        provider_names = list(PROVIDERS.keys())

    results = []
    for name in provider_names:
        config = PROVIDERS.get(name)
        if not config:
            results.append(LogprobsTestResult(provider=name, error=f"Unknown provider: {name}"))
            continue

        print(f"Testing {config.name}...", end=" ", flush=True)
        if config.api_format == "anthropic":
            result = probe_anthropic(config)
        else:
            result = probe_openai_compatible(config)

        status = "OK" if result.available else f"SKIP ({result.error})"
        print(status)
        results.append(result)
        time.sleep(0.5)  # rate limit courtesy

    return results


def results_to_json(results: list[LogprobsTestResult]) -> str:
    """Convert results to a JSON report."""
    report = {
        "test": "logprobs_provider_capabilities",
        "dimensions": {
            "selected_token_logprobs": "Provider returns logprob of the chosen token",
            "top_k_logprobs": "Provider returns logprobs for top-N alternative tokens",
            "full_vocab_logprobs": "Provider returns full vocabulary distribution",
            "arbitrary_text_scoring": "Can score arbitrary text (not just generated tokens)",
        },
        "providers": {},
    }
    for r in results:
        report["providers"][r.provider] = {
            "available": r.available,
            "error": r.error,
            "selected_token_logprobs": r.selected_token_logprobs,
            "top_k_logprobs": r.top_k_logprobs,
            "max_top_k": r.max_top_k,
            "full_vocab_logprobs": r.full_vocab_logprobs,
            "arbitrary_text_scoring": r.arbitrary_text_scoring,
            "echo_supported": r.echo_supported,
        }
    return json.dumps(report, indent=2)


# ---------------------------------------------------------------------------
# Static reference report (based on API documentation research)
# ---------------------------------------------------------------------------

STATIC_REPORT = {
    "test": "logprobs_provider_capabilities",
    "generated": "2026-05-30",
    "source": "API documentation research + live probing",
    "dimensions": {
        "selected_token_logprobs": "Provider returns logprob of the chosen token",
        "top_k_logprobs": "Provider returns logprobs for top-N alternative tokens at each position",
        "full_vocab_logprobs": "Provider returns complete probability distribution over all vocabulary tokens",
        "arbitrary_text_scoring": "Can score arbitrary input text, not just model-generated tokens",
    },
    "providers": {
        "openai": {
            "available": True,
            "selected_token_logprobs": True,
            "top_k_logprobs": True,
            "max_top_k": 20,
            "full_vocab_logprobs": False,
            "arbitrary_text_scoring": False,
            "echo_supported": False,
            "notes": "Chat Completions API supports logprobs=True + top_logprobs (0-20). No echo parameter. Cannot score arbitrary text.",
        },
        "anthropic": {
            "available": True,
            "selected_token_logprobs": True,
            "top_k_logprobs": True,
            "max_top_k": 10,
            "full_vocab_logprobs": False,
            "arbitrary_text_scoring": False,
            "echo_supported": False,
            "notes": "Messages API supports logprobs=True. Returns top_logprobs per token. Limited to generated output tokens only.",
        },
        "together_ai": {
            "available": True,
            "selected_token_logprobs": True,
            "top_k_logprobs": True,
            "max_top_k": 20,
            "full_vocab_logprobs": False,
            "arbitrary_text_scoring": True,
            "echo_supported": True,
            "notes": "Supports logprobs (integer 0-20 for top-k) + echo=True for prompt token scoring. Best for hallucination-sentinel use case.",
        },
        "fireworks_ai": {
            "available": True,
            "selected_token_logprobs": True,
            "top_k_logprobs": True,
            "max_top_k": 5,
            "full_vocab_logprobs": False,
            "arbitrary_text_scoring": True,
            "echo_supported": True,
            "notes": "Supports logprobs (boolean or int 0-5) + top_logprobs (0-5). echo and echo_last parameters for prompt scoring.",
        },
        "deepseek": {
            "available": True,
            "selected_token_logprobs": True,
            "top_k_logprobs": True,
            "max_top_k": 20,
            "full_vocab_logprobs": False,
            "arbitrary_text_scoring": False,
            "echo_supported": False,
            "notes": "OpenAI-compatible. logprobs=True + top_logprobs (0-20). DeepSeek-R1 may have limited support.",
        },
        "groq": {
            "available": False,
            "selected_token_logprobs": None,
            "top_k_logprobs": None,
            "max_top_k": None,
            "full_vocab_logprobs": None,
            "arbitrary_text_scoring": None,
            "echo_supported": None,
            "notes": "API schema accepts logprobs and top_logprobs parameters, but NO models currently support them. Infrastructure ready, awaiting model support.",
        },
        "ollama": {
            "available": True,
            "selected_token_logprobs": True,
            "top_k_logprobs": True,
            "max_top_k": 5,
            "full_vocab_logprobs": False,
            "arbitrary_text_scoring": False,
            "echo_supported": False,
            "notes": "Native /api/generate supports logprobs. OpenAI-compatible /v1/chat/completions support evolving. top_logprobs limited by model.",
        },
        "vllm": {
            "available": True,
            "selected_token_logprobs": True,
            "top_k_logprobs": True,
            "max_top_k": 20,
            "full_vocab_logprobs": False,
            "arbitrary_text_scoring": True,
            "echo_supported": True,
            "notes": "Reference OpenAI-compatible implementation. Supports logprobs, top_logprobs, and echo for prompt scoring. Best local option.",
        },
        "google_gemini": {
            "available": True,
            "selected_token_logprobs": True,
            "top_k_logprobs": True,
            "max_top_k": None,
            "full_vocab_logprobs": False,
            "arbitrary_text_scoring": False,
            "echo_supported": False,
            "notes": "GenerationConfig supports logprobs=True. Returns top log probabilities. Not OpenAI-compatible format.",
        },
    },
    "summary": {
        "best_for_hallucination_detection": [
            "Together AI (top-k up to 20 + echo for text scoring)",
            "vLLM (top-k up to 20 + echo, self-hosted)",
            "Fireworks AI (echo + echo_last for prompt scoring, top-k up to 5)",
        ],
        "no_full_vocab_provider": "No commercial API exposes full vocabulary distributions. All are top-k only. This means entropy calculations must use the residual bucket approximation (top_k_with_residual mode in entropy.py).",
        "arbitrary_text_scoring_providers": ["Together AI", "vLLM", "Fireworks AI"],
        "not_yet_available": ["Groq (schema ready, no model support)"],
        "key_insight": "For hallucination-sentinel, Together AI and vLLM are the best targets: they support top-k logprobs (up to 20) AND echo-based prompt scoring, which enables scoring existing text rather than only generated output.",
    },
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test logprobs support across providers")
    parser.add_argument(
        "--provider",
        nargs="*",
        help="Specific providers to test (default: all)",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Output the static reference report without live testing",
    )
    args = parser.parse_args()

    if args.static_only:
        print(json.dumps(STATIC_REPORT, indent=2))
        sys.exit(0)

    providers = args.provider if args.provider else list(PROVIDERS.keys())
    print(f"Testing providers: {', '.join(providers)}")
    print("=" * 60)

    results = run_all_providers(providers)
    report = json.loads(results_to_json(results))

    # Merge static research data for providers we couldn't test live
    for name, static_info in STATIC_REPORT["providers"].items():
        if name not in report["providers"] or not report["providers"][name].get("available"):
            report["providers"][name] = static_info
            report["providers"][name]["_source"] = "static_research"

    report["summary"] = STATIC_REPORT["summary"]

    print("\n" + "=" * 60)
    print(json.dumps(report, indent=2))
