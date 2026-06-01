# Provider Logprobs Compatibility

Hallucination Sentinel **requires** token-level log probabilities (logprobs) from
the model provider. Without logprobs, there is no entropy signal to analyze.

## How Logprobs Are Used

The CES algorithm needs per-token entropy values. Entropy is computed from the
probability distribution over the vocabulary at each token position:

```
h_t = -sum_v p_t(v) * log(p_t(v))
```

This requires knowing p_t(v) for each token v, which comes from logprobs.

### Three Capability Levels

| Level | What you get | What works |
|-------|-------------|------------|
| Full logits | Probability for every token in vocabulary | Exact entropy, CES, all baselines |
| Top-k logprobs | Logprob for top-k tokens (e.g. k=20) | Approximate entropy with residual bucket, CES, all baselines |
| Selected-token only | Logprob for only the token the model chose | Perplexity only. **CES unavailable.** |

## Provider Compatibility Table

| Provider | Preset | Top-k | Max k | Echo | Arbitrary Text Scoring | Status |
|----------|--------|-------|-------|------|----------------------|--------|
| OpenAI | `openai` | Yes | 20 | No | No | Tested |
| Together AI | `together` | Yes | 20 | Yes | Yes | Tested |
| Fireworks AI | `fireworks` | Yes | 5 | Yes | Yes | Tested |
| DeepSeek | `deepseek` | Yes | 20 | No | No | Tested |
| vLLM | `vllm` | Yes | 20 | Yes | Yes | Tested |
| Ollama | `ollama` | Limited | 5 | No | No | Experimental |
| Groq | `groq` | No | 0 | No | No | **No logprob support** |

### What Each Column Means

- **Top-k**: Can the provider return logprobs for the top-k alternative tokens at
  each position? Required for CES.
- **Max k**: Maximum number of top-k alternatives the provider returns. Higher is
  better for entropy approximation. The paper uses k=20.
- **Echo**: Can the provider score arbitrary existing text by "echoing" it back
  through the model? Required for scoring text you did not generate yourself.
- **Arbitrary Text Scoring**: Can you score any text, or only text the model
  generates? Echo support implies arbitrary text scoring.

## Smoke Testing a Provider

Before using a provider, run the smoke test to verify logprob support:

```bash
sentinel smoke-provider --provider openai
sentinel smoke-provider --provider together
sentinel smoke-provider --provider vllm --base-url http://localhost:8000/v1
```

The smoke test reports:
- Whether the provider is reachable
- Whether logprobs are returned
- Whether top-k alternatives are available
- Whether echo-based text scoring is supported

Run this before relying on a provider for CES scoring.

## Provider Details

### OpenAI

- Supports `logprobs=True` with `top_logprobs` (0-20)
- Does NOT support echo (cannot score arbitrary existing text)
- To score existing text, you must regenerate it from the same prompt with
  `temperature=0` and compare
- Works with: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`

### Together AI

- Supports `logprobs` (integer 0-20) and `echo=True`
- Can score arbitrary text via echo
- Good choice for both generation and post-hoc scoring
- Works with: Llama, Mistral, and other hosted models

### Fireworks AI

- Supports `logprobs` (bool or int 0-5) and `echo`/`echo_last`
- Top-k limited to 5 (lower entropy approximation quality)
- Can score arbitrary text via echo
- Works with: Llama, Mixtral, and other hosted models

### DeepSeek

- OpenAI-compatible API with `logprobs=True` and `top_logprobs` (0-20)
- Does NOT support echo
- Works with: `deepseek-chat`, `deepseek-coder`

### vLLM

- Reference OpenAI-compatible implementation
- Supports `logprobs`, `top_logprobs` (0-20), and `echo`
- Can score arbitrary text via echo
- Self-hosted: you control the model and server

### Ollama

- Native `/api/generate` has logprobs
- OpenAI-compatible endpoint support is evolving
- Top-k support is limited and version-dependent
- Marked as experimental: verify with smoke test before relying on it

### Groq

- The API schema accepts `logprobs` parameters but NO models return logprob data
  as of this writing
- Will fail the smoke test
- Do not use for CES scoring

## Custom / OpenAI-Compatible Providers

Any provider that implements the OpenAI `/v1/chat/completions` spec with
`logprobs` and `top_logprobs` parameters should work. Use the `custom` preset:

```python
from hallucination_sentinel.providers import OpenAICompatibleProvider

provider = OpenAICompatibleProvider.custom(
    base_url="https://your-api.example.com/v1",
    model="your-model",
    api_key="your-key",
    max_top_k=20,
    echo_supported=False,
)
```

Or use the CLI:

```bash
sentinel smoke-provider \
  --provider vllm \
  --base-url https://your-api.example.com/v1 \
  --model your-model
```

## HuggingFace / Local Models

Local models via HuggingFace Transformers give you full access to logits, which
is the ideal case. You get exact entropy (not approximate) from the full
vocabulary distribution.

This requires:
- `transformers` and `torch` installed. From this GitHub repo:
  `pip install "hallucination-sentinel[huggingface] @ git+https://github.com/Heman10x-NGU/hallucination-sentinel.git"`
- Enough GPU/CPU memory to load the model
- A teacher-forced forward pass for arbitrary text scoring

HuggingFace support is planned for a future release. Use vLLM or Ollama for
local model serving in the meantime.

## Troubleshooting

### "Provider returned no logprob data"

The provider or model does not support logprobs. Try a different provider or model.

### "Only selected-token logprobs returned"

The provider returns logprobs for only the chosen token, not alternatives.
CES cannot be computed. Perplexity is still available as a baseline.

### "Top-k logprobs available but k is low"

The provider returns fewer than 20 alternatives. Entropy is computed from the
available tokens plus a residual bucket. Lower k means less accurate entropy
approximation. The tool warns about this.

### "Echo not supported"

The provider cannot score arbitrary existing text. You can only score text that
the model generates in the same API call. To score existing text, regenerate it
from the same prompt with `temperature=0`.
