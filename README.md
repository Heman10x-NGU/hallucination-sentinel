# Hallucination Sentinel

**Single-pass uncertainty firewall for LLM outputs.**

Flags generations whose entropy profile looks inconsistent with calibrated
faithful outputs. Implements the Calibrated Entropy Score (CES) method from
Villani et al., 2026:
["Entropy Distribution as a Fingerprint for Hallucinations in Generative Models"](https://arxiv.org/abs/2605.28264).

---

> **This is a risk signal, not a truth oracle.** The CES score is NOT a
> hallucination probability. A low score does not prove text is correct.
> A high score does not prove text is wrong. Read [Limitations](#limitations)
> before using this in production.

---

## What This Does

1. Takes per-token log probabilities from an LLM (via API or local model)
2. Computes Shannon entropy at each token position
3. Compares the entropy profile against a calibrated reference distribution
4. Returns a CES ranking score and risk level (LOW / MEDIUM / HIGH / CRITICAL)
5. Optionally routes the output through a policy engine (allow / warn / block)

## What This Does NOT Do

- Detect hallucinations with certainty
- Work without logprobs or logits from the model
- Fact-check claims against external sources
- Guarantee that LOW-risk outputs are correct
- Catch confident falsehoods (low entropy = high confidence, even when wrong)

## Requirements

**Logprobs are mandatory.** The provider must expose token-level log
probabilities. See [Provider Compatibility](docs/provider_logprobs.md) for a
provider-by-provider table.

The core library requires Python >= 3.10, numpy, scipy, click, and rich.

## Quick Start

### Install

```bash
# Install from GitHub
pip install "git+https://github.com/Heman10x-NGU/hallucination-sentinel.git"

# With OpenAI provider support
pip install "hallucination-sentinel[openai] @ git+https://github.com/Heman10x-NGU/hallucination-sentinel.git"

# With HuggingFace local model support
pip install "hallucination-sentinel[huggingface] @ git+https://github.com/Heman10x-NGU/hallucination-sentinel.git"

# Everything
pip install "hallucination-sentinel[all] @ git+https://github.com/Heman10x-NGU/hallucination-sentinel.git"
```

### Offline Demo (no API key needed)

Run the full pipeline offline using pre-computed entropy sequences:

```bash
# 1. Create a toy calibration dataset
echo '{"entropy": [0.1, 0.2, 0.15, 0.3, 0.1, 0.2]}' > cal.jsonl
echo '{"entropy": [0.15, 0.25, 0.2, 0.18, 0.12, 0.22]}' >> cal.jsonl

# 2. Build a calibration artifact
sentinel calibrate --input cal.jsonl --output calibration.json

# 3. Score a pre-computed entropy sequence
echo '{"entropy": [0.5, 1.2, 2.8, 3.5, 2.1, 0.8]}' > score_input.json
sentinel score --entropy-json score_input.json --calibration calibration.json
```

Output:

```json
{
  "ces_score": 0.954321,
  "risk_level": "CRITICAL",
  "cdf_mean": 0.98,
  "cdf_max": 0.93,
  "mean_entropy": 1.816667,
  "max_entropy": 3.5,
  "token_count": 6,
  "warnings": [
    "short_text: only 6 tokens.  CES reliability degrades for very short generations."
  ]
}
```

### Real Provider Smoke Test

Test whether a provider supports logprobs before relying on it:

```bash
# Test OpenAI
export OPENAI_API_KEY="<your-openai-api-key>"
sentinel smoke-provider --provider openai

# Test Together AI
export TOGETHER_API_KEY="<your-together-api-key>"
sentinel smoke-provider --provider together

# Test a local vLLM server
sentinel smoke-provider --provider vllm --base-url http://localhost:8000/v1
```

The smoke test reports whether logprobs, top-k alternatives, and echo-based
text scoring are available.

### Score With a Real Provider

After the smoke test passes:

```bash
sentinel score-provider \
  --prompt prompt.txt \
  --output output.txt \
  --provider together \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
  --calibration calibration.json
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `sentinel calibrate` | Build calibration artifact from JSONL entropy data |
| `sentinel inspect-calibration` | Print calibration metadata |
| `sentinel score` | Score a pre-computed entropy sequence (offline) |
| `sentinel score-provider` | Score text using a provider's logprobs |
| `sentinel eval` | Evaluate CES against baseline metrics on labeled data |
| `sentinel smoke-provider` | Test provider logprob support |

## Python API

```python
import numpy as np
from hallucination_sentinel import compute_ces, load_calibration

# Load calibration
artifact = load_calibration("calibration.json")

# Score from entropy sequence
entropies = np.array([0.5, 1.2, 2.8, 3.5, 2.1, 0.8])
result = compute_ces(entropies, artifact)

print(result.ces_score)      # 0.954321 (ranking score, NOT probability)
print(result.risk_level)      # "CRITICAL"
print(result.mean_entropy)    # 1.816667
print(result.max_entropy)     # 3.5
print(result.warnings)        # ["short_text: only 6 tokens..."]
```

## Middleware / Integration

Gate LLM outputs before your agent or RAG system acts on them. Production
integrations must pass real entropy/logprob data from the model provider; plain
text alone is not enough for CES.

```python
import numpy as np

from hallucination_sentinel.calibration import load_calibration
from hallucination_sentinel.integrations import (
    guard_output_from_entropies,
    TaskCriticality,
)

calibration = load_calibration("calibration.json")

# Compute these from your provider's top-k logprobs or full logits.
entropies = np.array([0.42, 0.55, 1.80, 2.10, 0.67])

decision = guard_output_from_entropies(
    prompt="What is the capital of France?",
    output="The capital of France is Berlin.",
    entropies=entropies,
    calibration=calibration,
    provider="openai",
    policy=TaskCriticality.HIGH,
)

print(decision.action)        # PolicyAction.REQUIRE_EVIDENCE
print(decision.risk_level)    # RiskLevel.HIGH
print(decision.ces_score)     # 0.87 (NOT a probability)
print(decision.diagnostic_peaks)  # entropy regions that drove the score up
```

See `examples/` for:
- `batch_qa_monitor.py` -- batch QA monitoring with summary report
- `rag_answer_gate.py` -- RAG pipeline with hallucination gate
- `agent_tool_preflight.py` -- agent tool-call preflight check

## MCP Server

Hallucination Sentinel can run as an MCP server so agent workspaces can call
CES scoring tools directly. The MCP server is an integration layer, not a
text-only hallucination detector: it requires pre-computed entropy values,
top-k logprobs, or a provider that exposes logprobs.

```bash
pip install "hallucination-sentinel[mcp] @ git+https://github.com/Heman10x-NGU/hallucination-sentinel.git"

sentinel-mcp --calibration /absolute/path/to/calibration.json
```

Claude Desktop example:

```json
{
  "mcpServers": {
    "hallucination-sentinel": {
      "command": "sentinel-mcp",
      "args": ["--calibration", "/absolute/path/to/calibration.json"]
    }
  }
}
```

MCP tools:
- `score_entropy_sequence` -- score pre-computed token entropies
- `score_topk_logprobs` -- convert top-k logprobs to entropy and score them
- `score_provider_output` -- score a prompt/output pair through a logprob-capable provider
- `smoke_provider` -- verify provider logprob support
- `inspect_calibration` -- inspect calibration metadata

See [docs/mcp.md](docs/mcp.md) for setup, example prompts, and limitations.

## Limitations

**Read this section before deploying.** Full details: [docs/limitations.md](docs/limitations.md)

| Limitation | Impact |
|-----------|--------|
| **CES is NOT a probability** | `ces_score = 0.87` means "more extreme than 87% of reference", not "87% chance of hallucination" |
| **Logprobs required** | Cannot work with providers that do not expose token probabilities |
| **Short text unreliable** | Below ~10 tokens, CES has insufficient statistical signal |
| **Confident errors evade detection** | If the model is certain about a wrong answer, entropy is low and CES is low |
| **AUROC ~0.65** | Matches state-of-the-art single-pass methods but is not perfect |
| **Domain shift** | Calibration is tied to model + provider + task + decoding config |
| **Top-k is approximate** | Entropy from top-k logprobs is a lower bound on true entropy |
| **Not a fact-checker** | Measures model uncertainty, not factual correctness |
| **Diagnostic peaks are not hallucinated spans** | High-entropy regions are flagged as "local_entropy_peak", never as "hallucinated" |

## Calibration

Calibration builds a reference entropy distribution for your specific model,
task, and decoding configuration. See [docs/calibration.md](docs/calibration.md).

```bash
# Unsupervised (no labels needed)
sentinel calibrate --input data.jsonl --output cal.json --mode unsupervised

# Supervised (only faithful sequences used for reference)
sentinel calibrate --input data.jsonl --output cal.json --mode supervised
```

## Provider Compatibility

Hallucination Sentinel works with any provider that exposes token logprobs
through an OpenAI-compatible API. See [docs/provider_logprobs.md](docs/provider_logprobs.md).

**Tested providers:** OpenAI, Together AI, Fireworks AI, DeepSeek, vLLM

**Experimental:** Ollama (limited, evolving)

**No logprob support:** Groq (schema accepts logprobs but no models return them)

## Evaluation

Run CES against labeled data with baseline comparisons:

```bash
sentinel eval --input eval.jsonl --calibration calibration.json --output report.json
```

The eval report includes:
- AUROC / AUPRC with bootstrap confidence intervals
- Confusion matrices at configured thresholds
- Baseline comparisons (perplexity, mean entropy, generation length)
- Per-length-bucket calibration coverage
- Lag-1 autocorrelation diagnostics

## Architecture

```
LLM output (text + logprobs)
    |
    v
Entropy computation (per-token Shannon entropy)
    |
    v
CES scoring (geometric mean of F0(mean) and F0(max))
    |
    v
Threshold assignment (quantile or supervised)
    |
    v
Routing decision (allow / warn / require_evidence / human_review / block)
```

## License

MIT

## Citation

```bibtex
@article{villani2026entropy,
  title={Entropy Distribution as a Fingerprint for Hallucinations in Generative Models},
  author={Villani, Mattia J. and Deshpande, Pranav and Seshadri, Akshay and Yalovetzky, Romina and Kumar, Niraj},
  journal={arXiv preprint arXiv:2605.28264},
  year={2026}
}
```

## Acknowledgments

This project is an independent implementation of the Calibrated Entropy Score
method described by Villani et al. in the paper cited above.

## Contact

For bugs or feature requests, please open a GitHub issue.

For research, product, or collaboration inquiries, reach out on
[LinkedIn](https://www.linkedin.com/in/heman10x/).
