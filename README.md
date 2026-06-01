# Hallucination Sentinel

**Detect LLM hallucinations using entropy analysis.**

Single forward pass. Black-box. Works with any model.

Based on ["Entropy Distribution as a Fingerprint for Hallucinations in Generative Models"](https://arxiv.org/abs/2605.28264) — the Calibrated Entropy Score (CES) algorithm.

## Why This Exists

LLMs generate confident-sounding but factually wrong text. You can't tell the difference without manual verification.

**Hallucination Sentinel** analyzes the entropy (uncertainty) of token probabilities. Hallucinated text has higher entropy because the model is "guessing" rather than "knowing."

## Quick Start

```bash
# Install
pip install hallucination-sentinel

# Check text (demo mode)
hallucination-sentinel check "The capital of France is Berlin"
# Output: ⚠️ HALLUCINATION RISK: 0.87 (HIGH)

# Check with OpenAI API
export OPENAI_API_KEY=your-key
hallucination-sentinel check --api openai "Your LLM output here"

# Check with local model
hallucination-sentinel check --model meta-llama/Llama-3-8B "Your text here"
```

## How It Works

1. **Get token probabilities** from LLM (via API or local model)
2. **Calculate entropy** per token
3. **Compute CES score** (geometric mean of mean + max entropy, calibrated)
4. **Flag high-entropy segments** (potential hallucinations)

The CES algorithm:
- Requires only a **single forward pass**
- Works with **black-box access** (just needs logprobs)
- Provides **formal statistical guarantees**
- Matches multi-sample methods at **1/10th the cost**

## Output Format

```json
{
  "text": "The capital of France is Berlin",
  "ces_score": 0.87,
  "risk_level": "HIGH",
  "flagged_tokens": [
    {
      "token": "Berlin",
      "entropy": 2.34,
      "position": 7
    }
  ],
  "mean_entropy": 1.23,
  "max_entropy": 2.34,
  "token_count": 7
}
```

## Supported Backends

| Backend | Status | Notes |
|---------|--------|-------|
| OpenAI API | ✅ | Uses logprobs |
| HuggingFace | ✅ | Local models |
| Ollama | 🔜 | Coming soon |
| Azure OpenAI | 🔜 | Coming soon |

## Use Cases

- **Quality assurance** — Check LLM outputs before publishing
- **RAG validation** — Verify retrieved context is used correctly
- **Agent trust** — Flag uncertain agent actions (Daemons integration)
- **Content moderation** — Detect AI-generated misinformation

## Daemons Integration

Hallucination Sentinel is the trust layer for [Daemons](https://github.com/Heman10x-NGU/daemons) AI agents.

```python
from hallucination_sentinel import check

# Daemon agent generates output
output = daemon_agent.generate("Write a report about Q3 sales")

# Check for hallucinations
result = check(output)

if result.risk_level == "HIGH":
    # Don't execute — flag for human review
    human_review(output, result)
else:
    # Execute action
    execute_action(output)
```

## The Algorithm

CES = f(mean entropy, max entropy)

Where:
- **Mean entropy** = average uncertainty across all tokens
- **Max entropy** = peak uncertainty (tail of distribution)
- **CES** = geometric mean, calibrated against reference CDF

The key insight: **the shape of the entropy distribution matters, not just the mean.**

Hallucinated text has:
- Higher mean entropy (model is uncertain)
- Higher max entropy (model is "guessing" on specific tokens)
- Different distribution shape (fatter tails)

## Limitations

- **AUROC ~0.65** — Not perfect, but matches state-of-the-art
- **Short texts** — Works best with >10 tokens
- **Logprobs required** — Some APIs don't expose token probabilities
- **Calibration needed** — Reference distributions required for best results

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

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

- [Kurate.org](https://kurate.org) for paper rankings
- The CES paper authors for the algorithm
- The open source community for inspiration
