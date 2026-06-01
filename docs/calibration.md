# Calibration

Calibration is how Hallucination Sentinel learns what "normal" entropy looks like
for your model, task, and decoding configuration. Without a calibration artifact,
the CES score is uncalibrated and the risk levels are meaningless.

## Why Calibration Matters

The CES score measures how extreme a generation's entropy profile is **relative to
a reference distribution**. The reference distribution is the calibration artifact.

If you calibrate on short-form QA with temperature 0, then use that artifact to
score long-form creative writing with temperature 1.0, the scores will be wrong.
The entropy profiles are fundamentally different.

## Two Calibration Modes

### Unsupervised (default, no labels required)

Build the reference ECDF from all available entropy sequences, regardless of
whether they are faithful or hallucinated.

```bash
sentinel calibrate \
  --input calibration_data.jsonl \
  --output calibration.json \
  --mode unsupervised
```

Each JSONL line must have an `entropy` field containing a list of per-token
entropy values (nats):

```json
{"entropy": [0.12, 0.34, 0.56, 0.23]}
{"entropy": [0.89, 1.23, 2.45, 0.67, 0.45]}
```

Unsupervised calibration works well in practice. The paper reports that unsupervised
CES performs almost identically to supervised CES because moderate contamination
(hallucinated sequences in the reference pool) barely changes the ECDF ranking.

**Use unsupervised calibration when:** you do not have labeled data, or you want
a quick starting point.

### Supervised (requires labels)

Build the reference ECDF only from sequences labeled as faithful. Hallucinated
sequences are excluded from the reference pool but can be used for threshold
tuning.

```bash
sentinel calibrate \
  --input calibration_data.jsonl \
  --output calibration.json \
  --mode supervised
```

Each JSONL line must have `entropy` and `label` fields:

```json
{"entropy": [0.12, 0.34, 0.56, 0.23], "label": true}
{"entropy": [0.89, 1.23, 2.45, 0.67, 0.45], "label": false}
```

`label: true` means faithful (included in reference pool). `label: false` means
hallucinated (excluded from reference pool).

**Use supervised calibration when:** you have labeled data and want a cleaner
reference distribution.

## Calibration Artifact

The calibration artifact is a JSON file containing:

| Field | Description |
|-------|-------------|
| `schema_version` | Artifact format version (currently `"0.1"`) |
| `created_at` | ISO timestamp of creation |
| `model` | Model name used to generate calibration data |
| `provider` | Provider name |
| `task_family` | Task type (e.g. `"short_qa"`, `"math"`) |
| `decoding` | Decoding parameters (`temperature`, `top_p`, etc.) |
| `entropy_mode` | `"full"`, `"top_k"`, or `"top_k_with_residual"` |
| `entropy_base` | `"e"` (nats) or `"2"` (bits) |
| `top_logprobs` | Number of top-k logprobs used (0 for full logits) |
| `calibration_mode` | `"unsupervised"` or `"supervised"` |
| `token_count` | Total entropy tokens in reference pool |
| `sequence_count` | Number of sequences in reference pool |
| `faithful_sequence_count` | Number of faithful sequences (supervised only) |
| `length_summary` | Min/p50/p90/max of per-sequence token counts |
| `dkw` | DKW confidence band (`confidence`, `epsilon_bound`) |
| `ecdf_values` | Sorted array of all reference entropy values |
| `thresholds` | Risk level thresholds (`low`, `medium`, `high`) |
| `known_limitations` | List of limitation tags |

Inspect an artifact:

```bash
sentinel inspect-calibration --calibration calibration.json
```

## Threshold Policies

### Reference Quantile (default)

Divides CES scores into risk bands using percentiles of the reference ECDF:

- **LOW**: CES < p75
- **MEDIUM**: p75 <= CES < p90
- **HIGH**: p90 <= CES < p97
- **CRITICAL**: CES >= p97

This works without labels.

### Max TPR at FPR (supervised)

Finds the threshold that maximizes true-positive rate subject to a maximum
false-positive-rate constraint, using Youden's J statistic. Requires labeled
evaluation data with CES scores for both faithful and hallucinated examples.

## Domain Shift

The calibration artifact is tied to a specific:

- Model (e.g. `gpt-4o-mini`)
- Provider (e.g. `openai`)
- Task family (e.g. `short_qa`)
- Decoding configuration (e.g. `temperature=0`)
- Entropy mode (e.g. `top_k_with_residual`)
- Entropy base (e.g. `e` for nats)

If you change any of these, the reference ECDF may no longer be representative.
Signs of domain shift:

- Most generations score LOW or CRITICAL with nothing in between
- The `dkw.epsilon_bound` is large (> 0.1)
- Risk levels changed dramatically after switching models

**When you detect domain shift, recalibrate.**

## DKW Bound

The Dvoretzky-Kiefer-Wolfowitz (DKW) inequality gives a confidence bound on how
close the empirical CDF is to the true CDF:

```
P(sup|F_n(x) - F(x)| > eps) <= 2 * exp(-2 * n * eps^2)
```

For confidence = 0.95:

```
eps = sqrt(ln(2 / 0.05) / (2 * n))
```

With 500 samples, eps ~ 0.061. With 10,000 samples, eps ~ 0.014.

More calibration data means a tighter bound and more reliable CES scores.

## Building Calibration Data

To build a calibration dataset:

1. Generate outputs from your model on representative prompts
2. Extract per-token entropy sequences (using your provider's logprobs)
3. Save as JSONL with `entropy` fields
4. Optionally label each sequence as faithful or hallucinated

Example using the Python API:

```python
from hallucination_sentinel.providers import OpenAICompatibleProvider
from hallucination_sentinel.entropy import entropy_from_topk_logprobs
import json

provider = OpenAICompatibleProvider.from_preset("openai", model="gpt-4o-mini")

records = []
for prompt in your_prompts:
    result = provider.score_generation(prompt, max_tokens=100)
    if result.has_top_k():
        # Compute entropy from top-k logprobs
        entropies = []
        for t in result.token_logprobs:
            probs = np.array(list(t.top_logprobs.values()))
            probs = np.exp(probs)
            probs = np.clip(probs, 1e-10, None)
            probs = probs / probs.sum()
            entropies.append(-float(np.sum(probs * np.log(probs))))
        records.append({"entropy": entropies})

with open("calibration_data.jsonl", "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")
```

Then calibrate:

```bash
sentinel calibrate \
  --input calibration_data.jsonl \
  --output calibration.json \
  --model gpt-4o-mini \
  --provider openai
```
