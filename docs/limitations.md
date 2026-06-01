# Limitations

**Read this before using Hallucination Sentinel in production.**

## The CES score is NOT a hallucination probability

The Calibrated Entropy Score (CES) is a **ranking signal**, not a probability.
A CES of 0.85 does not mean "85% chance this is a hallucination."
It means this generation's entropy profile is more extreme than 85% of the
reference calibration pool.

To convert CES to a calibrated probability you need a labeled evaluation
dataset and a post-hoc calibration step (e.g. Platt scaling or isotonic
regression). The tool does not do this automatically.

## Logprobs / logits are required

Hallucination Sentinel **cannot work** without access to token-level
probabilities from the model. It needs either:

- Full vocabulary logprobs (ideal)
- Top-k logprobs per token (common, works well)
- At minimum, the selected-token logprob (perplexity only, CES unavailable)

If your provider does not expose logprobs at all, this tool has no signal
to work with. There is no text-only fallback that produces meaningful
uncertainty estimates.

### Text-only heuristic removed from production flow

The `guard_output()` function previously included a text-only heuristic that
estimated entropy from token length, digits, and capitalization patterns. This
heuristic has been **removed from the production flow** because it does not
produce meaningful uncertainty estimates.

**Current behavior:**
- `guard_output()` now raises `ProviderCapabilityError` when called without
  real logprobs from a provider.
- `guard_output_from_entropies()` is the offline/batch path for pre-computed
  entropy sequences.
- `guard_output_with_logprobs()` is the provider path for pre-fetched logprobs.
- `guard_output_with_text_heuristic_experimental()` preserves the old heuristic
  for offline experimentation only, with a deprecation warning.

**Migration guide:**
```python
# OLD (no longer works):
decision = guard_output(prompt, output, calibration=cal)

# NEW - with pre-computed entropy:
decision = guard_output_from_entropies(prompt, output, entropies, calibration=cal)

# NEW - with provider logprobs:
decision = guard_output_with_logprobs(prompt, output, logprobs, calibration=cal)

# EXPERIMENTAL ONLY (not for production):
decision = guard_output_with_text_heuristic_experimental(prompt, output, calibration=cal)
```

See [provider_logprobs.md](./provider_logprobs.md) for a provider-by-provider
compatibility table.

## Short generations are unreliable

CES reliability degrades sharply below ~10 tokens. The entropy distribution
of a 3-token generation does not carry enough statistical signal to
distinguish hallucinated from faithful text.

The tool emits a `short_text` warning when `token_count < 10` and a
`low_power` warning when `token_count < 100`. Treat these warnings as
meaningful: do not trust the risk level for short outputs.

## Confident falsehoods can evade detection

A model that hallucinates with **high confidence** (low entropy) will produce
a low CES score. The detector works by measuring uncertainty -- if the model
is certain about a wrong answer, the entropy profile looks identical to a
correct answer.

This is an inherent limitation of single-pass entropy-based detection.
Multi-sample methods (self-consistency, sampling-based disagreement) can
catch some of these cases, but at 10-50x the cost.

## AUROC is not 1.0

On published benchmarks the CES algorithm achieves AUROC in the 0.60-0.70
range, depending on the dataset and model. This matches or exceeds other
single-pass methods, but it means:

- False positives will occur (faithful text flagged as risky)
- False negatives will occur (hallucinated text passes through)

**Do not use CES as the sole gate for safety-critical decisions.**
Use it as one layer in a defense-in-depth strategy.

## Calibration domain shift

The calibration artifact captures the entropy distribution of a specific
model, provider, task family, and decoding configuration. If you change
any of these, the reference ECDF may no longer be representative.

Signs of domain shift:
- Most generations score LOW or CRITICAL with nothing in between
- The `dkw.epsilon_bound` in the calibration artifact is large (> 0.1)
- Risk levels changed dramatically after switching models

When you detect domain shift, recalibrate.

## Top-k entropy is an approximation

When the provider only returns top-k logprobs (not full vocabulary), the
entropy is computed from the top-k tokens plus a residual bucket for the
remaining probability mass. This is a lower bound on true entropy.

The tool warns about this with `top_k_entropy` and checks for
`entropy_mode_mismatch` if the calibration was built with a different mode.

## Not a fact-checker

Hallucination Sentinel measures **model uncertainty**, not **factual
correctness**. It cannot tell you whether a specific claim is true or false.
It can only tell you whether the model was uncertain when generating it.

A generation with LOW CES can still contain errors.
A generation with HIGH CES can still be factually correct.

## Summary of what this tool does and does not do

| Claim | Status |
|-------|--------|
| Detects hallucinations | No -- detects uncertainty patterns correlated with hallucination |
| Works with any model | No -- requires models/providers that expose logprobs |
| Proves text is correct | No -- LOW CES does not mean "safe" |
| Replaces human review | No -- use as a triage signal, not a verdict |
| Works on short text | Unreliable below ~10 tokens |
| Catches confident errors | No -- high-confidence errors look like correct answers |
