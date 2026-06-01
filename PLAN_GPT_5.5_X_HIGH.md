# Hallucination Sentinel - Plan GPT 5.5 X High

## One-Line Verdict

Build this, but build it as a calibrated hallucination-risk monitor, not as a truth oracle.

The original plan is directionally right, but it needs a stricter implementation path. CES is useful because it is cheap, single-pass, and logprob/logit based. It does not prove factual correctness. The product value is in fast triage, reliable calibration, honest limits, and clean integration into LLM/RAG/agent workflows.

## Source Grounding

Paper: "Entropy Distribution as a Fingerprint for Hallucinations in Generative Models" by Villani, Deshpande, Seshadri, Yalovetzky, and Kumar, arXiv:2605.28264v1, May 27, 2026.

Core paper claims to preserve:

- CES uses token-level entropy distribution shape, especially mean and tail behavior, not only average uncertainty.
- CES requires a single forward pass and black-box access to token logits or logprobs.
- CES combines calibrated mean entropy and calibrated maximum entropy through a reference CDF.
- CES is a statistical hallucination-risk test, not a factuality proof.
- The reported performance is useful but modest enough that product messaging must be conservative.
- Short generations, top-k logprob APIs, non-QA tasks, long-form text, stochastic decoding, and poor calibration data are major risk zones.

## Full Source Review Addendum

The full LaTeX source adds several important implementation constraints that were not obvious from the summary:

- Entropy is Shannon entropy with natural logarithm. Store `entropy_base = "e"` by default and do not silently mix nats and bits.
- Local model experiments use full logits via Hugging Face `output_scores=True`; API experiments use token logprobs with `top_logprobs=20`.
- The paper evaluates open-ended QA and math tasks with multi-token answers. Multiple-choice, one-word, and extremely short answers are outside the strongest evidence zone.
- The paper's reported AUROCs are in-sample ranking results because each 500-sample experiment is used both to build the reference ECDF and compute AUROC. The product must use held-out evaluation before making public claims.
- CES unsupervised is not just a demo mode. In the paper it performs almost identically to supervised CES because moderate reference contamination barely changes the ECDF ranking. Treat unsupervised calibration as the default developer path, with supervised calibration as the higher-trust path.
- CES is best understood as the strongest cheap single-pass detector under minimal access. It is not clearly better than every method: KLE, embedding regression, semantic entropy, and P(True) can beat it on some settings, but they cost more calls or require external models.
- CES is a ranking/routing score, not a calibrated probability of hallucination. If users need probabilities, add post-hoc calibration later.
- The paper does not validate token-span hallucination localization as a core result. High-entropy token peaks can be shown as diagnostics, but must not be labeled as hallucinated spans.
- If `d_KS(F0, F1) = 0`, no entropy-distribution test has power. Confident falsehoods, wrong training data, and logical reasoning errors can evade CES.

## What Must Change From The Existing Plan

1. Replace "works with any model" with "works with models/providers that expose token logits or enough token logprobs."
2. Add a Day 0 feasibility gate for MIMO/OpenAI-compatible logprobs. If the API does not return logprobs, MIMO can still execute the code plan, but it cannot be the first detection backend.
3. Treat calibration as a product asset, not an implementation detail.
4. Add a held-out evaluation harness before public claims.
5. Avoid saying "this detects hallucinations" without qualifiers. Say "this flags outputs whose entropy profile looks inconsistent with calibrated faithful outputs."
6. Remove premature hosted SaaS work from the first milestone. Ship library, CLI, eval harness, and agent/RAG middleware first.
7. Add explicit limitations, failure modes, and threshold policy.
8. Replace "flagged hallucinated tokens" with "entropy diagnostics"; the paper does not prove span-level localization.

## Product Thesis

Hallucination Sentinel is a low-latency risk gate for LLM outputs.

It should sit between model generation and downstream action:

```text
LLM/RAG/Agent output
        |
        v
Token logprobs/logits captured
        |
        v
CES risk score + token entropy diagnostics
        |
        +--> low risk: allow / continue
        +--> medium risk: attach warning / request evidence
        +--> high risk: route to verification or human review
```

The first product should not be a dashboard. It should be:

- a Python package,
- a CLI,
- an OpenAI-compatible logprob provider,
- a local Hugging Face provider,
- a calibration/evaluation harness,
- an agent/RAG middleware wrapper.

## Strategic Positioning

Do not position this as "hallucination solved."

Position it as:

"A single-pass uncertainty firewall for LLM outputs. It cheaply flags suspicious generations before agents, RAG systems, or production workflows act on them."

Best early users:

- AI engineers building RAG systems.
- Agent builders who need pre-action risk gates.
- Internal platform teams evaluating LLM outputs at scale.
- Researchers implementing lightweight uncertainty baselines.

Bad early users:

- Non-technical users expecting perfect truth detection.
- Enterprise buyers needing SOC2, dashboard, audit workflows, and legal-grade factuality on day one.
- Teams whose model provider exposes no logprobs and who cannot switch models.

## Core Algorithm Spec

For each generated token position `t`, obtain the model's next-token probability distribution `p_t`.

Token entropy:

```text
h_t = -sum_v p_t(v) * log(p_t(v))
```

Use natural logarithms by default. If users request bits, make the base explicit and persist it in calibration artifacts, because thresholds and ECDFs are not compatible across entropy bases.

For a generation with `m` tokens:

```text
mean_entropy = mean(h_1 ... h_m)
max_entropy = max(h_1 ... h_m)
```

Build a reference empirical CDF `F0` from token entropies of calibrated faithful/non-hallucinated outputs.

CES score:

```text
CES(h) = sqrt(F0(mean_entropy) * F0(max_entropy))
```

Interpretation:

- Higher score means the generation's entropy profile is more extreme relative to the faithful reference distribution.
- Score is comparable only under a compatible calibration context: same or similar model, task family, decoding mode, and logprob extraction method.
- For top-k API logprobs, entropy is approximate. Track and expose this approximation.

## Top-k Logprob Approximation Policy

Many APIs expose only top-k token logprobs. Full-vocabulary entropy is then unavailable.

Implement this explicitly:

1. Convert top-k logprobs to probabilities.
2. Compute `observed_mass = sum(top_k_probs)`.
3. Add an optional residual bucket with probability `max(0, 1 - observed_mass)`.
4. Compute entropy over `top_k + residual_bucket`.
5. Return metadata:
   - `entropy_mode = "full" | "top_k" | "top_k_with_residual"`
   - `observed_mass`
   - `top_k`
   - `approximation_warning`

Do not hide this. It is one of the main differences between a toy implementation and a trustworthy one.

Provider target for the first API backend: support `top_logprobs=20`, because this is the paper's API setting. Lower values should work mechanically, but mark them as lower-confidence entropy approximations.

## Calibration Strategy

Calibration is the moat of the first real version.

Support two modes:

### Mode A - Supervised Calibration

Input JSONL:

```json
{"prompt":"...","output":"...","label":"faithful","metadata":{"task":"qa","model":"..."}}
{"prompt":"...","output":"...","label":"hallucinated","metadata":{"task":"qa","model":"..."}}
```

Use only `label = faithful` rows to build `F0`.

Use hallucinated rows only for threshold selection and evaluation.

### Mode B - Unsupervised Calibration

Input JSONL without labels:

```json
{"prompt":"...","output":"...","metadata":{"task":"qa","model":"..."}}
```

Build `F0` from pooled entropy sequences, but mark the calibration as contamination-prone.

Unsupervised calibration should be the default zero-label developer path. The full source reports that unsupervised CES is essentially tied with supervised CES across the benchmark grid and is robust even under heavy contamination/noisy labels.

Still, do not make production claims from unsupervised calibration until a held-out eval report exists for the user's task family.

### Calibration Artifacts

Save calibration artifacts as versioned JSON:

```json
{
  "schema_version": "0.1",
  "created_at": "2026-06-01T00:00:00Z",
  "model": "provider/model",
  "provider": "openai_compatible",
  "task_family": "short_qa",
  "decoding": {"temperature": 0, "top_p": 1, "max_new_tokens": 128},
  "entropy_mode": "top_k_with_residual",
  "entropy_base": "e",
  "top_logprobs": 20,
  "calibration_mode": "unsupervised",
  "token_count": 12345,
  "sequence_count": 500,
  "faithful_sequence_count": 410,
  "length_summary": {"p50": 14, "p90": 96, "min": 2, "max": 256},
  "dkw": {"confidence": 0.95, "epsilon_bound": 0.061},
  "ecdf_values": [0.01, 0.02],
  "thresholds": {
    "low": 0.75,
    "medium": 0.90,
    "high": 0.97
  },
  "known_limitations": ["short_text", "top_k_entropy"]
}
```

Implementation note: storing all entropy samples is acceptable for v0.1. Later, switch to quantile sketches if artifacts become large.

## Threshold Policy

Default thresholds should be quantile based, not hand-wavy.

Initial defaults:

- `LOW`: CES < p75 of calibration/reference scores
- `MEDIUM`: p75 <= CES < p90
- `HIGH`: p90 <= CES < p97
- `CRITICAL`: CES >= p97

For supervised calibration, allow threshold tuning against held-out labels:

- maximize Youden's J for balanced detection,
- or target a configured false-positive rate,
- or target a configured recall for high-risk workflows.

Expose the threshold policy in output. Never return a naked score without calibration metadata.

The paper suggests choosing thresholds from ROC curves by maximizing true positive rate subject to a false-positive-rate bound. Implement that explicitly as `threshold_policy = "max_tpr_at_fpr"` for supervised eval sets.

Do not describe CES as a probability. `ces_score = 0.87` means "high relative rank under the chosen reference ECDF," not "87% probability of hallucination."

## Output Schema

Return structured JSON everywhere, including CLI output.

```json
{
  "schema_version": "0.1",
  "text": "...",
  "model": "provider/model",
  "provider": "openai_compatible",
  "ces_score": 0.87,
  "score_is_probability": false,
  "risk_level": "HIGH",
  "threshold_policy": "reference_quantile",
  "cdf_mean": 0.81,
  "cdf_max": 0.94,
  "mean_entropy": 1.23,
  "max_entropy": 2.34,
  "token_count": 42,
  "entropy_mode": "top_k_with_residual",
  "entropy_base": "e",
  "top_logprobs": 20,
  "observed_mass_mean": 0.94,
  "short_text_warning": false,
  "calibration_id": "sha256:...",
  "diagnostic_peaks": [
    {
      "start_token": 17,
      "end_token": 20,
      "text": "...",
      "entropy": 2.34,
      "reason": "local_entropy_peak"
    }
  ],
  "warnings": []
}
```

Risk levels must be treated as routing advice, not truth labels.

`diagnostic_peaks` are local entropy peaks for debugging and UX. They are not verified hallucinated spans.

## V0.1 Architecture

Recommended structure:

```text
hallucination-sentinel/
  pyproject.toml
  README.md
  PLAN_GPT_5.5_X_HIGH.md
  src/
    hallucination_sentinel/
      __init__.py
      ces.py
      entropy.py
      calibration.py
      thresholds.py
      schemas.py
      cli.py
      eval.py
      providers/
        __init__.py
        base.py
        openai_compatible.py
        huggingface.py
      integrations/
        __init__.py
        middleware.py
  tests/
    test_entropy.py
    test_calibration.py
    test_ces.py
    test_thresholds.py
    test_cli.py
  examples/
    calibration_toy.jsonl
    score_sample.json
  docs/
    limitations.md
    calibration.md
    provider_logprobs.md
```

## Provider Interface

Create a provider protocol so the core CES code is independent from model APIs.

```python
class TokenLogprobProvider(Protocol):
    def score_generation(
        self,
        prompt: str,
        output: str,
        *,
        model: str,
        temperature: float | None = None,
        top_logprobs: int | None = 20,
    ) -> TokenLogprobResult:
        ...
```

Important: Some APIs only return logprobs for newly generated tokens, not arbitrary existing text. If scoring arbitrary text is unavailable, the provider must either:

- regenerate deterministically from the prompt and score that generated output,
- or use a local/HF model that supports teacher-forced scoring,
- or fail with a clear error.

This distinction must be explicit in docs and CLI commands.

Provider results must persist:

- selected tokens,
- selected-token logprobs,
- top-k alternatives per token when available,
- raw provider response shape/version where safe,
- decoding parameters,
- whether entropy is from full logits, top-k logprobs, or selected-token logprobs only.

Selected-token logprob alone is not enough for CES. It can support perplexity, but not token entropy.

## CLI Commands

Use `typer` or `click`. Prefer `typer` for typed command signatures.

Required commands:

```bash
sentinel smoke-provider --provider openai-compatible --model "$MODEL"
sentinel calibrate --input examples/calibration_toy.jsonl --output calibration.json
sentinel inspect-calibration --calibration calibration.json
sentinel score --entropy-json entropy_sequence.json --calibration calibration.json
sentinel score-provider --prompt prompt.txt --output output.txt --provider openai-compatible --model "$MODEL" --calibration calibration.json
sentinel eval --input eval.jsonl --calibration calibration.json --output eval_report.json
sentinel replicate-paper-lite --provider openai-compatible --model "$MODEL" --dataset triviaqa --n 100 --output report.json
sentinel serve --calibration calibration.json --port 8787
```

V0.1 can skip `serve` if time gets tight. Do not skip `calibrate`, offline `score`, provider smoke, or `eval`.

## MIMO V2.5 Pro Execution Setup

Use MIMO V2.5 Pro as the coding model/implementation worker. Do not assume it is automatically a compatible detection backend.

Before implementation, run this feasibility test against the MIMO API:

1. Can it return token logprobs for generated tokens?
2. Can it return top-k logprobs, not just selected-token logprob?
3. Can it score a provided output against a prompt, or only return logprobs for generated text?
4. Are logprobs available through an OpenAI-compatible `/chat/completions` or `/responses` endpoint?
5. Can the requested decoding settings be fixed and replayed for calibration? Local greedy is ideal; API temperature may be provider-specific, but it must be recorded.

If the answer is no:

- Use MIMO to write code.
- Use OpenAI-compatible providers that expose logprobs, or Hugging Face/local models, as detection backends.
- Keep MIMO backend support as "pending logprob availability."

Suggested environment variables:

```bash
export MIMO_API_KEY="..."
export MIMO_BASE_URL="..."
export MIMO_MODEL="mimo-v2.5-pro"
```

Do not hardcode keys. Do not commit provider secrets.

## Execution Plan

### Phase 0 - Feasibility And Paper-Fidelity Check

Goal: prevent building a nice-looking wrapper around the wrong assumption.

Tasks:

- Confirm which provider exposes usable token logprobs.
- Implement a tiny provider smoke script.
- Verify whether arbitrary text scoring is possible.
- Decide first supported backend:
  - Option A: OpenAI-compatible API with logprobs.
  - Option B: Hugging Face local model.
  - Option C: both, if simple.
- Record limitations in `docs/provider_logprobs.md`.

Acceptance criteria:

- A command proves whether provider logprobs are available.
- It separately reports selected-token logprobs vs top-k logprobs.
- Failure message is clear if logprobs are missing.
- Plan does not depend on a provider capability that has not been verified.

### Phase 1 - Core CES Library

Goal: deterministic, tested implementation of entropy, ECDF calibration, CES, and thresholds.

Tasks:

- Create package structure.
- Implement entropy from full probabilities.
- Implement entropy from top-k logprobs with residual bucket.
- Implement perplexity and length-normalized entropy baselines from the same entropy/logprob data.
- Implement ECDF artifact build/load/save.
- Implement calibration diagnostics: token count, sequence count, length summary, and DKW-style epsilon bound.
- Implement CES scoring.
- Implement optional one-sample/two-sample KS diagnostic score.
- Implement threshold assignment.
- Implement warnings for:
  - token count < 10,
  - token count < 100 for low-power factoid QA,
  - top-k entropy approximation,
  - selected-token-logprob-only provider response,
  - entropy base mismatch,
  - calibration/model mismatch,
  - missing calibration metadata.
- Add unit tests with fixed fixtures.

Acceptance criteria:

- `pytest` passes.
- CES score is deterministic on fixture data.
- Calibration artifact round-trips from disk.
- No provider/API calls are needed for core tests.

### Phase 2 - Provider Integration

Goal: plug real model APIs into the core library without contaminating algorithm code.

Tasks:

- Implement `TokenLogprobProvider` base types.
- Implement `OpenAICompatibleProvider`.
- Implement `HuggingFaceProvider` as optional extra.
- Add smoke command.
- Add integration tests that can be skipped when API keys are absent.
- Add clear errors for providers that cannot return logprobs.
- For Hugging Face, use `generate(..., return_dict_in_generate=True, output_scores=True)` for generated-token entropy and teacher-forced forward pass for arbitrary-output scoring.

Acceptance criteria:

- Core tests run offline.
- Provider smoke can verify logprob support.
- If MIMO lacks logprobs, the system says so cleanly and suggests local/HF or another API backend.

### Phase 3 - CLI And JSON Workflows

Goal: make the package useful from terminal before building product UI.

Tasks:

- Implement `sentinel calibrate`.
- Implement `sentinel inspect-calibration`.
- Implement offline `sentinel score` from a saved entropy sequence.
- Implement `sentinel score-provider` only after a real provider has passed smoke checks.
- Implement `sentinel eval`.
- Add JSON and rich terminal output modes.
- Add examples for:
  - toy calibration,
  - scoring a saved output,
  - scoring through a real provider when credentials exist,
  - evaluating labeled JSONL.

Acceptance criteria:

- A user can calibrate from JSONL.
- A user can score one saved entropy sequence without network access.
- A user can score one provider-backed output only when provider logprobs are verified.
- A user can run held-out eval and get AUROC/AUPRC if labels exist.
- CLI exits non-zero on invalid calibration/provider mismatch.

### Phase 4 - Evaluation Harness

Goal: make public claims defensible.

Tasks:

- Build `eval.jsonl` loader.
- Compute:
  - AUROC,
  - AUPRC,
  - AUARC / accuracy-rejection curve if labels include correctness,
  - confusion matrix at thresholds,
  - false positive rate,
  - false negative rate,
  - bootstrap confidence intervals,
  - lag-1 autocorrelation and effective sample size diagnostics,
  - calibration coverage by token length bucket.
- Add baseline metrics:
  - perplexity,
  - generation length,
  - mean entropy only,
  - max entropy only,
  - length-normalized entropy,
  - KS score if implemented,
  - raw geometric mean as an ablation.
- Split calibration and evaluation sets.
- Add per-length-bin reporting: `<5`, `5-9`, `10-49`, `50-99`, `>=100` generated tokens.
- Add contamination/noisy-label simulation for calibration robustness.
- Add report output as JSON and Markdown.

Acceptance criteria:

- No in-sample score claims.
- Report compares CES against at least LN-Entropy, Perplexity, Generation Length, mean-only, and max-only baselines.
- Report highlights short-answer degradation.
- Report flags high entropy-autocorrelation cohorts where the paper's i.i.d. assumption is weaker.
- Report states whether CES beats the cheap baselines enough to justify use on that task.

### Phase 5 - Agent/RAG Middleware

Goal: turn the library into a product wedge.

Tasks:

- Implement middleware function:

```python
def guard_output(prompt, output, *, calibration, provider, policy):
    ...
```

- Add policy actions:
  - `allow`,
  - `warn`,
  - `require_evidence`,
  - `human_review`,
  - `block`.
- Route by risk plus task criticality, not score alone.
- Add LangChain/LangGraph-style wrapper only if it stays small.
- Add examples:
  - RAG answer risk gate,
  - agent tool-call preflight,
  - batch QA monitor.

Acceptance criteria:

- Middleware returns structured routing decision.
- Example shows high-risk output being routed to review.
- No dashboard required.

### Phase 6 - Public Release

Goal: ship an honest OSS artifact.

Tasks:

- Rewrite README to remove unsupported claims.
- Add limitations docs.
- Add calibration docs.
- Add provider compatibility docs.
- Add quickstart.
- Publish GitHub repo.
- Optional: publish to PyPI after at least one real provider works.

Acceptance criteria:

- README says what works today, not what should work later.
- Limitations are impossible to miss.
- Demo can run without paid API.
- Real provider path works when credentials are supplied.

### Phase 7 - Paper-Lite Replication

Goal: prove the implementation follows the paper closely enough before using it as a product wedge.

Tasks:

- Implement a small open-ended QA benchmark adapter for at least one public dataset.
- Use 5-shot prompting where feasible.
- Generate up to 128 tokens for QA and 256 for math-style tasks.
- Store prompt, output, token data, entropy sequence, model, provider, decoding, and labels.
- Support labels from:
  - exact/overlap metrics when gold answers exist,
  - optional LLM judge,
  - user-provided labels.
- Compare CES, unsupervised CES, LN-Entropy, Perplexity, Generation Length, and KS.

Acceptance criteria:

- `sentinel replicate-paper-lite` can produce a small report on 100 samples.
- The report clearly marks itself as "paper-lite" and not a reproduction of the 80-cell benchmark.
- The report states whether the observed effect size is strong enough for this task.

## MIMO Implementation Prompts

Use these as sequential prompts. Do not give MIMO the whole project and ask it to improvise.

### Prompt 1 - Scaffold And Tests

```text
You are implementing Hallucination Sentinel from PLAN_GPT_5.5_X_HIGH.md.

Task: create the Python package scaffold only.

Requirements:
- Use src/hallucination_sentinel layout.
- Add pyproject.toml.
- Add pytest configuration.
- Add modules: entropy.py, calibration.py, ces.py, thresholds.py, schemas.py, cli.py.
- Add providers/base.py and providers/openai_compatible.py stubs.
- Add tests with placeholder imports.
- Do not implement API calls yet.
- Do not add dashboard or cloud code.

Return a concise summary and commands to run tests.
```

### Prompt 2 - Entropy Core

```text
Implement entropy.py and its tests.

Requirements:
- Function for entropy from probability list.
- Function for entropy from logprob list.
- Function for approximate entropy from top-k logprobs with optional residual bucket.
- Use natural log by default and include `entropy_base` in metadata.
- Return metadata: observed_mass, entropy_mode, top_k/top_logprobs, entropy_base.
- Validate probabilities and handle numerical stability.
- Add selected-token negative log-likelihood helpers for perplexity baseline.
- Add deterministic pytest cases.
```

### Prompt 3 - Calibration And CES

```text
Implement calibration.py, ces.py, thresholds.py, and tests.

Requirements:
- Build ECDF from faithful entropy samples.
- Build ECDF from all entropy samples for unsupervised mode.
- Save/load calibration artifact JSON.
- Store length summary and DKW-style epsilon bound in calibration artifact.
- Implement CES = sqrt(F0(mean_entropy) * F0(max_entropy)).
- Return cdf_mean and cdf_max, not just the final score.
- Implement quantile threshold policy and max-TPR-at-FPR threshold policy.
- Add mismatch warnings for model/task/entropy mode/entropy base/top_logprobs/decoding.
- Add tests for ECDF behavior, CES determinism, artifact round-trip, and threshold assignment.
```

### Prompt 4 - CLI Offline Workflow

```text
Implement CLI commands for offline calibration and scoring.

Required commands:
- sentinel calibrate --input calibration.jsonl --output calibration.json
- sentinel inspect-calibration --calibration calibration.json
- sentinel score --entropy-json entropy_sequence.json --calibration calibration.json
- sentinel eval --input eval.jsonl --calibration calibration.json --output eval_report.json

Do not call external APIs in this prompt.
Use JSON outputs.
Add tests with Typer CliRunner or Click CliRunner.
```

### Prompt 5 - OpenAI-Compatible Provider

```text
Implement providers/openai_compatible.py.

Requirements:
- Read api_key, base_url, and model from arguments/env vars.
- Add smoke_provider function that checks whether selected-token logprobs and top-k logprobs are returned separately.
- Default to requesting top_logprobs=20 when supported.
- Do not hardcode MIMO/OpenAI keys.
- If only selected-token logprobs are available, allow perplexity baseline but raise ProviderCapabilityError for CES entropy.
- If logprobs are unavailable, raise a clear ProviderCapabilityError.
- Keep provider integration tests skipped unless env vars are present.
```

### Prompt 6 - MIMO Feasibility

```text
Add a command:

sentinel smoke-provider --provider openai-compatible --base-url "$MIMO_BASE_URL" --model "$MIMO_MODEL"

It should print:
- provider reachable,
- logprobs supported,
- top-k logprobs supported,
- selected-token-only vs entropy-capable,
- max top_logprobs accepted if detectable,
- generated-token scoring supported,
- arbitrary-output scoring supported if detectable,
- decoding parameters used,
- recommended backend status.

If MIMO V2.5 Pro does not expose logprobs, mark it unsupported as a detection backend but still usable as a coding API.
```

### Prompt 7 - Evaluation Report

```text
Implement eval report generation.

Requirements:
- Accept labeled JSONL with entropy sequences and labels.
- Split calibration/eval if requested.
- Compute AUROC, AUPRC, AUARC, and bootstrap CI when sklearn/numpy support exists.
- Always compute confusion matrix at configured thresholds.
- Compare CES against LN-Entropy, Perplexity, Generation Length, mean_entropy, max_entropy, KS score if available, and raw geom(mean,max).
- Report per-length buckets and short-output warnings.
- Add optional contamination/noisy-label simulation.
- Output JSON and Markdown.
- Add tests for metric computation on toy data.
```

### Prompt 8 - Product Middleware

```text
Implement integrations/middleware.py.

Requirements:
- guard_output returns a routing decision: allow, warn, require_evidence, human_review, block.
- Decision policy should be configurable by risk level and workflow criticality.
- Include warnings and calibration metadata in output.
- Expose diagnostic_peaks but do not call them hallucinated spans.
- Add examples for RAG answer gating and agent tool-call preflight.
```

### Prompt 9 - README Rewrite

```text
Rewrite README.md based on actual implemented features.

Rules:
- No unsupported backend status.
- No "works with any model" claim.
- Clearly say logprobs/logits are required.
- Clearly say this is a risk signal, not proof of truth.
- Clearly say CES score is not a hallucination probability.
- Clearly say short generations and confident falsehoods can evade the detector.
- Include quickstart for offline demo and real provider smoke test.
- Include limitations section.
```

### Prompt 10 - Paper-Lite Replication

```text
Implement a small paper-lite replication command.

Command:
- sentinel replicate-paper-lite --dataset triviaqa --n 100 --provider openai-compatible --model "$MODEL" --output report.json

Requirements:
- Use open-ended QA format.
- Use 5-shot prompting where feasible.
- Generate up to 128 tokens for QA.
- Request top_logprobs=20 for API providers when supported.
- Store prompt, output, labels, entropy sequences, provider metadata, decoding metadata, and method scores.
- Compare CES unsupervised, CES supervised if labels exist, LN-Entropy, Perplexity, Generation Length, and KS if implemented.
- Mark output clearly as paper-lite, not full reproduction.
```

## Hard Acceptance Tests Before Calling V0.1 Done

Do not call the project done until all of these pass:

- `pytest` passes offline.
- `sentinel calibrate` works on toy JSONL.
- `sentinel score` works on a toy entropy sequence and calibration artifact.
- `sentinel eval` produces a report on toy labeled data.
- Provider smoke test gives a truthful pass/fail for logprob support.
- README does not claim unsupported providers.
- Calibration artifact includes model/provider/task/entropy/decoding/top_logprobs metadata.
- Calibration artifact includes length summary and DKW-style epsilon bound.
- Short outputs trigger warnings.
- Top-k entropy returns approximation metadata.
- Selected-token-only logprob providers cannot run CES and say why.
- Eval report compares against LN-Entropy, Perplexity, and Generation Length.
- Eval report includes length-bucket and autocorrelation diagnostics.
- CES score is never described as a probability.
- Diagnostic token peaks are not described as hallucinated spans.

## Product Roadmap After V0.1

### V0.2 - Real Benchmarks

- Add small QA benchmark adapter.
- Add held-out eval reports.
- Compare against LN-Entropy, Perplexity, Generation Length, mean entropy, max entropy, and KS.
- Add paper-lite replication report.
- Publish honest benchmark numbers.

### V0.3 - Integration Layer

- Add LangGraph/LangChain middleware.
- Add OpenTelemetry-compatible logging.
- Add batch scoring for production logs.

### V0.4 - Hosted API Only If Pull Exists

Build hosted API only after:

- at least 20 external users/stars/issues, or
- 3 teams ask for hosted scoring, or
- one credible design partner wants integration.

Hosted API before that is distraction.

## Startup/Product Reality Check

This is a strong portfolio and credibility project. It is not automatically a VC-scale company.

The possible path to VC-scale is not "CES library." It is:

1. Start with lightweight hallucination-risk monitoring.
2. Expand into agent/RAG reliability observability.
3. Own calibration datasets and eval reports per task/domain/model.
4. Become the trust/routing layer before AI systems take actions.

The wedge is developer adoption. The moat, if any, comes from:

- calibration artifacts,
- provider compatibility,
- eval datasets,
- workflow integrations,
- production feedback loops.

## Messaging

Use:

> "Hallucination Sentinel is a single-pass uncertainty gate for LLM outputs. It uses calibrated token entropy profiles to flag generations that look risky before your agent or RAG system acts on them."

Avoid:

> "Detect hallucinations with 87% accuracy."

Avoid:

> "Works with any model."

Avoid:

> "Truth layer for AI."

Better:

> "Fast risk signal. Not a truth oracle."

## Final Build Order

1. Scaffold package and tests.
2. Implement entropy utilities.
3. Implement calibration and CES.
4. Implement offline CLI.
5. Implement provider smoke test.
6. Implement real provider only after logprob capability is verified.
7. Implement eval harness.
8. Implement paper-lite replication.
9. Implement middleware.
10. Rewrite README honestly.
11. Ship OSS v0.1.

If time is limited, cut hosted API, dashboard, Hugging Face support, and LangChain wrappers before cutting calibration, eval, or provider smoke tests.
