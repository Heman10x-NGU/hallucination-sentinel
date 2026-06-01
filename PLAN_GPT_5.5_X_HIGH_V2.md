# Hallucination Sentinel - Plan GPT 5.5 X High V2

## Purpose

This V2 plan is a repair-and-execution plan for the current `hallucination-sentinel` repo.

The repo already has a strong v0.1 scaffold, but the real local review found several P1/P2 issues that must be fixed before this can be trusted as a product component or executed cleanly through Claude Code with the MIMO V2.5 Pro API.

Use MIMO V2.5 Pro as the coding model in Claude Code. Do not assume MIMO is already a valid hallucination-detection backend until its logprob support is verified.

## Real Test Evidence From Current Repo

Date tested: 2026-06-01.

Repo path:

```bash
/Users/heman10x/Downloads/codex_dev/hallucination-sentinel
```

Verified working:

```bash
.venv/bin/pytest -q
```

Result:

```text
277 passed, 6 skipped in 1.32s
```

The skipped tests are live provider tests requiring API keys.

Verified installed CLI:

```bash
.venv/bin/sentinel --help
```

Actual commands:

```text
calibrate
eval
inspect-calibration
score
score-provider
smoke-provider
```

Offline CLI E2E works mechanically:

```bash
sentinel calibrate --input cal.jsonl --output calibration.json
sentinel score --entropy-json score.json --calibration calibration.json
sentinel inspect-calibration --calibration calibration.json
```

But the current demo can accidentally pass because its toy entropy values are below 1.0. The default threshold bug appears when normal entropy values are above 1.0.

Threshold bug repro:

```python
import numpy as np
from hallucination_sentinel.calibration import build_calibration
from hallucination_sentinel.thresholds import assign_thresholds
from hallucination_sentinel.ces import compute_ces

seqs = [np.linspace(1.0, 3.0, 50) for _ in range(20)]
art = build_calibration(seqs)
thresholds = assign_thresholds(art)
result = compute_ces(np.array([10.0] * 50), art)

print(thresholds)
print(result.ces_score, result.cdf_mean, result.cdf_max, result.risk_level)
```

Current bad output:

```text
{'low': 2.5102, 'medium': 2.8000, 'high': 2.9591}
1.0 1.0 1.0 LOW
```

This is wrong. A CES score of `1.0` should never be `LOW`.

MIMO provider probe:

```bash
.venv/bin/sentinel smoke-provider --provider mimo
```

Current output:

```text
Error: Unknown provider 'mimo'. Available: openai, together, fireworks, deepseek, vllm, ollama, groq
```

Current environment:

```text
MIMO_API_KEY=MISSING
MIMO_BASE_URL=MISSING
MIMO_MODEL=MISSING
```

Escalated provider probe:

```bash
.venv/bin/python -m tests.test_logprobs_providers --provider mimo
```

Current output:

```text
Mimo (custom): available=false, error="Illegal header value b'Bearer '"
```

Root cause: the test path attempts to send an empty `Authorization: Bearer ` header because `MIMO_API_KEY` is not set. Before live MIMO testing, export the MIMO env vars or make the provider code handle missing/optional keys cleanly.

## Verdict

Current v0.1 is not product-safe yet.

It is a good library scaffold with working tests, docs, examples, and CLI shape. The blockers are not volume-of-code blockers. They are correctness-contract blockers:

- thresholds compare incompatible units,
- default middleware can score text without logprobs despite docs saying that is meaningless,
- labels mean different things in different places,
- provider scoring ignores the prompt,
- MIMO is not a first-class provider path yet,
- CLI/docs claim more evaluation behavior than the CLI actually implements.

Fix these before adding UI, SaaS, hosted API, or marketing claims.

## P1-1: Fix Thresholds Comparing Raw Entropy To CES

### Current Problem

`thresholds.assign_thresholds()` uses raw token entropy ECDF values:

```python
thresholds = percentiles(calibration_artifact.ecdf_values)
```

Then `compute_ces()` compares:

```python
ces_score in [0, 1]
```

against thresholds that can be `2.5`, `2.8`, `3.0`, etc. This makes high-risk outputs return `LOW`.

### Correct Design

Default thresholds must be in CES-score space, not entropy space.

Add sequence-level reference CES scores to the calibration artifact:

```python
reference_ces_scores: list[float]
threshold_domain: str = "ces"
threshold_policy: str = "reference_ces_quantile"
```

Build flow:

1. Build `F0` from all selected reference token entropies.
2. For each selected reference sequence:
   - compute `mean_entropy`,
   - compute `max_entropy`,
   - compute `cdf_mean = F0(mean_entropy)`,
   - compute `cdf_max = F0(max_entropy)`,
   - compute `reference_ces = sqrt(cdf_mean * cdf_max)`.
3. Store sorted `reference_ces_scores`.
4. Default thresholds are percentiles of `reference_ces_scores`, not `ecdf_values`.

If an older artifact lacks `reference_ces_scores`, do not silently use `ecdf_values`. Either:

- use static fallback thresholds `{low: 0.75, medium: 0.90, high: 0.97}` with a warning, or
- require recalibration.

Prefer recalibration for production; fallback is acceptable for backward compatibility.

### Files To Change

```text
src/hallucination_sentinel/calibration.py
src/hallucination_sentinel/thresholds.py
src/hallucination_sentinel/ces.py
tests/test_calibration.py
tests/test_thresholds.py
tests/test_ces.py
docs/calibration.md
README.md
```

### Tests To Add First

```python
def test_reference_quantile_thresholds_are_in_ces_space():
    art = build_calibration([np.linspace(1.0, 3.0, 50) for _ in range(20)])
    thresholds = assign_thresholds(art)
    assert 0.0 <= thresholds["low"] <= 1.0
    assert 0.0 <= thresholds["medium"] <= 1.0
    assert 0.0 <= thresholds["high"] <= 1.0

def test_extreme_ces_is_not_low_when_entropy_units_exceed_one():
    art = build_calibration([np.linspace(1.0, 3.0, 50) for _ in range(20)])
    assign_thresholds(art)
    result = compute_ces(np.array([10.0] * 50), art)
    assert result.ces_score == 1.0
    assert result.risk_level == "CRITICAL"
```

### Acceptance Criteria

```bash
.venv/bin/pytest -q tests/test_thresholds.py tests/test_ces.py tests/test_calibration.py
```

Must pass.

The repro must change from:

```text
CES 1.0 -> LOW
```

to:

```text
CES 1.0 -> CRITICAL
```

## P1-2: Remove Text-Only Heuristic From Default Middleware

### Current Problem

`guard_output()` claims to gate a prompt/output pair but calls `_score_text_entropies()`, which estimates entropy from token length, digits, and capitalization.

That contradicts `docs/limitations.md`, which correctly says there is no meaningful text-only fallback without logprobs.

This is dangerous because the RAG and agent examples call `guard_output()` with plain text and look production-ready.

### Correct Design

Make the default safe:

1. `guard_output()` must require real token entropy/logprobs or a provider object capable of scoring.
2. Keep `guard_output_from_entropies()` as the offline and batch-safe path.
3. Keep `guard_output_with_logprobs()` as the provider-safe path.
4. Remove `_score_text_entropies()` from production flow.
5. If a heuristic is kept, rename it clearly:

```python
guard_output_with_text_heuristic_experimental(...)
```

and include a warning:

```text
text_heuristic_not_ces: no provider logprobs were used; this is not CES.
```

The default `SentinelMiddleware` should either:

- wrap an LLM/provider that returns `CompletionLogprobs`, or
- accept an `entropy_extractor` callback, or
- raise a clear error saying logprobs are required.

### Files To Change

```text
src/hallucination_sentinel/integrations/middleware.py
examples/rag_answer_gate.py
examples/agent_tool_preflight.py
tests/
docs/limitations.md
README.md
```

### Tests To Add First

```python
def test_guard_output_without_logprobs_raises():
    art = build_calibration([np.array([0.1, 0.2, 0.3])])
    assign_thresholds(art)
    with pytest.raises(ProviderCapabilityError, match="logprobs"):
        guard_output("prompt", "output", calibration=art)

def test_guard_output_from_entropies_still_works():
    art = build_calibration([np.array([0.1, 0.2, 0.3])])
    assign_thresholds(art)
    decision = guard_output_from_entropies(
        "prompt",
        "output",
        np.array([0.9, 1.2, 1.5]),
        calibration=art,
    )
    assert decision.ces_score >= 0.0
```

### Acceptance Criteria

- No public example presents plain text scoring as real CES.
- Middleware cannot silently run a fake entropy proxy.
- Docs and examples say: use `guard_output_from_entropies()` for offline demo; use `guard_output_with_logprobs()` for production.

## P1-3: Normalize Label Semantics Across Calibration And Eval

### Current Problem

The CLI uses `label: true` as faithful during calibration, but eval logic treats truthy labels as the positive class for hallucination metrics.

This can invert AUROC/AUPRC and confusion matrices.

### Correct Design

Create one label parser used everywhere:

```python
@dataclass(frozen=True)
class ParsedLabel:
    faithful: bool
    hallucinated: bool

def parse_label(value, *, context: str) -> ParsedLabel:
    ...
```

Supported input forms:

Calibration JSONL:

```json
{"label": "faithful"}
{"label": "hallucinated"}
{"label": true}
{"label": false}
```

For legacy calibration booleans:

```text
true = faithful
false = hallucinated
```

Eval JSONL should prefer explicit fields:

```json
{"label": "faithful"}
{"label": "hallucinated"}
```

or:

```json
{"is_hallucinated": true}
```

For integer eval labels:

```text
1 = hallucinated
0 = faithful
```

Important Python detail: `bool` is a subclass of `int`, so check `isinstance(value, bool)` before checking integers.

### Files To Change

```text
src/hallucination_sentinel/calibration.py
src/hallucination_sentinel/cli.py
src/hallucination_sentinel/eval.py
src/hallucination_sentinel/schemas.py
tests/test_calibration.py
tests/test_cli.py
tests/test_eval.py
docs/calibration.md
README.md
```

### Tests To Add First

```python
def test_supervised_calibration_string_labels_include_only_faithful():
    seqs = [np.array([0.1]), np.array([9.9])]
    labels = ["faithful", "hallucinated"]
    art = build_calibration(seqs, labels=labels, mode="supervised")
    assert art.sequence_count == 1
    assert art.ecdf_values == [0.1]

def test_eval_bool_label_true_maps_to_faithful_for_legacy_cli_format():
    parsed = parse_label(True, context="calibration")
    assert parsed.faithful is True
    assert parsed.hallucinated is False

def test_eval_integer_one_maps_to_hallucinated():
    parsed = parse_label(1, context="eval")
    assert parsed.hallucinated is True
```

### Acceptance Criteria

- Calibration and eval cannot disagree about positive class.
- Report JSON explicitly states:

```json
{
  "positive_class": "hallucinated",
  "label_schema": "label: faithful|hallucinated or is_hallucinated: bool"
}
```

## P1-4: Fix Provider Scoring So Prompt Is Not Ignored

### Current Problem

`score-provider` reads `prompt_text` and `output_text`, but then calls:

```python
prov.score_text(output_text)
```

The prompt is ignored. This makes the command arbitrary-text scoring, not prompt-conditioned output scoring.

### Correct Design

Split provider methods by capability:

```python
class BaseProvider(ABC):
    def generate_with_logprobs(self, prompt: str, ...) -> CompletionLogprobs:
        ...

    def score_output(self, prompt: str, output: str, ...) -> CompletionLogprobs:
        ...

    def score_text(self, text: str, ...) -> CompletionLogprobs:
        ...
```

Semantics:

- `generate_with_logprobs`: supported by providers that return logprobs for generated tokens.
- `score_output`: supported only by providers with echo, teacher forcing, or explicit scoring APIs.
- `score_text`: arbitrary standalone text scoring, useful for debugging but not equivalent to prompt-conditioned hallucination risk.

CLI design:

```bash
sentinel score-provider \
  --mode generated \
  --prompt prompt.txt \
  --provider mimo \
  --model "$MIMO_MODEL" \
  --calibration calibration.json
```

```bash
sentinel score-provider \
  --mode score-output \
  --prompt prompt.txt \
  --output output.txt \
  --provider together \
  --model "$MODEL" \
  --calibration calibration.json
```

```bash
sentinel score-provider \
  --mode score-text \
  --output output.txt \
  --provider vllm \
  --model "$MODEL" \
  --calibration calibration.json
```

Rules:

- If `--mode score-output` is requested and provider cannot score output conditioned on prompt, fail clearly.
- If only generated-token logprobs are available, score the generated output returned by the provider, not an arbitrary provided file.
- Include `scoring_mode` in JSON output.

### Files To Change

```text
src/hallucination_sentinel/providers/base.py
src/hallucination_sentinel/providers/openai_compatible.py
src/hallucination_sentinel/cli.py
tests/test_providers.py
tests/test_cli.py
docs/provider_logprobs.md
README.md
```

### Tests To Add First

```python
def test_score_provider_passes_prompt_and_output_to_score_output(mock_provider):
    ...

def test_score_output_fails_when_provider_lacks_echo_or_teacher_forcing():
    ...

def test_generated_mode_does_not_require_output_file():
    ...
```

### Acceptance Criteria

- There is no code path where `--prompt` is accepted but ignored.
- CLI output includes:

```json
{
  "scoring_mode": "generated|score_output|score_text",
  "prompt_conditioned": true,
  "echo_used": false
}
```

## P2-1: Add First-Class MIMO/OpenAI-Compatible Provider Path

### Current Problem

The repo has a generic `OpenAICompatibleProvider.custom()` path, but the CLI does not expose a MIMO preset. `sentinel smoke-provider --provider mimo` fails.

The current MIMO probe also sends an empty `Bearer ` header when `MIMO_API_KEY` is missing.

### Correct Design

Add either a real `mimo` provider preset or a generic `custom` provider CLI path. Prefer both.

Provider spec:

```python
"mimo": ProviderSpec(
    name="MIMO",
    base_url=os.environ.get("MIMO_BASE_URL", "http://localhost:8080/v1"),
    api_key_env="MIMO_API_KEY",
    model=os.environ.get("MIMO_MODEL", "mimo-v2.5-pro"),
    max_top_k=20,
    echo_supported=False,
    requires_api_key=True,
    notes="OpenAI-compatible if MIMO endpoint exposes /chat/completions with logprobs.",
)
```

Add `requires_api_key` to `ProviderSpec`.

Authorization header policy:

```python
headers = {"Content-Type": "application/json"}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
elif spec.requires_api_key:
    raise ValueError("API key missing. Set MIMO_API_KEY or pass --api-key.")
```

CLI additions:

```bash
sentinel smoke-provider \
  --provider mimo \
  --base-url "$MIMO_BASE_URL" \
  --model "$MIMO_MODEL" \
  --api-key "$MIMO_API_KEY"
```

Also support:

```bash
sentinel smoke-provider \
  --provider custom \
  --base-url "$MIMO_BASE_URL" \
  --model "$MIMO_MODEL" \
  --api-key "$MIMO_API_KEY" \
  --max-top-k 20
```

### MIMO Setup For Claude Code

Before running provider tests:

```bash
export MIMO_API_KEY="..."
export MIMO_BASE_URL="https://YOUR_MIMO_OPENAI_COMPATIBLE_BASE_URL/v1"
export MIMO_MODEL="mimo-v2.5-pro"
```

Then:

```bash
.venv/bin/sentinel smoke-provider \
  --provider mimo \
  --base-url "$MIMO_BASE_URL" \
  --model "$MIMO_MODEL" \
  --api-key "$MIMO_API_KEY"
```

If MIMO returns selected-token logprobs only:

- mark CES unavailable,
- keep perplexity baseline available,
- use MIMO for coding only,
- use Together/vLLM/Fireworks/OpenAI-compatible logprob backend for detection.

If MIMO returns top-k logprobs:

- set `entropy_mode = "top_k_with_residual"`,
- store `top_logprobs`,
- run calibration/eval on MIMO outputs.

### Tests To Add First

```python
def test_mimo_provider_preset_exists():
    assert "mimo" in PROVIDER_SPECS

def test_missing_mimo_key_has_clear_error(monkeypatch):
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    provider = OpenAICompatibleProvider.from_preset("mimo")
    with pytest.raises(ValueError, match="MIMO_API_KEY"):
        provider.check_health()

def test_empty_api_key_does_not_send_illegal_bearer_header():
    ...
```

### Acceptance Criteria

- `sentinel smoke-provider --provider mimo` no longer fails as unknown provider.
- Missing key produces a clean config error.
- Present key reaches the API and reports:
  - selected-token logprobs,
  - top-k logprobs,
  - max top-k,
  - arbitrary text scoring support.

## P2-2: Fix CLI Command Mismatch: `check` vs `score`

### Current Problem

The completion summary says:

```text
sentinel check
```

But the actual CLI has:

```text
sentinel score
```

and:

```bash
sentinel check --help
```

returns:

```text
Error: No such command 'check'
```

### Correct Design

Either update all docs and summaries to say `score`, or add `check` as an alias.

Prefer adding alias:

```python
@main.command("check")
...
def check(...):
    return score(...)
```

But avoid duplicate code. Extract shared implementation:

```python
def _score_entropy_file(entropy_json: str, calibration: str) -> dict:
    ...
```

Then both commands call it.

### Acceptance Criteria

```bash
sentinel score --entropy-json score.json --calibration calibration.json
sentinel check --entropy-json score.json --calibration calibration.json
```

Both work and produce the same JSON fields.

## P2-3: Align README Eval Claims With Real CLI

### Current Problem

README promises:

- bootstrap confidence intervals,
- per-length-bucket calibration coverage,
- lag-1 autocorrelation diagnostics,
- multiple confusion matrices at configured thresholds.

The current CLI `eval` only computes basic AUROC/AUPRC and one median-threshold confusion matrix.

The richer `src/hallucination_sentinel/eval.py` exists, but CLI does not fully use it.

### Correct Design

Wire CLI `eval` into `eval.py` instead of reimplementing a smaller eval path.

CLI should support:

```bash
sentinel eval \
  --input eval.jsonl \
  --calibration calibration.json \
  --output eval_report.json \
  --markdown eval_report.md \
  --threshold 0.5 \
  --threshold 0.7 \
  --threshold 0.8 \
  --threshold 0.9 \
  --bootstrap \
  --n-bootstrap 1000
```

Input modes:

1. Raw entropy eval rows:

```json
{"entropy": [0.1, 0.2], "label": "faithful"}
{"entropy": [2.1, 3.2], "label": "hallucinated"}
```

2. Pre-scored eval rows:

```json
{"ces_score": 0.8, "label": 1, "token_count": 50, "token_entropies": [1.0, 2.0]}
```

Normalize both into `EvalRecord`.

### Acceptance Criteria

- README claims match actual CLI output.
- CLI report includes:
  - AUROC,
  - AUPRC,
  - bootstrap confidence intervals when enabled,
  - confusion matrices at requested thresholds,
  - length buckets,
  - autocorrelation diagnostics,
  - baseline comparisons when baseline fields exist.

## P2-4: Make `compute_token_entropies()` Harder To Misuse

### Current Problem

`compute_token_entropies()` uses `0.0` placeholder entropies when only selected-token logprobs exist. That can flow into CES if callers ignore warnings.

Mode auto-detection also relies on optional `mode` fields instead of actual data shape.

### Correct Design

- If only selected-token logprobs exist, return `entropy_mode="selected_only"` and `entropies=None` or raise a clear `ProviderCapabilityError` for CES paths.
- Do not append `0.0` placeholder entropy for selected-only mode.
- Detect mode from actual keys:
  - `logprobs` -> `full`,
  - `top_logprobs` -> `top_k_with_residual`,
  - `selected_logprob` only -> `selected_only`.

### Acceptance Criteria

- CES cannot be computed from selected-token-only placeholder zeros.
- Perplexity still works from selected logprobs.
- Tests cover all three provider capability levels.

## Execution Order For Claude Code With MIMO

Paste this instruction into Claude Code after opening the repo:

```text
Read PLAN_GPT_5.5_X_HIGH_V2.md. Implement it in strict priority order.

Rules:
1. Add failing tests before each fix.
2. Fix P1-1 thresholds first.
3. Fix P1-2 middleware logprob contract second.
4. Fix P1-3 label semantics third.
5. Fix P1-4 provider scoring prompt/output semantics fourth.
6. Then do P2 MIMO/custom provider, check alias, eval CLI alignment, and selected-only hardening.
7. Run pytest after each phase.
8. Do not add hosted SaaS, dashboard, auth, database, or UI.
9. Do not market this as hallucination detection certainty. Keep "risk signal" language.
10. Preserve existing public APIs where possible, but prefer explicit errors over silent fake scoring.
```

Recommended MIMO/Claude Code environment:

```bash
cd /Users/heman10x/Downloads/codex_dev/hallucination-sentinel
source .venv/bin/activate
export MIMO_API_KEY="..."
export MIMO_BASE_URL="..."
export MIMO_MODEL="mimo-v2.5-pro"
```

Then run:

```bash
pytest -q
sentinel --help
sentinel smoke-provider --provider mimo --base-url "$MIMO_BASE_URL" --model "$MIMO_MODEL" --api-key "$MIMO_API_KEY"
```

If MIMO does not expose top-k logprobs, continue using MIMO as the coding model only and use a verified top-k logprob backend for Sentinel detection.

## Final Verification Matrix

Run all of this before calling V2 complete:

```bash
.venv/bin/pytest -q
.venv/bin/sentinel --help
.venv/bin/sentinel check --help
.venv/bin/sentinel score --help
.venv/bin/sentinel eval --help
.venv/bin/sentinel smoke-provider --provider mimo --base-url "$MIMO_BASE_URL" --model "$MIMO_MODEL" --api-key "$MIMO_API_KEY"
.venv/bin/python -m tests.test_logprobs_providers --provider mimo
```

Run the threshold repro again and require:

```text
CES 1.0 -> CRITICAL
```

Run a no-logprobs middleware call and require:

```text
clear ProviderCapabilityError
```

Run an offline batch example and require:

```text
guard_output_from_entropies works
```

Run eval with both label styles:

```json
{"label": "faithful"}
{"label": "hallucinated"}
{"is_hallucinated": true}
{"label": 1}
```

and require the output report to state:

```json
{"positive_class": "hallucinated"}
```

## Definition Of Done

V2 is done only when:

- thresholds are CES-domain thresholds,
- no default product path pretends text-only heuristics are CES,
- label semantics are explicit and tested,
- provider prompt/output scoring is semantically correct,
- MIMO/custom provider can be smoke-tested from CLI,
- missing MIMO keys fail cleanly,
- `sentinel check` either exists as an alias or all docs remove it,
- CLI eval output matches README claims,
- all tests pass,
- live provider tests are either passing with keys or clearly skipped with documented reason.
