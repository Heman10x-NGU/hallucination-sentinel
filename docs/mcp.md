# MCP Server

Hallucination Sentinel exposes a Model Context Protocol (MCP) server for agent
workspaces that need calibrated entropy risk scoring.

This is not a text-only hallucination detector. The server needs one of:

- a pre-computed per-token entropy sequence
- top-k token logprobs
- a provider call that can return top-k logprobs for the output being scored

Claude, Cursor, Codex, and other agent hosts do not automatically expose their
own internal token logprobs to MCP tools. If a host only sends plain text, the
server cannot honestly compute CES.

## Install

```bash
pip install "hallucination-sentinel[mcp] @ git+https://github.com/Heman10x-NGU/hallucination-sentinel.git"
```

## Run

```bash
sentinel-mcp --calibration /absolute/path/to/calibration.json
```

The MCP server runs over stdio. It does not write logs to stdout.

## Claude Desktop Configuration

Add this to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hallucination-sentinel": {
      "command": "sentinel-mcp",
      "args": ["--calibration", "/absolute/path/to/calibration.json"],
      "env": {
        "TOGETHER_API_KEY": "your-provider-key"
      }
    }
  }
}
```

Prefer environment variables for API keys. Do not paste secrets into prompts.

## Tools

### `score_entropy_sequence`

Scores a sequence of pre-computed token entropies.

Example prompt:

```text
Use hallucination-sentinel to score this entropy sequence:
[0.32, 0.41, 0.58, 1.94, 2.12, 0.77]
```

### `score_topk_logprobs`

Converts top-k logprobs to approximate per-token entropy and computes CES.

Example tool input:

```json
{
  "topk_logprobs": [
    {" Paris": -0.10, " London": -2.10, " Berlin": -2.80},
    {" is": -0.05, " was": -3.00},
    {" the": -0.02, " a": -3.50},
    {" capital": -0.20, " largest": -1.90},
    {" of": -0.01, " in": -4.10},
    {" France": -0.15, " Germany": -2.20}
  ]
}
```

### `score_provider_output`

Scores an existing prompt/output pair through a provider that supports
echo-style or teacher-forced output scoring.

Providers with echo support are the best fit for this tool. The server rejects
selected-token-only logprobs because CES requires top-k logprobs or full logits
to estimate entropy.

Example prompt:

```text
Use hallucination-sentinel to score this answer with provider logprobs.
Provider: together
Model: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
Prompt: What is the capital of France?
Output: The capital of France is Berlin.
```

### `smoke_provider`

Checks whether a provider can return logprobs.

Example prompt:

```text
Use hallucination-sentinel to smoke-test Together AI logprob support for
meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo.
```

### `inspect_calibration`

Returns calibration metadata: model, provider, task family, thresholds, ECDF
size, and known limitations.

## Response Shape

Scoring tools return structured JSON including:

- `ces_score`
- `risk_level`
- `mean_entropy`
- `max_entropy`
- `token_count`
- `warnings`
- `action`
- `risk_signal_note`
- `calibration`

Every scoring response includes this reminder:

```text
CES is a calibrated entropy risk signal, not a truth oracle. A high score does
not prove hallucination, and a low score does not prove correctness.
```

## Limitations

- MCP does not magically expose the host model's internal logprobs.
- Plain text alone is insufficient for production CES scoring.
- Top-k entropy is approximate because providers usually do not expose the full
  vocabulary distribution.
- Calibration is tied to the model, provider, task family, and decoding setup.
- This is a risk-routing tool, not a replacement for fact-checking or evidence
  retrieval.
