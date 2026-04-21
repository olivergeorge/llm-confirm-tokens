# llm-confirm-tokens

Interactive "you are about to send N tokens, proceed?" gate for the
[`llm`](https://llm.datasette.io) CLI.

When enabled, the plugin intercepts each prompt immediately before it
reaches the model, counts the tokens on the resolved request
(system + fragments + prompt), and asks for confirmation on `/dev/tty`:

```
$ files-to-prompt llm_replay | llm "what colour is the wind?"
Total tokens: 7,391. Proceed? [Y/n]:
```

Anything other than a bare `Enter`, `y`, or `yes` raises
`llm.CancelPrompt` — the upstream API is **not** called, the
conversation is not updated, and the CLI exits non-zero.

## Requirements

This plugin depends on the `register_prompt_gates` hookspec. It is
currently available on the `llm-prompt-gates-hook` branch of
[olivergeorge/llm](https://github.com/olivergeorge/llm) pending upstream
merge.

## Install

```bash
llm install llm-confirm-tokens
```

Accurate token counting uses `tiktoken`'s `cl100k_base` encoding; without
it the plugin falls back to a `len(text) // 4` heuristic.

```bash
llm install 'llm-confirm-tokens[tiktoken]'
```

## What it counts

The estimate covers the whole resolved prompt, locally and without any
network calls:

| Part | Estimation |
| ---- | ---------- |
| `prompt`, `system`, fragments | tiktoken (`cl100k_base`) or `len // 4` |
| Text attachments (`text/*`, JSON, YAML, XML) | read from disk + tokenised |
| Image attachments | 258 tokens each (matches Gemini's per-image cost) |
| PDF attachments | `pages × 258`; page count inferred from the raw bytes |
| Tool definitions | JSON-serialised and tokenised |
| Prior tool results | `str(output)` tokenised |
| Structured-output schema | JSON-serialised and tokenised |
| URL-only attachments | flat 300 (never fetched) |
| Unknown binary blobs | flat 300 |

Numbers won't match a specific provider's internal tokeniser exactly —
different providers charge different per-image / per-page rates — but
they are typically within 10–30 % of `llm -u`, which is the "did I just
attach a 200-page PDF?" signal the plugin is meant to catch.

## Exact counts for Claude (opt-in)

If you want billing-grade accuracy for Claude models — including images,
PDFs, and tool schemas — install the `anthropic` extra and set
`LLM_CONFIRM_TOKENS_EXACT=1`:

```bash
llm install 'llm-confirm-tokens[anthropic]'
export LLM_CONFIRM_TOKENS=1
export LLM_CONFIRM_TOKENS_EXACT=1
```

When exact mode is on, a Claude model prompt is sent to Anthropic's
free [`/v1/messages/count_tokens`](https://docs.anthropic.com/en/api/messages-count-tokens)
endpoint before the real call. One extra round-trip (~100–300ms) per
gated prompt, no billing. Anything that doesn't match — non-Claude
models, SDK missing, key missing, network error — silently falls back
to the local heuristic, so turning exact mode on never breaks gating.

URL-only attachments are deliberately not fetched by the adapter (same
as the heuristic) — the plugin never makes pre-flight HTTP calls of
its own beyond the single `count_tokens` request.

## Enable

The plugin is **opt-in** so installing it does not change the behaviour
of existing scripts. Enable per shell session:

```bash
export LLM_CONFIRM_TOKENS=1
```

Options via environment variables:

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `LLM_CONFIRM_TOKENS` | *unset* | Set to `1` / `true` / `yes` / `on` to register the gate. Unset or `0` means "plugin installed but inactive". |
| `LLM_CONFIRM_TOKENS_THRESHOLD` | `0` | Only prompt when the estimated token count is at or above this number. `0` means confirm on every prompt. |
| `LLM_CONFIRM_TOKENS_YES` | *unset* | Auto-approve without prompting. Useful inside `LLM_CONFIRM_TOKENS=1` shells when running a batch script you trust. |
| `LLM_CONFIRM_TOKENS_EXACT` | *unset* | Opt-in: use provider-native count APIs instead of the local heuristic when a matching SDK is installed. See "Exact counts" below. |

The confirmation is read from `/dev/tty`, so the plugin works correctly
even when `stdin` is a piped file (`cat big.txt | llm …`). On systems
without a `/dev/tty` (some CI sandboxes) the plugin prints the token
count to stderr and proceeds — interactive scripts get the prompt,
non-interactive scripts are not blocked.

## Using it as a Python library

The gate is plain Python; you can instantiate it directly if you'd
rather not gate on env vars:

```python
import llm
from llm_confirm_tokens import ConfirmTokensGate

my_gate = ConfirmTokensGate(
    threshold=1000,
    tokens_fn=lambda prompt: my_counter(prompt),
    ask=lambda n: click.confirm(f"{n} tokens, proceed?"),
)

@llm.hookimpl
def register_prompt_gates(register):
    register(my_gate)
```

`check(prompt, model)` is the entry point used by `llm`; raise
`llm.CancelPrompt` from any custom `ask` to cancel.

## Development

```bash
uv sync --dev
uv run pytest
```

The repo depends on an editable `../llm` checkout for the hookspec; see
`pyproject.toml`'s `[tool.uv.sources]` if your layout differs.
