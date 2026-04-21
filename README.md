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

## Exact counts (opt-in)

If you want billing-grade accuracy — including images, PDFs, and tool
schemas — install the extra for your provider and set
`LLM_CONFIRM_TOKENS_EXACT=1`:

```bash
llm install 'llm-confirm-tokens[anthropic]'   # Claude models
llm install 'llm-confirm-tokens[gemini]'      # Gemini models
llm install 'llm-confirm-tokens[openai]'      # GPT / o-series models

export LLM_CONFIRM_TOKENS=1
export LLM_CONFIRM_TOKENS_EXACT=1
```

When exact mode is on and the model matches a registered adapter, the
plugin calls the provider's free pre-flight count endpoint before the
real request:

| Provider | Endpoint | SDK | Extra |
| -------- | -------- | --- | ----- |
| Anthropic | [`/v1/messages/count_tokens`](https://docs.anthropic.com/en/api/messages-count-tokens) | `anthropic` | `[anthropic]` |
| Google Gemini | `Client.models.count_tokens` | `google-genai` | `[gemini]` |
| OpenAI | `/v1/responses/input_tokens` | `openai>=2.0` | `[openai]` |

One extra round-trip (~100–300 ms) per gated prompt, no billing.
Anything that doesn't match — non-matching model, SDK missing, key
missing, network error — silently falls back to the local heuristic,
so turning exact mode on never breaks gating.

Keys are resolved first from the canonical env var
(`ANTHROPIC_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`,
`OPENAI_API_KEY`) and then from llm's own keyring, so no extra
configuration is needed for users who already have `llm keys set`
configured.

URL-only attachments are deliberately not fetched (same as the
heuristic) — the plugin never makes pre-flight HTTP calls of its own
beyond the single `count_tokens` request.

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

The confirmation is read from `/dev/tty` (POSIX) or `CONIN$` (Windows),
so the plugin works correctly even when `stdin` is piped
(`cat big.txt | llm …`). If no interactive terminal is available at
all — typical in CI sandboxes — the plugin **fails closed**: it prints
the token count to stderr and raises `CancelPrompt`. Scripts that
want to run unattended with the plugin installed should set
`LLM_CONFIRM_TOKENS_YES=1`.

Heuristic estimates are prefixed with `~` and exact counts show the
provider name, so you can tell at a glance which one you got:

```
Total tokens: ~7,391. Proceed? [Y/n]:         # heuristic
Total tokens: 7,412 (anthropic). Proceed? [Y/n]:   # exact
```

If exact mode is enabled but the adapter fails (missing key, network
flake, SDK too old), the plugin falls back to the heuristic and
writes a one-line notice to stderr so you know silent degradation
isn't happening.

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
