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

This plugin depends on the `register_prompt_gates` hookspec, which is
**not yet in upstream `llm`**.

`register_prompt_gates` lets a plugin register a gate that runs on the
resolved prompt *immediately before* `Model.execute(...)` is called,
and raise `llm.CancelPrompt` to abort the request before any tokens
leave the machine. That's the whole feature — without the hookspec
there is nowhere in `llm`'s surface to intercept a prompt at that
point, and the plugin would have to monkey-patch `_BaseResponse` or
subclass every `Model` (fragile across `llm` versions, hostile to
other plugins).

The hookspec lives on the `llm-prompt-gates-hook` branch of
[olivergeorge/llm](https://github.com/olivergeorge/llm) (also merged
into `llm-replay-combined` alongside the replay hookspecs) pending
upstream merge.

## Install

Install the fork of `llm` that carries the hookspec, then install the
plugin against it:

```bash
# Clone and install the fork on the branch with the hookspec
git clone -b llm-prompt-gates-hook https://github.com/olivergeorge/llm.git
llm install -e ./llm

# Install the plugin
llm install llm-confirm-tokens
```

If you already have a local checkout of `../llm`, `pyproject.toml`
points at it as an editable dependency — just check out the
`llm-prompt-gates-hook` (or `llm-replay-combined`) branch there and
run `uv sync` / `pip install -e .`.

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

## Differences & assumptions per adapter

Each provider's `count_tokens` surface has small quirks. The adapters
all prioritise **working on every model in a provider's lineup** over
matching billing to the nearest token. The shorthand is: an exact
count is typically within ~5% of the real bill, always erring high.

### Anthropic (`[anthropic]`)

- **System prompt**: sent as the top-level `system` string
  (Claude's own Messages API shape). Matches billing exactly.
- **Attachments**: images become `{"type":"image", ...}`, PDFs become
  `{"type":"document", ...}`. URL-only attachments are dropped rather
  than fetched — the adapter makes exactly one HTTP call (the
  `count_tokens` request itself) and no more.
- **Tools**: included. Serialised as `{name, description, input_schema}`
  exactly as Claude expects.
- **Tool results from prior turns**: not included. Representing tool
  use faithfully requires a specific alternating assistant/user message
  history whose invariants `count_tokens` enforces, and
  mis-structuring it would make the request fail rather than just
  under-count. The local heuristic has already folded tool results
  into its baseline, so the leftover under-count is minor in practice.

### Gemini (`[gemini]`)

- **System prompt**: inlined as leading user-role text parts, **not**
  routed through `CountTokensConfig.system_instruction`. The
  `count_tokens` endpoint rejects `system_instruction` for several
  Gemini models (e.g. `gemini-flash-lite` raises `ValueError:
  system_instruction parameter is not supported in Gemini API`),
  even when the same model accepts it on `generate_content`.
  Inlining keeps the adapter model-agnostic at the cost of ~5% of
  envelope tokens that don't actually hit the bill.
- **Attachments**: images and PDFs go in as `inline_data` parts with
  base64-encoded content. URL-only attachments are dropped (no HEAD
  or GET just to count).
- **Tools and tool results**: not included. Gemini's tool format
  requires a multi-turn message history whose invariants count_tokens
  validates. The local heuristic already accounts for tools.
- **Tokeniser parity**: Gemini tokenises text in `parts[]` the same
  way as in `system_instruction`; the delta is envelope-only.

### OpenAI (`[openai]`)

- **Endpoint**: `/v1/responses/input_tokens`, i.e. the Responses-API
  preflight counter. The Chat Completions and Responses APIs share
  the same tokeniser, so this count is accurate even for `llm`
  models that send their real prompt via chat completions.
- **System prompt**: sent as the top-level `instructions` kwarg (what
  the Responses API calls it).
- **Attachments**: images become `input_image` with a base64 data URL;
  PDFs become `input_file` with `filename` + `file_data`. URL-only
  attachments are dropped.
- **Tools and schemas**: not included in the current adapter. Tool
  schemas materially change the counted total on OpenAI — this is a
  known gap; the heuristic's JSON-serialised tool estimate is used
  as a rough baseline until the adapter grows tool support.
- **Minimum SDK**: `openai>=2.0` for the `responses.input_tokens`
  resource. Older SDKs trigger the silent-fallback-plus-stderr-warning
  path.

### Shared assumptions

- **URL-only attachments are never fetched** by any adapter. A
  gating tool should not cause new outbound HTTP traffic to third
  parties the user hasn't already consented to — the single
  `count_tokens` request to their own provider is the one exception.
- **Binary attachments we don't know how to handle** (audio, video,
  bespoke file types) are dropped by the adapters. The heuristic
  still counts them against its flat `_UNKNOWN_BINARY_TOKENS` budget,
  so the estimate doesn't silently skip to zero.
- **Adapter errors always fall back to the heuristic**, with a one-
  line stderr notice naming the provider and the exception. Gating
  is the feature; exact mode is a refinement. A broken refinement
  must never break the feature.

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
