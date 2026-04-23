# llm-confirm-tokens

Interactive "you are about to send N tokens, proceed?" gate for the
[`llm`](https://llm.datasette.io) CLI.

When enabled, the plugin intercepts each prompt immediately before it
reaches the model, counts the tokens on the resolved request
(system + fragments + prompt), and asks for confirmation on `/dev/tty`:

```
$ files-to-prompt llm_replay | llm "what colour is the wind?"
7.4k input tokens (estimate). Proceed? [Y/n]:
```

Anything other than a bare `Enter`, `y`, or `yes` raises
`llm.CancelPrompt` — the upstream API is **not** called, the
conversation is not updated, and the CLI exits non-zero.

## Requirements

This plugin depends on **two hookspecs that are not yet in upstream
`llm`**. Both live on branches of
[olivergeorge/llm](https://github.com/olivergeorge/llm), pending
upstream merge. The simplest way to satisfy both is to check out the
combined branch; alternatively you can merge the two branches
individually if you already carry other forks.

| Hook | Purpose in this plugin | Branch |
| ---- | ---------------------- | ------ |
| [`register_prompt_gates`](https://github.com/olivergeorge/llm/blob/llm-prompt-gates-hook/docs/plugins/plugin-hooks.md#register_prompt_gatesregister) | Register the gate that runs before `Model.execute` so we can count tokens and raise `CancelPrompt`. | [`llm-prompt-gates-hook`](https://github.com/olivergeorge/llm/tree/llm-prompt-gates-hook) |
| [`after_log_to_db`](https://github.com/olivergeorge/llm/blob/llm-after-log-to-db/docs/plugins/plugin-hooks.md#after_log_to_dbresponse-db) | Fires after the real response has been persisted. Powers `LLM_CONFIRM_TOKENS_DRIFT_WARN` by comparing the pre-flight heuristic to the provider's billed `input_tokens`. | [`llm-after-log-to-db`](https://github.com/olivergeorge/llm/tree/llm-after-log-to-db) |
| Both of the above + the replay-stores hookspec | Single-branch install. | [`combined-prs`](https://github.com/olivergeorge/llm/tree/combined-prs) |

Without `register_prompt_gates` there is nowhere in `llm`'s surface to
intercept a prompt before it leaves the machine — the plugin would have
to monkey-patch `_BaseResponse` or subclass every `Model`, which is
fragile across versions and hostile to other plugins. Without
`after_log_to_db` the drift warning degrades to exact-mode-only (no
signal when the heuristic drifts on a model without an exact-count
adapter).

### `register_prompt_gates` signature: `check(prompt, model, conversation=None)`

Core calls each registered gate with three arguments:

- `prompt` — the fully-resolved `llm.Prompt`.
- `model` — the `llm.Model` that will execute the prompt.
- `conversation` — the `llm.Conversation` the prompt belongs to, or
  `None` for one-shot prompts.

The `conversation` kwarg is what lets this plugin cost `llm -c`
correctly — `conversation.responses` holds the prior turns the model
plugin will re-send alongside the new prompt, and we fold them into
the count.

Two compatibility details in core:

- Core passes `conversation` as a keyword argument and falls back to
  `check(prompt, model)` if the gate's signature doesn't accept it, so
  pre-existing gates keep working (they just won't see history).
- Async responses invoke `acheck(prompt, model, conversation=None)`
  when present, falling back to the sync `check` otherwise.

If you're tracking a pinned revision of the fork, any commit on or
after the "pass conversation to gate.check" change on
`llm-prompt-gates-hook` (or its `combined-prs` equivalent) carries the
`conversation` kwarg. Older gates on older cores still run, so the pin
is forward-compatible.

### Adapters still own provider wire formats — for now

Today the plugin ships per-provider "exact count" adapters that
rebuild each provider's `count_tokens` payload from the `Prompt` and
`Conversation`. That's a duplicate of what the model plugins
themselves do inside `Model.execute`, and it drifts every time a
provider changes attachment shapes or role names. See
[ADR 0001: Model-owned token counting](docs/adr/0001-model-owned-token-counting.md)
for the proposal to move counting into the model plugins themselves
(`Model.count_tokens(prompt, conversation)`), with the heuristic as
the fallback — status: **draft**, feedback welcome.

## Install

Install the fork of `llm` that carries both hookspecs, then install
the plugin against it. `combined-prs` is the single-branch superset
that carries `register_prompt_gates`, `after_log_to_db`, and the
replay-stores hookspec in one place:

```bash
# Clone and install the fork on the combined branch
git clone -b combined-prs https://github.com/olivergeorge/llm.git
llm install -e ./llm

# Install the plugin
llm install llm-confirm-tokens
```

If you only need gating (no drift warnings), `llm-prompt-gates-hook`
is sufficient on its own.

If you already have a local checkout of `../llm`, `pyproject.toml`
points at it as an editable dependency — check out `combined-prs`
there (or `llm-prompt-gates-hook` if you don't need drift warnings)
and run `uv sync` / `pip install -e .`.

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
| Image attachments | provider-aware formula from parsed width/height — see below |
| PDF attachments | `pages × per-page-rate`; page count from `/Type /Page` markers (bytes fallback only for compressed PDFs) and per-page rate picked by provider — see below |
| Tool definitions | JSON-serialised and tokenised |
| Prior tool results | `str(output)` tokenised |
| Structured-output schema | JSON-serialised and tokenised |
| URL-only attachments | flat 300 (never fetched) |
| Unknown binary blobs | flat 300 |
| Prior conversation turns (`llm -c` / `--cid`) | user body + attachments + assistant output, per turn |

### Conversation history (`llm -c`)

When a prompt is a continuation of an existing conversation, the model
plugin replays every prior turn to the provider on the next request —
so the bill on turn five of a chat includes everything from turns one
through four. The gate sees this via a `conversation` argument core
now passes through, and folds each prior turn's user body,
attachments, and assistant output into the count. Without this, a
continued chat would under-count to the size of only the newest
message. System prompts are still counted exactly once (on the current
turn), matching what providers actually accept. Tool calls and tool
results from prior turns are skipped for the same reason they're
skipped on the current-turn exact adapters — mis-structuring them can
cause the count endpoint to reject the request.

### Image formula by provider

The heuristic parses width and height straight from the image header
(PNG, JPEG, GIF, WebP — no Pillow dependency) and applies the
provider's own documented formula:

| Model class | Rule | Source |
| ----------- | ---- | ------ |
| Gemini (default) | ≤ 384×384 → 258 tokens; otherwise `tile = clamp(min(w,h)/1.5, 256, 768)` and each tile costs 258 | [Gemini image understanding: image tokenization](https://ai.google.dev/gemini-api/docs/image-understanding) |
| Anthropic `claude-*` | Downscale so longest side ≤ 1568, then `round(w × h / 750)` | [Claude vision: evaluating image size](https://platform.claude.com/docs/en/build-with-claude/vision#evaluate-image-size) |
| OpenAI `gpt-*` / `o1`/`o3`/`o4` | Fit in 2048², scale shortest side to 768, `85 + 170 × ceil(w/512) × ceil(h/512)` (high-detail tiles) | [OpenAI vision: calculating costs](https://platform.openai.com/docs/guides/vision) |

When we can't parse the image (exotic formats, corrupt bytes), each
image falls back to a flat 258 tokens.

### PDF per-page rate by provider

PDF costs are given as a **range per page**, not a single number — the
documented rates turn out to be lower bounds for bare text pages, while
real PDFs with embedded images hit the upper bound or higher. The
confirmation prompt shows the range directly so you can see the
uncertainty at a glance:

```
3.1k–12k input tokens (estimate). Proceed? [Y/n]:
```

Text-only prompts, parseable images, and tool schemas collapse to a
single number (`low == high`), so the range only appears when there's
real uncertainty to communicate.

| Model class | Per-page range | Source |
| ----------- | -------------- | ------ |
| Gemini (default) | 258 – 1,032 | [Google: "each document page equals 258 tokens"](https://ai.google.dev/gemini-api/docs/document-processing); high bound applies Gemini's own tile formula to the documented 768×768 minimum rendered page size (~4 tiles × 258). |
| Anthropic `claude-*` | 1,500 – 3,000 | [Anthropic's documented 1,500–3,000 text tokens per page range](https://platform.claude.com/docs/en/docs/build-with-claude/pdf-support) — image tokens are charged on top, so real bills can still exceed the upper bound. |
| OpenAI `gpt-*` / `o1`/`o3`/`o4` | 255 – 765 | Inferred (OpenAI's [file-inputs docs](https://platform.openai.com/docs/guides/pdf-files) don't give a per-page number). Low = single low-detail tile; high = high-detail letter-size page (85 + 170 × 4). |

The page count itself comes from counting `/Type /Page` markers in the
PDF stream; the bytes-per-page fallback is only used when the PDF is
compressed into an object stream and the regex returns 0.

The drift warning (`LLM_CONFIRM_TOKENS_DRIFT_WARN`) is designed to
catch workloads that leave the range entirely — e.g. very image-heavy
PDFs where Gemini's embedded-image billing exceeds even the high bound.

### These are a best guess, not billing-grade

Local estimates will drift from provider billing whenever a provider
changes how it charges images, PDFs, tools, or system envelopes — and
that happens without warning. The confirmation line prefixes heuristic
counts with a `(estimate)` suffix and a humanised number —
`1.0k input tokens (estimate). Proceed?` — so you can see at a glance
that this isn't a bill. Exact counts from a provider's own tokeniser
stay at full precision and carry the provider name instead:
`7,412 input tokens (anthropic). Proceed?`. Typical heuristic accuracy
against `llm -u` is
within 10–30 % when our formula matches the provider's current rule;
expect larger gaps during pricing rollovers. For the "did I just
attach a 200-page PDF?" failure mode the heuristic is more than enough;
for cost accounting, use **exact mode** below.

If you want to know when the heuristic is drifting on *your*
workload, set `LLM_CONFIRM_TOKENS_DRIFT_WARN` to a percentage (e.g.
`25`). The plugin compares the pre-flight heuristic to the actual
billed count from the response (`after_log_to_db` hook) and writes a
one-line stderr notice — but **only when the estimate under-counted
reality**, since that's the case that risks bill shock. Over-counts
(the "pleasantly surprised" case) stay silent.

```
llm-confirm-tokens: heuristic 258 under-counts vs gemini-flash-latest
billed 1,080 by 76% — local estimates are a best guess, not billing-grade.
```

Detection works in **both modes**:

- **Heuristic-only mode** — the primary value. You have no local
  ground truth, so this is the only way to discover that the formula
  doesn't match how your provider bills today. Uses the response's
  `input_tokens` after the fact; no extra API calls.
- **Exact mode** — acts as a sanity check on the adapter itself. If
  the exact count and the heuristic diverge, either the heuristic is
  stale or the adapter is sending a badly-shaped request. Either way,
  the warning fires as soon as the gate runs, without waiting for the
  response.

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
| `LLM_CONFIRM_TOKENS_THRESHOLD` | `0` | Only prompt when the estimated token count is at or above this number (the floor of "ask me"). `0` means confirm on every prompt. |
| `LLM_CONFIRM_TOKENS_MAX` | `0` | Hard ceiling — cancel without asking when the estimate reaches this number. `0` means no ceiling. Evaluated before `LLM_CONFIRM_TOKENS_YES`, so scripts that auto-approve can still refuse runaway payloads. |
| `LLM_CONFIRM_TOKENS_YES` | *unset* | Auto-approve without prompting. Useful inside `LLM_CONFIRM_TOKENS=1` shells when running a batch script you trust. |
| `LLM_CONFIRM_TOKENS_EXACT` | *unset* | Opt-in: use provider-native count APIs instead of the local heuristic when a matching SDK is installed. See "Exact counts" below. |
| `LLM_CONFIRM_TOKENS_DRIFT_WARN` | *unset* | Percentage threshold for drift warnings. Compares the pre-flight heuristic against the billed count (after the response completes) — and, when exact mode is on, also against the exact count (before the response is sent). Off by default to keep the gate quiet. |
| `LLM_CONFIRM_TOKENS_DRY_RUN` | *unset* | Count tokens and exit 0 without sending the prompt. Prints the human-readable estimate to stderr and the raw integer to stdout, so `TOKENS=$(LLM_CONFIRM_TOKENS_DRY_RUN=1 llm …)` captures the number cleanly. Bypasses `THRESHOLD`, `MAX`, and `YES` — the user asked for the number, not a gate. Registers the gate on its own, so you do not also need `LLM_CONFIRM_TOKENS=1`. |

The confirmation is read from `/dev/tty` (POSIX) or `CONIN$` (Windows),
so the plugin works correctly even when `stdin` is piped
(`cat big.txt | llm …`). If no interactive terminal is available at
all — typical in CI sandboxes — the plugin **fails closed**: it prints
the token count to stderr and raises `CancelPrompt`. Scripts that
want to run unattended with the plugin installed should set
`LLM_CONFIRM_TOKENS_YES=1`.

The confirmation prompt always follows the shape
`{count} input tokens ({source}). Proceed? [Y/n]:`. The source is
`estimate` for heuristic counts and the provider name when exact mode
returned an authoritative count, so you can tell at a glance which one
you got:

```
7.4k input tokens (estimate). Proceed? [Y/n]:       # heuristic (rounded)
7,412 input tokens (anthropic). Proceed? [Y/n]:     # exact (precise)
```

If exact mode is enabled but the adapter fails (missing key, network
flake, SDK too old), the plugin falls back to the heuristic and
writes a one-line notice to stderr so you know silent degradation
isn't happening.

### Recipes

`THRESHOLD`, `MAX`, and `YES` compose into three bands — auto-approve
small prompts, confirm mid-sized ones, refuse anything huge — which
covers most of the workflows this plugin was built for:

```bash
# Interactive shell: confirm every prompt above 1k tokens, refuse anything
# that would send more than 50k. Small prompts go through silently.
export LLM_CONFIRM_TOKENS=1
export LLM_CONFIRM_TOKENS_THRESHOLD=1000
export LLM_CONFIRM_TOKENS_MAX=50000
```

```bash
# "Let me pipe anything into Gemini Pro, but cap the blast radius."
# MAX trumps YES, so you never get asked but you also never send >50k.
alias llm-gemini='LLM_CONFIRM_TOKENS=1 LLM_CONFIRM_TOKENS_YES=1 \
  LLM_CONFIRM_TOKENS_MAX=50000 llm -m gemini-pro'
```

```bash
# Batch script: auto-approve everything, but bail out if a single prompt
# ever balloons past your sanity budget. Exits non-zero so `set -e` works.
LLM_CONFIRM_TOKENS=1 LLM_CONFIRM_TOKENS_YES=1 LLM_CONFIRM_TOKENS_MAX=200000 \
  my-batch-script.sh
```

```bash
# Dry run: print the count, don't send the prompt. Raw integer on stdout,
# human-readable estimate on stderr, exits 0 so set -e scripts are happy.
TOKENS=$(LLM_CONFIRM_TOKENS_DRY_RUN=1 llm -m gpt-4o -f big.md 'summarise')
echo "would send $TOKENS tokens"
```

`LLM_CONFIRM_TOKENS_MAX` compares the **heuristic's upper bound** by
default — the same conservative choice the confirmation prompt uses.
That means the ceiling can fire on a PDF whose true cost lands well
below the bound (see "PDF per-page rate" above). If you need a
billing-grade ceiling, pair it with exact mode:

```bash
export LLM_CONFIRM_TOKENS_EXACT=1
export LLM_CONFIRM_TOKENS_MAX=50000
```

Exact mode collapses the range to a single provider-supplied number,
so `MAX` refuses only when the provider itself says the request is
over budget.

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
