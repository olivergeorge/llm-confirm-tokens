# llm-confirm-tokens: a pre-flight "you're about to send N tokens, proceed?" gate for `llm`

I've been piping `files-to-prompt` output into `llm` and occasionally discovering — after the fact, in the next day's spend report — that I'd accidentally stuffed a 50k-token repo dump into a Claude Opus call. So I wrote **llm-confirm-tokens**: a gate that counts tokens on the resolved request and asks for confirmation before anything leaves the machine.

https://github.com/olivergeorge/llm-confirm-tokens

**Upfront: this is a proof-of-concept.** It's had minimal testing beyond my own workflow, depends on two unmerged hookspecs in a fork of `llm` (see below), and is tagged `0.1a0` for a reason. I'm posting it to get feedback on the shape — particularly the "estimate vs. exact" split and the pre-flight drift warning — rather than because it's ready to depend on.

## What it looks like

```
$ files-to-prompt src/ | llm 'explain the request pipeline'
7.4k input tokens (estimate). Proceed? [Y/n]:
```

Anything other than `Enter` / `y` / `yes` raises `llm.CancelPrompt` — the upstream API is **not** called, the conversation is not updated, and `llm` exits non-zero. Confirmation reads from `/dev/tty`, so piping stdin (`cat big.txt | llm …`) still works. No TTY available → **fail closed**: scripts that want to run unattended set `LLM_CONFIRM_TOKENS_YES=1`.

Opt-in per shell:

```bash
export LLM_CONFIRM_TOKENS=1
export LLM_CONFIRM_TOKENS_THRESHOLD=1000   # only ask above N tokens
```

## Estimate vs. exact

The confirmation line always names its source, so you can tell at a glance what you're looking at:

```
7.4k input tokens (estimate). Proceed? [Y/n]:       # local heuristic (tiktoken cl100k_base)
7,412 input tokens (anthropic). Proceed? [Y/n]:     # provider count_tokens API
```

Exact mode is opt-in — `LLM_CONFIRM_TOKENS_EXACT=1` plus the extra for your provider:

```bash
llm install 'llm-confirm-tokens[anthropic]'   # /v1/messages/count_tokens
llm install 'llm-confirm-tokens[gemini]'      # Client.models.count_tokens
llm install 'llm-confirm-tokens[openai]'      # /v1/responses/input_tokens
```

One extra round-trip (~100–300 ms) per gated prompt, no billing. Any failure path (missing key, network flake, SDK too old, non-matching model) silently falls back to the heuristic with a stderr notice — exact mode is a refinement, never a gate on gating.

## What it counts

Everything in the resolved prompt, locally and without network calls: `prompt`, `system`, fragments (via tiktoken), text attachments, tool definitions, structured-output schemas, and prior conversation turns. The `conversation` kwarg core now passes to gates is what lets `llm -c` turn five include the cost of turns one through four — the same bytes the model plugin is about to re-send.

Images use the provider's own documented tile formula (Gemini 258/tile, Claude `w×h/750`, OpenAI `85 + 170 × ceil(w/512) × ceil(h/512)`), parsed straight from the image header without Pillow. PDFs are the messy case: the documented rates are per-page *ranges*, not single numbers, because image-heavy pages blow past the text-only rate. The plugin shows the range directly rather than pretending to know better:

```
3.1k–12k input tokens (estimate). Proceed? [Y/n]:
```

Text-only prompts collapse to a single number (`low == high`), so the range only appears when there's real uncertainty.

## The drift warning

If the heuristic drifts on your workload, you want to know — especially after a provider rolls out new image or PDF pricing. `LLM_CONFIRM_TOKENS_DRIFT_WARN=25` compares the pre-flight heuristic to the actual billed `input_tokens` from the response, and prints a one-line stderr notice **only when the estimate under-counted** (over-counts are the "pleasantly surprised" case and stay silent):

```
llm-confirm-tokens: heuristic 258 under-counts vs gemini-flash-latest
billed 1,080 by 76% — local estimates are a best guess, not billing-grade.
```

Detection works in both modes — in heuristic-only it's your only signal; in exact mode it acts as a sanity check on the adapter itself.

## The honest caveat

This plugin depends on **two hookspecs not yet in upstream `llm`** — `register_prompt_gates` (interceptive, so the gate runs before `Model.execute`) and `after_log_to_db` (observational, to drive the drift warning under the user's existing `--log` policy). Both live on branches of [my `llm` fork](https://github.com/olivergeorge/llm); a `combined-prs` branch carries both plus the one [llm-replay](https://github.com/olivergeorge/llm-replay) needs. The README has the one-liner install.

Without `register_prompt_gates` the plugin would have to monkey-patch `_BaseResponse` or subclass every `Model` — fragile across versions and hostile to other plugins. Without `after_log_to_db` the drift warning degrades to exact-mode-only.

The other thing I'd value feedback on is in [ADR 0001](https://github.com/olivergeorge/llm-confirm-tokens/blob/main/docs/adr/0001-model-owned-token-counting.md): the per-provider "exact count" adapters are duplicating what each model plugin already does inside `Model.execute` — rebuilding the wire payload every time a provider changes attachment shapes or role names. The ADR proposes moving counting into the model plugins themselves as `Model.count_tokens(prompt, conversation)`, with the heuristic as the fallback. Status: draft — would love a second opinion before investing further.

Single-author PoC, only exercised against my own workflow. If your provider lineup, SDK versions, or attachment shapes look different from mine, the adapter layer is where it's most likely to trip, and I'd value bug reports.
