# ADR 0001: Model-owned token counting

Status: **Draft** — not yet accepted; seeking feedback from `llm` core.

## Context

`llm-confirm-tokens` counts tokens before a prompt is sent. It runs via
the `register_prompt_gates` hookspec on the
[`llm-prompt-gates-hook`](https://github.com/olivergeorge/llm/tree/llm-prompt-gates-hook)
fork, receiving `(prompt, model, conversation)` from core.

To produce an accurate count the plugin must know what the model will
actually send. Today the plugin does that *twice*:

1. **Heuristic path** — flattens the prompt and attachments itself,
   applies a per-provider image/PDF formula, and tokenises locally with
   tiktoken. Updated in this release to fold `conversation.responses`
   into the estimate so `llm -c` stops under-counting.

2. **Exact-mode adapters** (`llm_confirm_tokens/_adapters.py`) — one
   per provider (Anthropic, Gemini, OpenAI). Each adapter rebuilds the
   provider's wire format from the `Prompt`: attachments become
   `inline_data` / base64 / `input_file`, tool schemas get re-serialised,
   prior turns get re-threaded as alternating user/assistant messages,
   and the result is handed to the provider's free `count_tokens`
   endpoint.

This duplicates work the model plugins already do. OpenAI, Anthropic,
and Gemini each have a `build_messages(prompt, conversation)` +
`build_kwargs(...)` / `build_request_body(...)` pair inside their
`Model.execute` implementation. Our adapters are a second, slightly
different implementation of the same translation — one that quietly
drifts from the real one whenever a model plugin adds a new attachment
type, changes how tools are shaped, or updates how it threads
conversation history.

Each time a provider changes its wire format (new attachment types,
new role names, new input-part shapes) we have to follow — and silently
under-count until we do.

## Problem statement

Payload construction lives inside the model plugin, but the code that
needs to *count* that payload lives outside it. We need a seam that
lets a gate ask the model "what will you send?" without re-implementing
the answer.

## Options considered

### Option A — Duck-type `build_messages` / `build_kwargs` on `Model`

The adapter tries `model.build_messages(prompt, conversation)` and the
matching kwargs builder, uses what it gets, falls back to the current
hand-rolled adapter otherwise.

- **Pro**: zero change to `llm` core; uses what already exists in the
  three main plugins.
- **Pro**: works today against `llm-anthropic`, `llm-gemini`, and
  core's OpenAI plugin without coordination.
- **Con**: implicit contract — signatures already differ
  (`build_kwargs(prompt, stream)` in OpenAI vs
  `build_kwargs(prompt, conversation)` in Anthropic). Third-party
  plugins may not implement it at all, or implement it with a different
  signature.
- **Con**: no way to know *which* kwargs are safe to pass to
  `count_tokens` vs which are execution-time only (stream settings,
  options, keys). The adapter still needs per-provider knowledge to
  strip those.
- **Con**: encodes an undocumented API. Any time a model plugin
  renames an internal method, this breaks silently — counts drift and
  users don't notice until the drift-warn threshold fires.

### Option B — Formalise an optional `TokenCountable` protocol on `Model`

Add an optional method to `llm.Model`:

```python
class Model:
    def count_tokens(
        self, prompt: Prompt, conversation: Conversation | None = None
    ) -> int:
        """Return the provider's exact input-token count for this prompt.
        Raises NotImplementedError if the model has no count endpoint."""
        raise NotImplementedError
```

Model plugins that have a free preflight count (Anthropic, Gemini,
OpenAI) implement it themselves — they already know their wire format,
their SDK, their key resolution. `llm-confirm-tokens` becomes a thin
caller:

```python
try:
    tokens = model.count_tokens(prompt, conversation)
except NotImplementedError:
    tokens = heuristic_count(prompt, conversation, model)
```

- **Pro**: each plugin owns its own competence — matches how `llm` is
  already designed around `Model.execute`, `Model.get_key`,
  `Model.model_id`.
- **Pro**: no adapters to maintain in this plugin.
- **Pro**: other plugins get the same affordance for free — a cost
  estimator, a request-size limiter, an audit logger, anything that
  needs a pre-send token count can call the same method.
- **Pro**: model plugins can decide for themselves whether their count
  is billing-grade or approximate, and surface that via a
  `count_is_exact` property or a structured return type (e.g.
  `TokenCount(count=7412, source="anthropic")`).
- **Con**: core addition — needs `llm` core to land the optional
  method and documentation.
- **Con**: rollout is slow — until model plugins implement the method,
  every model looks "unsupported" and falls back to the heuristic.
  Same end-user experience as today's "provider not in exact-mode
  adapter list" state, so not a regression.

### Option C — `before_send` post-payload hook

Add a new hookspec that fires *after* a model plugin has built its
provider payload but before the HTTP send:

```python
@hookspec
def before_send(prompt, model, conversation, payload):
    "Inspect or veto the finalised provider payload"
```

Requires model plugins to call `pm.hook.before_send(...)` inside
`execute`, or for `llm` core to standardise payload construction as a
separable `Model.build_payload()` step so core can call the hook.

- **Pro**: the correct architectural seam. Gates see exactly what's
  about to go out, not a reconstruction.
- **Pro**: unlocks other plugins too — logging, redaction, cost
  accounting, request shaping.
- **Con**: biggest change. Requires every model plugin to cooperate
  (or a core refactor that moves payload construction out of
  `execute`).
- **Con**: the payload is provider-shaped, so the gate still needs
  provider knowledge to tokenise it — unless the hook runs *after*
  the send but before the stream starts, which changes failure modes.

## Decision (proposed)

**Recommend Option B — `Model.count_tokens` as an optional method.**

It matches `llm`'s "each plugin owns its competence" shape, requires
the smallest core addition, and removes the main source of drift
between this plugin and the providers it counts for. The model plugins
already call `count_tokens` internally in some cases
(e.g. usage reporting); formalising a public method is a small step.

**Transition plan:**

1. Land `Model.count_tokens` as an optional method in `llm` core
   (raises `NotImplementedError` by default). Document in
   `plugin-hooks.md` alongside `execute`.
2. `llm-confirm-tokens` adds a thin "ask the model first" path before
   its adapters. If `model.count_tokens(...)` returns a number, use it;
   otherwise fall through to the existing adapters, then to the
   heuristic.
3. Once upstream model plugins (`llm-anthropic`, `llm-gemini`, core's
   OpenAI plugin) implement `count_tokens`, retire the adapters in
   this plugin. The heuristic stays as the fallback for third-party
   plugins that don't implement it.

Until step 3 the plugin ships both paths side by side so nothing
regresses.

## Consequences

- The plugin stops being a source of truth for provider wire formats.
  Fewer updates needed here when providers change encoding rules.
- Third-party plugins get a clear place to add exact counting if they
  want it — no need to patch `llm-confirm-tokens`.
- The `LLM_CONFIRM_TOKENS_DRIFT_WARN` path becomes more meaningful: it
  only fires on heuristic-vs-billed drift, because exact mode is now
  (by construction) the provider's own number.
- Adapters in `_adapters.py` become vestigial. They can stay as a
  transitional fallback until the main provider plugins adopt
  `count_tokens`, then be deleted.

## Open questions

- **Return type.** `int` is simplest; `TokenCount(value, source,
  breakdown)` is more useful (lets the gate show "1,232 input tokens
  + 18 image tokens" and lets drift detection distinguish sources).
  Leaning toward a structured return, since adding fields later to an
  `int` is impossible.
- **Cost.** Should `count_tokens` be free/local-only, or is a network
  call acceptable? The existing adapters already make one round-trip
  (~100-300ms). Keeping that latency contract explicit in the docstring
  is probably enough.
- **Async.** Mirror `Model.execute` — provide `Model.acount_tokens`
  for async models, defaulting to running the sync version in a
  threadpool.
- **Conversation shape.** Pass `Conversation` or a pre-built message
  list? Conversation keeps the model plugin in charge of how it
  replays history; a message list lets callers count hypothetical
  conversations that aren't yet persisted. Probably Conversation for
  parity with `execute`, with a helper for the message-list case.
