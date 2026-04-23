"""llm-confirm-tokens: interactive token-count confirmation for llm prompts.

Registers a :class:`PromptGate` via the ``register_prompt_gates`` hookspec.
When enabled (``LLM_CONFIRM_TOKENS=1``) the gate counts tokens on the
resolved prompt and, if the total is at or above the configured threshold,
prints ``7.4k input tokens (estimate). Proceed? [Y/n]:`` to ``/dev/tty`` and waits for
the user. Anything other than an empty response or ``y``/``yes`` raises
:class:`llm.CancelPrompt`, aborting the prompt before the upstream API is
called.

Opt-in rather than on-by-default so installing the plugin does not change
the behaviour of any existing script.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import llm
from llm import hookimpl

__all__ = [
    "ConfirmTokensGate",
    "count_prompt_tokens",
    "estimate_tokens",
    "estimate_tokens_detailed",
    "register_prompt_gates",
]


_TRUTHY = ("1", "true", "yes", "on")


def _is_enabled() -> bool:
    return os.environ.get("LLM_CONFIRM_TOKENS", "").strip().lower() in _TRUTHY


def _assume_yes() -> bool:
    return os.environ.get("LLM_CONFIRM_TOKENS_YES", "").strip().lower() in _TRUTHY


def _dry_run() -> bool:
    """Print the estimate and exit 0 instead of sending the prompt.

    Separate from ``LLM_CONFIRM_TOKENS`` because dry-run is a "just tell
    me the number" use case — forcing users to also set the gate-enable
    var would be friction for no reason. ``register_prompt_gates``
    treats either flag as enough to register the gate.
    """
    return os.environ.get("LLM_CONFIRM_TOKENS_DRY_RUN", "").strip().lower() in _TRUTHY


def _exact_mode() -> bool:
    """Opt-in flag for using provider count APIs instead of local heuristics."""
    return os.environ.get("LLM_CONFIRM_TOKENS_EXACT", "").strip().lower() in _TRUTHY


def _threshold() -> int:
    raw = os.environ.get("LLM_CONFIRM_TOKENS_THRESHOLD", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _max_tokens() -> int:
    """Parse ``LLM_CONFIRM_TOKENS_MAX`` — unset or 0 means no ceiling."""
    raw = os.environ.get("LLM_CONFIRM_TOKENS_MAX", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


# Fallback image cost used when we can't read the image's dimensions —
# 258 matches Gemini's flat "small image" cost and is within a round
# of the three providers' per-image baselines. When we *can* read
# dimensions (see ``_image_dimensions``) we apply the provider-specific
# formula in ``_image_tokens_for_provider`` instead; the flat number is
# only the last-resort path.
_IMAGE_TOKENS = 258
_UNKNOWN_BINARY_TOKENS = 300

# PDF per-page token estimates. Where the provider documents a
# specific number we use it verbatim; where they don't, we infer.
# Drift between the documented figure and real billing is the exact
# signal LLM_CONFIRM_TOKENS_DRIFT_WARN is designed to surface.
#
# - Gemini: 258 tokens per page per
#   https://ai.google.dev/gemini-api/docs/document-processing
#   ("each document page equals 258 tokens"). Real-world mixed-content
#   PDFs on flash-lite have been observed at ~530/page, so the
#   heuristic will flag them via drift — that's the intended design.
# - Anthropic: midpoint of the documented 1,500–3,000 text-tokens-
#   per-page range at
#   https://platform.claude.com/docs/en/docs/build-with-claude/pdf-support
#   ("Each page typically uses 1,500-3,000 tokens per page depending
#   on content density"). Image tokens are charged separately by
#   Anthropic (image formula on rendered page), but the render
#   resolution isn't documented, so we don't try to model it — the
#   drift warning catches the delta when it matters.
# - OpenAI: no explicit per-page figure in the file-inputs docs at
#   https://platform.openai.com/docs/guides/pdf-files. Keep the
#   inferred ~500/page (aligns with high-detail tile cost for a
#   letter-sized rendered page via OpenAI's standard image formula).
# Per-provider PDF per-page ranges. The low bound is what the provider
# documents for a bare text page; the high bound accounts for the
# image/tile component real PDFs routinely hit. The confirmation prompt
# shows the range directly when it's material, so users see "this might
# cost between X and Y" rather than a single number that's often wrong.
#
# - Gemini: low=258 (docs: "each document page equals 258 tokens").
#   high=1032 applies Gemini's image-tile formula to the documented
#   768×768 minimum render size (4 tiles × 258), which bounds the
#   image-containing case observed empirically (~532/page).
# - Anthropic: explicit 1,500-3,000 text-token range from docs.
#   Image tokens are charged on top at an undocumented render size
#   — the 3,000 upper bound is known to be a text-only lower-bound
#   of reality when pages contain images, so drift still fires for
#   those cases.
# - OpenAI: low=255 (single low-detail tile via 85+170); high=765
#   (high-detail letter-size: 85+170×4). No authoritative per-page
#   number; the range mirrors OpenAI's documented image-tile
#   formula for a rendered page.
_PDF_TOKENS_PER_PAGE_RANGE: dict[str, tuple[int, int]] = {
    "gemini": (258, 1032),
    "anthropic": (1500, 3000),
    "openai": (255, 765),
}


def _pdf_tokens_per_page_range(provider: str) -> tuple[int, int]:
    return _PDF_TOKENS_PER_PAGE_RANGE.get(
        provider, _PDF_TOKENS_PER_PAGE_RANGE["gemini"]
    )

# PDF /Type /Page entries: accurate for uncompressed PDFs, an under-count
# for PDFs whose object streams are compressed. The byte-size fallback in
# ``_pdf_page_count`` papers over that common case.
_PDF_PAGE_PATTERN = re.compile(rb"/Type\s*/Page(?![a-zA-Z])")


def _make_counter() -> Callable[[str], int]:
    """Return a ``text -> int`` token counter.

    Uses tiktoken's ``cl100k_base`` encoding when available (accurate for
    OpenAI and reasonable for most others); falls back to
    ``max(1, len(text) // 4)`` when tiktoken isn't installed so the plugin
    still works on any Python environment.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        # disallowed_special=() lets user strings containing tokens like
        # ``<|endoftext|>`` pass through as literal bytes instead of
        # raising ValueError — which would otherwise crash llm mid-prompt
        # the moment anyone asks the model about GPT internals.
        return lambda text: len(enc.encode(text, disallowed_special=()))
    except Exception:

        def _heuristic(text: str) -> int:
            return max(1, len(text) // 4) if text else 0

        return _heuristic


def _prompt_text(prompt: llm.Prompt) -> str:
    """Flatten the text portion of a prompt (system + body + fragments)."""
    parts: list[str] = []
    system = getattr(prompt, "system", None)
    if system:
        parts.append(system)
    for sf in getattr(prompt, "system_fragments", []) or []:
        parts.append(str(sf))
    body = getattr(prompt, "prompt", None) or getattr(prompt, "_prompt", None)
    if body:
        parts.append(body)
    for f in getattr(prompt, "fragments", []) or []:
        parts.append(str(f))
    return "\n".join(p for p in parts if p)


def _prompt_user_text(prompt: llm.Prompt) -> str:
    """Flatten a prompt's user text only — body + fragments, no system.

    Prior turns in a conversation re-send the system prompt exactly once
    (via the current turn); counting it on each historical turn would
    triple-count it on a three-turn chat. The user-side body is what the
    provider actually re-serialises turn-by-turn, so that's what we count.
    """
    parts: list[str] = []
    body = getattr(prompt, "prompt", None) or getattr(prompt, "_prompt", None)
    if body:
        parts.append(body)
    for f in getattr(prompt, "fragments", []) or []:
        parts.append(str(f))
    return "\n".join(p for p in parts if p)


def _response_output_text(response: Any) -> str:
    """Return the assistant text a prior response produced.

    Prefers ``response._chunks`` because that's what both live streaming
    and ``Response.from_row`` (the ``-c`` hydration path) populate. Falls
    back to ``response.text()`` for any exotic Response subclass that
    doesn't expose ``_chunks``. Returns ``""`` rather than raising so a
    malformed prior turn can't break gating on the new prompt.
    """
    chunks = getattr(response, "_chunks", None)
    if chunks:
        return "".join(str(c) for c in chunks)
    text_fn = getattr(response, "text", None)
    if callable(text_fn):
        try:
            return text_fn() or ""
        except Exception:
            return ""
    return ""


def _attachment_bytes(a: Any) -> bytes | None:
    """Return bytes for an attachment without triggering a network fetch.

    URL-only attachments are intentionally not resolved: the cost estimate
    is a pre-flight check, and fetching every URL just to count tokens
    would add latency (and side effects — logs, rate limits) to every
    prompt. Such attachments fall back to ``_BINARY_ATTACHMENT_TOKENS``.
    """
    content = getattr(a, "content", None)
    if content:
        return content
    path = getattr(a, "path", None)
    if path:
        try:
            return Path(path).read_bytes()
        except OSError:
            return None
    return None


def _detect_mime(a: Any, data: bytes | None) -> str:
    """Best-effort MIME detection that never performs network I/O.

    Prefers the attachment's declared ``.type`` (what the model itself will
    see), then sniffs the path (llm ships puremagic), then sniffs the
    bytes. URL-only attachments without a declared type return "".
    """
    declared = getattr(a, "type", None)
    if declared:
        return declared
    path = getattr(a, "path", None)
    if path:
        try:
            from llm.utils import mimetype_from_path

            sniffed = mimetype_from_path(path)
            if sniffed:
                return sniffed
        except Exception:
            pass
    if data:
        try:
            from llm.utils import mimetype_from_string

            sniffed = mimetype_from_string(data)
            if sniffed:
                return sniffed
        except Exception:
            pass
    return ""


def _pdf_page_count(data: bytes) -> int:
    """Estimate pages in a PDF without pulling in a PDF parser.

    Counts ``/Type /Page`` markers in the raw bytes — authoritative when
    visible, since each real page object declares itself that way. The
    regex returns 0 when the PDF packs its page objects into a
    compressed ``/ObjStm`` (typical for modern producers like pdfTeX,
    Word, LibreOffice), so we only use the 50KB-per-page byte estimate
    as a blind fallback. Using ``max(regex, bytes)`` — as an earlier
    version did — lets the byte heuristic over-ride an accurate regex
    count on content-rich PDFs (a 7MB / 24-page PDF estimated at 146
    pages), which is why the composition is ``regex or bytes``.
    """
    regex_count = len(_PDF_PAGE_PATTERN.findall(data))
    if regex_count > 0:
        return regex_count
    return max(1, len(data) // 50_000)


_TEXTY_MIMES = {"application/json", "application/xml", "application/x-yaml"}


def _image_dimensions(data: bytes) -> tuple[int, int] | None:
    """Return ``(width, height)`` for PNG/JPEG/GIF/WebP bytes, else ``None``.

    Zero-dependency format sniffing: pulls width/height straight from each
    format's fixed-offset header so we can apply provider-specific image
    token formulas without pulling in Pillow. ``None`` means "couldn't
    parse" — callers fall back to a flat per-image cost rather than
    fabricating dimensions.
    """
    if not data or len(data) < 24:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        try:
            w, h = struct.unpack(">II", data[16:24])
            return (w, h) if w > 0 and h > 0 else None
        except struct.error:
            return None
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        w, h = struct.unpack("<HH", data[6:10])
        return (w, h) if w > 0 and h > 0 else None
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8 " and len(data) >= 30:
            w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
            h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
            return (w, h) if w > 0 and h > 0 else None
        if chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
            b = data[21:25]
            w = 1 + (b[0] | ((b[1] & 0x3F) << 8))
            h = 1 + ((b[1] >> 6) | (b[2] << 2) | ((b[3] & 0x0F) << 10))
            return (w, h)
        if chunk == b"VP8X" and len(data) >= 30:
            w = 1 + int.from_bytes(data[24:27], "little")
            h = 1 + int.from_bytes(data[27:30], "little")
            return (w, h)
        return None
    if data[:2] == b"\xff\xd8":
        i = 2
        size = len(data)
        while i + 8 < size:
            if data[i] != 0xFF:
                return None
            while i < size and data[i] == 0xFF:
                i += 1
            if i >= size:
                return None
            marker = data[i]
            i += 1
            # Standalone markers carry no payload.
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                continue
            if i + 2 > size:
                return None
            seg_len = struct.unpack(">H", data[i : i + 2])[0]
            # SOF0..SOFn except DHT(C4), JPG(C8), DAC(CC): payload is
            # [len(2) precision(1) height(2) width(2) ...].
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                if seg_len < 7 or i + seg_len > size:
                    return None
                h, w = struct.unpack(">HH", data[i + 3 : i + 7])
                return (w, h) if w > 0 and h > 0 else None
            i += seg_len
        return None
    return None


def _detect_provider(model: Any) -> str:
    """Classify a model for the image-token heuristic.

    Returns ``"anthropic"``, ``"openai"``, or ``"gemini"`` (the default).
    Works off ``model_id`` string prefixes so the heuristic can pick the
    right formula even when we're only told the model name — no adapter
    key, no network. ``None`` and unrecognised models fall back to
    Gemini's formula, which is the middle-of-the-road option.
    """
    mid = (getattr(model, "model_id", "") or "").lower() if model is not None else ""
    if (
        mid.startswith("claude-")
        or mid.startswith("anthropic/")
        or mid.startswith("claude/")
    ):
        return "anthropic"
    if (
        mid.startswith("gpt-")
        or mid.startswith("chatgpt-")
        or mid.startswith("o1")
        or mid.startswith("o3")
        or mid.startswith("o4")
        or mid.startswith("openai/")
    ):
        return "openai"
    return "gemini"


# Anthropic: images are downscaled so the longest side is ≤ 1568 px, then
# charged at ~(width × height) / 750 tokens. Per
# https://docs.anthropic.com/en/docs/build-with-claude/vision.
_ANTHROPIC_MAX_DIM = 1568


def _anthropic_image_tokens(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        return _IMAGE_TOKENS
    scale = min(1.0, _ANTHROPIC_MAX_DIM / max(width, height))
    w, h = width * scale, height * scale
    return max(1, round((w * h) / 750))


# OpenAI high-detail tiling for GPT-4o-class models: scale to fit
# 2048×2048, then scale so the shortest side is 768, then count
# 512×512 tiles. Per https://platform.openai.com/docs/guides/vision.
_OPENAI_IMAGE_BASE = 85
_OPENAI_IMAGE_PER_TILE = 170


def _openai_image_tokens(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        return _IMAGE_TOKENS
    longest = max(width, height)
    w, h = float(width), float(height)
    if longest > 2048:
        s = 2048 / longest
        w, h = w * s, h * s
    shortest = min(w, h)
    if shortest > 768:
        s = 768 / shortest
        w, h = w * s, h * s
    tiles = math.ceil(w / 512) * math.ceil(h / 512)
    return _OPENAI_IMAGE_BASE + _OPENAI_IMAGE_PER_TILE * tiles


# Gemini 2.x: images with both dims ≤ 384 cost 258 tokens; larger
# images tile at tile_size = clamp(min(w, h) / 1.5, 256, 768) with
# each tile charged at 258 tokens. Per
# https://ai.google.dev/gemini-api/docs/tokens.
def _gemini_image_tokens(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        return _IMAGE_TOKENS
    if max(width, height) <= 384:
        return _IMAGE_TOKENS
    tile_size = max(256.0, min(768.0, min(width, height) / 1.5))
    tiles = math.ceil(width / tile_size) * math.ceil(height / tile_size)
    return tiles * _IMAGE_TOKENS


def _image_tokens_for_provider(provider: str, width: int, height: int) -> int:
    if provider == "anthropic":
        return _anthropic_image_tokens(width, height)
    if provider == "openai":
        return _openai_image_tokens(width, height)
    return _gemini_image_tokens(width, height)


def _looks_like_text(decoded: str) -> bool:
    """True if the decoded string is predominantly printable characters.

    UTF-8 happily decodes binary payloads that happen to use low code
    points (control chars, hex blobs), so a raw successful decode is not
    proof of "text". The 5%-control-char cap lines up with how tools like
    ``file(1)`` distinguish text from data.
    """
    if not decoded:
        return True
    control = sum(1 for c in decoded if ord(c) < 0x20 and c not in "\n\r\t")
    return (control / len(decoded)) <= 0.05


def _count_attachment_range(
    count: Callable[[str], int], a: Any, provider: str = "gemini"
) -> tuple[int, int]:
    """Return ``(low, high)`` token cost for one attachment.

    A collapsed tuple (``low == high``) means we have an exact value;
    a widened tuple means the attachment type has inherent uncertainty
    (PDFs hit per-page ranges depending on embedded image content,
    image attachments we can't measure fall back to a provider-range
    envelope). The prompt-level counter sums these and collapses the
    result for display.
    """
    data = _attachment_bytes(a)
    mime = _detect_mime(a, data)
    if mime.startswith("image/"):
        if data is not None:
            dims = _image_dimensions(data)
            if dims is not None:
                exact = _image_tokens_for_provider(provider, *dims)
                return (exact, exact)
        # No dims parsed → fall back to a flat per-image estimate. Keep
        # it single-valued rather than provider-specific range to avoid
        # widening every image attachment; the drift warning covers the
        # miss if a provider bills radically differently.
        return (_IMAGE_TOKENS, _IMAGE_TOKENS)
    if mime == "application/pdf":
        low, high = _pdf_tokens_per_page_range(provider)
        pages = _pdf_page_count(data) if data is not None else 1
        return (pages * low, pages * high)
    if data is None:
        return (_UNKNOWN_BINARY_TOKENS, _UNKNOWN_BINARY_TOKENS)
    texty_by_mime = mime.startswith("text/") or mime in _TEXTY_MIMES
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError:
        return (_UNKNOWN_BINARY_TOKENS, _UNKNOWN_BINARY_TOKENS)
    if texty_by_mime or _looks_like_text(decoded):
        n = count(decoded)
        return (n, n)
    return (_UNKNOWN_BINARY_TOKENS, _UNKNOWN_BINARY_TOKENS)


def _count_prior_turns(
    count: Callable[[str], int], conversation: Any, provider: str
) -> tuple[int, int]:
    """Sum tokens for every prior user+assistant exchange in ``conversation``.

    For each prior response we fold in:
    - the user body (``prompt`` + fragments) — no system, since the
      current turn already carries the system envelope the provider
      will actually re-send.
    - the user's attachments, which most providers re-encode on every
      turn (images are re-uploaded as inline_data / base64 each time).
    - the assistant's output text.

    Tools and tool results from prior turns are deliberately skipped
    here — matching the exact-mode adapters, which avoid them because
    representing tool use faithfully requires alternating assistant/
    user invariants the provider count endpoints validate. The overall
    under-count is small in practice and the drift-warning path would
    surface it if a workload depended on it.
    """
    low = high = 0
    for prior in getattr(conversation, "responses", None) or []:
        prior_prompt = getattr(prior, "prompt", None)
        if prior_prompt is not None:
            user_text = _prompt_user_text(prior_prompt)
            if user_text:
                n = count(user_text)
                low += n
                high += n
            for a in getattr(prior_prompt, "attachments", []) or []:
                a_low, a_high = _count_attachment_range(count, a, provider)
                low += a_low
                high += a_high
        output_text = _response_output_text(prior)
        if output_text:
            n = count(output_text)
            low += n
            high += n
    return (low, high)


def count_prompt_tokens_range(
    prompt: llm.Prompt, model: Any = None, conversation: Any = None
) -> tuple[int, int]:
    """Return ``(low, high)`` token cost for ``prompt``.

    ``low == high`` means the count has no attachment-driven
    uncertainty (text-only prompts, parseable images, etc.); a widened
    range means at least one attachment has inherent uncertainty (PDFs
    always; image-dim parse failures would too if we widened them).
    Callers that want a single number can use :func:`count_prompt_tokens`
    (returns the ``high`` bound — the conservative choice for gating).

    When ``conversation`` is provided (e.g. ``llm -c``), the token
    cost of every prior turn's user body, attachments and assistant
    output is folded in — because that's what the model plugin will
    actually re-send to the provider alongside the new prompt. Without
    this the estimate silently under-counts a continued chat to the
    size of just the latest message.
    """
    count = _make_counter()
    provider = _detect_provider(model)
    text_tokens = count(_prompt_text(prompt))
    low = high = text_tokens
    for a in getattr(prompt, "attachments", []) or []:
        a_low, a_high = _count_attachment_range(count, a, provider)
        low += a_low
        high += a_high
    tools = getattr(prompt, "tools", []) or []
    if tools:
        tools_payload = [
            {
                "name": getattr(t, "name", None),
                "description": getattr(t, "description", None),
                "input_schema": getattr(t, "input_schema", None),
            }
            for t in tools
        ]
        n = count(json.dumps(tools_payload, default=str))
        low += n
        high += n
    for tr in getattr(prompt, "tool_results", []) or []:
        out = getattr(tr, "output", None)
        if out:
            n = count(str(out))
            low += n
            high += n
    schema = getattr(prompt, "schema", None)
    if schema is not None:
        n = count(json.dumps(schema, default=str))
        low += n
        high += n
    if conversation is not None:
        prior_low, prior_high = _count_prior_turns(count, conversation, provider)
        low += prior_low
        high += prior_high
    return (low, high)


def count_prompt_tokens(
    prompt: llm.Prompt, model: Any = None, conversation: Any = None
) -> int:
    """Estimate the full token cost of ``prompt`` before it is sent.

    Returns the ``high`` bound from :func:`count_prompt_tokens_range`
    — the conservative choice for gating, since the gate's "did I
    really mean to send this much?" question is better answered
    pessimistically. Callers that want both ends of the band should
    call :func:`count_prompt_tokens_range` directly.
    """
    return count_prompt_tokens_range(prompt, model, conversation)[1]


_DRIFT_STASH_ATTR = "_confirm_tokens_heuristic"
_DRIFT_MODEL_STASH_ATTR = "_confirm_tokens_model_id"


def estimate_tokens_detailed(
    prompt: llm.Prompt, model: Any = None, conversation: Any = None
) -> tuple[int, int, str]:
    """Return ``(low, high, source)`` for a pre-flight estimate of ``prompt``.

    ``low == high`` means either exact mode produced a definitive count
    or the prompt has no attachment-driven uncertainty (text, parseable
    images, tools). Heuristic mode widens the range for PDFs, where
    documented per-page rates and real billing diverge materially.

    Source is ``"heuristic"`` or the provider name when an exact-count
    adapter produced the figure. Adapter failures fall through to the
    local heuristic, but are *not* silent — a one-line notice goes to
    stderr so degradation is visible.

    ``conversation`` — when set (e.g. ``llm -c``) — contributes the cost
    of every prior turn's user body, attachments and assistant output
    both to the heuristic range and to exact-mode adapters. Without it
    a continued chat would silently under-count to the size of only the
    newest message.

    The heuristic range is always computed and stashed on the prompt so
    the ``after_log_to_db`` hook can compare it against the billed count
    once the response arrives. That makes drift detection work even in
    heuristic-only mode — the mode where it matters most, because that's
    where users have no ground truth to sanity-check against locally.
    """
    low, high = count_prompt_tokens_range(prompt, model, conversation)
    _stash_heuristic(prompt, low, high, model)

    if model is not None and _exact_mode():
        from . import _adapters

        for adapter in _adapters.iter_adapters():
            if not adapter.matches(model):
                continue
            name = type(adapter).__name__.removesuffix("Adapter").lower() or "adapter"
            try:
                exact = adapter.count(prompt, model, conversation=conversation)
                _maybe_warn_drift(exact, low, high, name)
                return (exact, exact, name)
            except Exception as exc:
                sys.stderr.write(
                    f"llm-confirm-tokens: {name} exact-count failed "
                    f"({type(exc).__name__}: {exc}); using heuristic.\n"
                )
                break
    return (low, high, "heuristic")


def _drift_threshold_pct() -> float | None:
    """Parse ``LLM_CONFIRM_TOKENS_DRIFT_WARN`` as a percentage, else None."""
    raw = os.environ.get("LLM_CONFIRM_TOKENS_DRIFT_WARN", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _stash_heuristic(
    prompt: llm.Prompt, low: int, high: int, model: Any
) -> None:
    """Record the pre-flight heuristic range on the prompt for drift compare.

    Stores the full ``(low, high)`` so ``after_log_to_db`` can tell
    whether the billed number landed inside the estimate's band (no
    drift) or outside it (worth warning about). Wrapped in try/except
    so a frozen or slotted Prompt subclass can't break gating.
    """
    try:
        setattr(prompt, _DRIFT_STASH_ATTR, (low, high))
        setattr(prompt, _DRIFT_MODEL_STASH_ATTR, getattr(model, "model_id", None))
    except Exception:
        pass


def _maybe_warn_drift(
    actual: int, low: int, high: int, source: str
) -> None:
    """Warn to stderr only when the heuristic ``under``-counted by ≥ threshold.

    Bill shock is the failure mode we care about: a prompt the user
    thought was cheap turning out to be expensive. Over-counts (estimate
    exceeded reality) are the "pleasantly surprised" case — not worth
    interrupting with a stderr warning. So this fires only when
    ``actual`` exceeds the estimated upper bound by the configured
    percentage. ``source`` names the ground-truth channel
    (``"gemini"`` / ``"anthropic"`` / ``"openai"`` for the pre-flight
    exact-count path, or ``"<model_id> billed"`` for the post-response
    path).
    """
    threshold = _drift_threshold_pct()
    if threshold is None or actual <= 0:
        return
    # Actual inside the range or below it → no bill-shock risk, stay quiet.
    if actual <= high:
        return
    delta_pct = (actual - high) / actual * 100
    if delta_pct < threshold:
        return
    heuristic_str = f"{high:,}" if low == high else f"{low:,}–{high:,}"
    sys.stderr.write(
        f"llm-confirm-tokens: heuristic {heuristic_str} under-counts "
        f"vs {source} {actual:,} by {delta_pct:.0f}% — local estimates "
        f"are a best guess, not billing-grade.\n"
    )


def estimate_tokens(
    prompt: llm.Prompt, model: Any = None, conversation: Any = None
) -> int:
    """Return a pre-flight token estimate for ``prompt`` (upper bound).

    Thin wrapper over :func:`estimate_tokens_detailed` for callers that
    only want a single number — returns the high bound of the range,
    matching :func:`count_prompt_tokens`.
    """
    _, high, _ = estimate_tokens_detailed(prompt, model, conversation)
    return high


def _humanize_estimate(n: int) -> str:
    """Round ``n`` to a scannable order-of-magnitude string.

    Estimates shouldn't show billing-grade precision — "7.4k tokens" is
    both easier to scan and more honest than "7,391" when the underlying
    number carries ±20% heuristic error. Rules:
    - < 100: exact (small numbers are already readable)
    - 100–999: nearest 10
    - 1,000–9,999: one-decimal k (e.g. "1.6k", "7.4k")
    - 10,000–999,999: whole-number k (e.g. "12k", "258k")
    - 1,000,000+: one-decimal M
    """
    if n < 100:
        return str(n)
    if n < 1000:
        return f"{round(n, -1):,}"
    if n < 10_000:
        return f"{n / 1000:.1f}k"
    if n < 1_000_000:
        return f"{round(n / 1000):,}k"
    return f"{n / 1_000_000:.1f}M"


def _format_total(low: int, high: int, source: str) -> str:
    """Format the number(s) and source tag for the confirmation prompt.

    Layout is always ``{number-or-range} input tokens ({source})`` — the
    number-of-tokens is first (fastest to scan), the unit makes the count
    self-labelling, and the source in parens names where the count came
    from ("estimate" for the local heuristic; provider name for exact
    counts). Heuristic numbers are humanised (``7.4k``) because the
    underlying value is already fuzzy; exact counts stay at full
    precision because they're what the provider will actually bill.
    """
    if source == "heuristic":
        label = "estimate"
        number = (
            _humanize_estimate(high)
            if low == high
            else f"{_humanize_estimate(low)}–{_humanize_estimate(high)}"
        )
    else:
        label = source
        number = f"{high:,}" if low == high else f"{low:,}–{high:,}"
    return f"{number} input tokens ({label})"


def _emit_dry_run(low: int, high: int, source: str) -> None:
    """Print the estimate and exit 0 — never returns.

    Splits the output across streams so scripts can consume either:
    the human-readable line (with source + range when applicable) goes
    to stderr, and a single integer — the conservative ``high`` bound,
    matching :func:`count_prompt_tokens` — goes to stdout. That lets
    ``TOKENS=$(LLM_CONFIRM_TOKENS_DRY_RUN=1 llm -m gpt-4o 'hi')`` work
    cleanly.

    Exits 0 rather than raising ``llm.CancelPrompt`` because a dry-run
    is a successful operation: the user asked for a number and got one.
    ``CancelPrompt`` would print a "canceled" notice and exit non-zero,
    which ``set -e`` scripts would treat as a failure.
    """
    sys.stderr.write(f"{_format_total(low, high, source)}\n")
    sys.stderr.flush()
    sys.stdout.write(f"{high}\n")
    sys.stdout.flush()
    sys.exit(0)


def _ask_via_tty(
    low: int, high: int | None = None, source: str = "heuristic"
) -> bool:
    """Prompt the user for a proceed/cancel decision and return True to proceed.

    ``high`` defaults to ``low`` so legacy callers that pass a single
    number still work (``_ask_via_tty(1234, source="heuristic")``).

    Shows a single number when the estimate has no attachment-driven
    uncertainty (``low == high``) and a ``low-high`` range otherwise —
    letting users see where our confidence is wider than a point estimate
    implies. The ``~`` prefix still signals "heuristic, not billing-grade"
    regardless of whether a range is displayed.

    Prefers ``sys.stdin`` + ``sys.stderr`` when both are interactive —
    opening the console via standard streams avoids the ``r+`` seek
    failure that bit us on macOS (``UnsupportedOperation: File or stream
    is not seekable`` when calling ``open('/dev/tty', 'r+')``). When
    streams have been redirected (e.g. ``cat big.txt | llm …``), falls
    back to opening the platform's console device separately for read
    and write so the seek issue can't recur.

    Fails **closed** when no interactive terminal is available anywhere
    — if the plugin is enabled but the user can't be asked, declining
    the prompt is safer than silently approving a large payload. Batch
    scripts that want auto-approval should set
    ``LLM_CONFIRM_TOKENS_YES=1``, which bypasses this function entirely.
    """
    if high is None:
        high = low
    total = _format_total(low, high, source)
    message = f"{total}. Proceed? [Y/n]: "

    # Path 1: sys.stdin + sys.stderr when both are interactive. No special
    # file opens, no seek, and it Just Works in the majority of cases.
    if sys.stdin.isatty() and sys.stderr.isatty():
        try:
            sys.stderr.write(message)
            sys.stderr.flush()
            answer = sys.stdin.readline()
            if not answer:
                return False  # EOF — treat as decline
            return answer.strip().lower() in ("", "y", "yes")
        except OSError:
            pass

    # Path 2: streams redirected (typical `cat big.txt | llm …`). Open the
    # controlling terminal directly, using *separate* read and write file
    # handles — opening once with ``r+`` triggers an implicit seek, which
    # /dev/tty does not support on macOS.
    tty_pair = (
        ("CONOUT$", "CONIN$")
        if sys.platform == "win32"
        else ("/dev/tty", "/dev/tty")
    )
    out_path, in_path = tty_pair
    try:
        with open(out_path, "w") as tty_out, open(in_path) as tty_in:
            tty_out.write(message)
            tty_out.flush()
            answer = tty_in.readline()
        if not answer:
            return False  # EOF
        return answer.strip().lower() in ("", "y", "yes")
    except OSError:
        pass

    # No interactive terminal available anywhere — tell the user why we're
    # declining so they can fix the environment or set YES=1 explicitly.
    sys.stderr.write(
        f"llm-confirm-tokens: {total}, but no tty available "
        "to confirm. Set LLM_CONFIRM_TOKENS_YES=1 to auto-approve in "
        "non-interactive environments.\n"
    )
    return False


class ConfirmTokensGate:
    """Prompt gate that confirms with the user before large prompts are sent.

    Inject ``tokens_fn`` and ``ask`` for tests — in production the defaults
    (exact counts for providers that support them, heuristic otherwise, and
    platform-appropriate tty prompting) are used.

    ``tokens_fn`` may return ``int`` (legacy — treated as ``low == high``),
    ``(int, str)`` legacy ``(tokens, source)``, ``(low, high)``, or
    ``(low, high, source)``. ``ask`` may accept ``(low, high, source)``
    (preferred), ``(tokens, source)``, or ``(tokens,)`` — we try each
    shape in order so older test code keeps working.
    """

    def __init__(
        self,
        *,
        threshold: int = 0,
        max_tokens: int = 0,
        tokens_fn: Callable[..., int] | None = None,
        ask: Callable[..., bool] | None = None,
    ) -> None:
        self.threshold = threshold
        self.max_tokens = max_tokens
        self._tokens_fn = tokens_fn
        self._ask = ask or _ask_via_tty

    def _count(
        self, prompt: llm.Prompt, model: Any, conversation: Any
    ) -> tuple[int, int, str]:
        """Return ``(low, high, source)``.

        When no ``tokens_fn`` was injected, we call
        :func:`estimate_tokens_detailed` directly so the true source
        ("heuristic", "anthropic", "gemini", "openai") and the real
        range (wider on PDFs, collapsed elsewhere) both flow through
        to the confirmation message.

        Custom ``tokens_fn`` callables are tried in order of richest to
        sparsest signature: ``(prompt, model, conversation)`` →
        ``(prompt, model)`` → ``(prompt,)``. That lets older injected
        counters keep working while new ones can opt in to seeing
        history.
        """
        if self._tokens_fn is None:
            return estimate_tokens_detailed(prompt, model, conversation)
        for args in (
            (prompt, model, conversation),
            (prompt, model),
            (prompt,),
        ):
            try:
                result = self._tokens_fn(*args)
                break
            except TypeError:
                continue
        else:
            result = self._tokens_fn(prompt)
        return _normalise_tokens_result(result)

    def _invoke_ask(self, low: int, high: int, source: str) -> bool:
        for args in ((low, high, source), (high, source), (high,)):
            try:
                return self._ask(*args)
            except TypeError:
                continue
        return self._ask(high)

    def check(
        self,
        prompt: llm.Prompt,
        model: Any,
        conversation: Any = None,
    ) -> None:
        low, high, source = self._count(prompt, model, conversation)
        # Dry-run short-circuits everything else — threshold, max, and
        # yes are gating concerns, but the user only asked for the number.
        # Exits 0 so scripts can consume stdout without set -e pain.
        if _dry_run():
            _emit_dry_run(low, high, source)
        # Gate conservatively on the high bound — the "did I really
        # mean to send this much?" question is better answered
        # pessimistically when the estimate has a width.
        if self.max_tokens and high >= self.max_tokens:
            # Hard ceiling trumps YES so scripts that auto-approve can
            # still refuse runaway payloads.
            total = _format_total(low, high, source)
            raise llm.CancelPrompt(
                f"{total} exceeds LLM_CONFIRM_TOKENS_MAX={self.max_tokens:,}"
            )
        if high < self.threshold:
            return
        if _assume_yes():
            return
        if not self._invoke_ask(low, high, source):
            total = _format_total(low, high, source)
            raise llm.CancelPrompt(f"user declined {total}")


def _normalise_tokens_result(result: Any) -> tuple[int, int, str]:
    """Coerce any supported ``tokens_fn`` return shape into ``(low, high, source)``.

    Accepts ``int``, ``(low, high)``, ``(tokens, source)``, and
    ``(low, high, source)`` so pre-range-API tests still work. Invalid
    shapes fall back to treating the first element as both low and high
    under the "heuristic" label rather than raising — robustness matters
    more than strictness in a gate path.
    """
    if isinstance(result, tuple):
        if len(result) == 3:
            low, high, source = result
            return (int(low), int(high), str(source))
        if len(result) == 2:
            a, b = result
            if isinstance(b, str):
                n = int(a)
                return (n, n, b)
            return (int(a), int(b), "heuristic")
    n = int(result)
    return (n, n, "heuristic")


@hookimpl
def register_prompt_gates(register: Any) -> None:
    """Register the confirm-tokens gate when the plugin is enabled.

    Keyed on the ``LLM_CONFIRM_TOKENS`` env var so the plugin is a no-op
    for users who have installed it but not opted in.
    ``LLM_CONFIRM_TOKENS_DRY_RUN=1`` also registers the gate on its own
    so users can count-without-sending without having to also flip the
    confirmation gate on.
    """
    if not (_is_enabled() or _dry_run()):
        return
    register(ConfirmTokensGate(threshold=_threshold(), max_tokens=_max_tokens()))


@hookimpl
def after_log_to_db(response: Any, db: Any) -> None:
    """Compare the stashed heuristic against the billed token count.

    Fires after the real response has been logged, so ``response.input_tokens``
    is the authoritative bill. Only warns when the user has opted in via
    ``LLM_CONFIRM_TOKENS_DRIFT_WARN`` *and* the gate actually ran (so a
    heuristic was stashed). Critical for heuristic-only users: without
    this, they'd never know their local formula is off for their model.
    """
    threshold = _drift_threshold_pct()
    if threshold is None:
        return
    prompt = getattr(response, "prompt", None)
    if prompt is None:
        return
    stash = getattr(prompt, _DRIFT_STASH_ATTR, None)
    if stash is None:
        return
    if isinstance(stash, tuple):
        low, high = stash
    else:
        # Pre-range API stash was a bare int — treat as a collapsed range.
        low = high = int(stash)
    actual = getattr(response, "input_tokens", None)
    if actual is None or actual <= 0:
        return
    model_id = getattr(prompt, _DRIFT_MODEL_STASH_ATTR, None) or "model"
    _maybe_warn_drift(int(actual), int(low), int(high), f"{model_id} billed")
