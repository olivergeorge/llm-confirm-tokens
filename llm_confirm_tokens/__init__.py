"""llm-confirm-tokens: interactive token-count confirmation for llm prompts.

Registers a :class:`PromptGate` via the ``register_prompt_gates`` hookspec.
When enabled (``LLM_CONFIRM_TOKENS=1``) the gate counts tokens on the
resolved prompt and, if the total is at or above the configured threshold,
prints ``Total tokens: N. Proceed? [Y/n]:`` to ``/dev/tty`` and waits for
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


def _exact_mode() -> bool:
    """Opt-in flag for using provider count APIs instead of local heuristics."""
    return os.environ.get("LLM_CONFIRM_TOKENS_EXACT", "").strip().lower() in _TRUTHY


def _threshold() -> int:
    raw = os.environ.get("LLM_CONFIRM_TOKENS_THRESHOLD", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


# Fallback image cost used when we can't read the image's dimensions —
# 258 matches Gemini's flat "small image" cost and is within a round
# of the three providers' per-image baselines. When we *can* read
# dimensions (see ``_image_dimensions``) we apply the provider-specific
# formula in ``_image_tokens_for_provider`` instead; the flat number is
# only the last-resort path. PDF pages still use 258 as Gemini does.
_IMAGE_TOKENS = 258
_PDF_TOKENS_PER_PAGE = 258
_UNKNOWN_BINARY_TOKENS = 300

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

    Counts ``/Type /Page`` markers in the raw bytes. Compressed PDFs hide
    their page objects inside compressed streams, so the regex returns 0
    on those — ``max`` with a bytes-per-page fallback (50KB/page, a
    middle-of-the-road value for mixed content) keeps the estimate honest
    on modern PDFs.
    """
    regex_count = len(_PDF_PAGE_PATTERN.findall(data))
    byte_estimate = max(1, len(data) // 50_000)
    return max(regex_count, byte_estimate)


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


def _count_attachment_tokens(
    count: Callable[[str], int], a: Any, provider: str = "gemini"
) -> int:
    data = _attachment_bytes(a)
    mime = _detect_mime(a, data)
    if mime.startswith("image/"):
        if data is not None:
            dims = _image_dimensions(data)
            if dims is not None:
                return _image_tokens_for_provider(provider, *dims)
        return _IMAGE_TOKENS
    if mime == "application/pdf":
        if data is None:
            return _PDF_TOKENS_PER_PAGE
        return _pdf_page_count(data) * _PDF_TOKENS_PER_PAGE
    if data is None:
        return _UNKNOWN_BINARY_TOKENS
    texty_by_mime = mime.startswith("text/") or mime in _TEXTY_MIMES
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError:
        return _UNKNOWN_BINARY_TOKENS
    if texty_by_mime or _looks_like_text(decoded):
        return count(decoded)
    return _UNKNOWN_BINARY_TOKENS


def count_prompt_tokens(prompt: llm.Prompt, model: Any = None) -> int:
    """Estimate the full token cost of ``prompt`` before it is sent.

    Covers the text body (system, user prompt, fragments), attachments
    (text attachments decoded and tokenised; binary or URL-only
    attachments charged a flat ``_BINARY_ATTACHMENT_TOKENS``), tool
    schemas, tool results from prior turns, and any structured-output
    JSON schema. The number is an estimate — provider-side tokenisers
    disagree on edge cases — but it catches the common "I forgot I
    attached a 4MB PDF" failure mode.

    ``model`` is optional; when supplied, image attachments are scored
    with that provider's image-token formula (Gemini tiling, Anthropic
    pixel-rate, OpenAI tile+base). Without it, images default to
    Gemini's rule — the middle-of-the-road option.
    """
    count = _make_counter()
    provider = _detect_provider(model)
    tokens = count(_prompt_text(prompt))
    for a in getattr(prompt, "attachments", []) or []:
        tokens += _count_attachment_tokens(count, a, provider)
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
        tokens += count(json.dumps(tools_payload, default=str))
    for tr in getattr(prompt, "tool_results", []) or []:
        out = getattr(tr, "output", None)
        if out:
            tokens += count(str(out))
    schema = getattr(prompt, "schema", None)
    if schema is not None:
        tokens += count(json.dumps(schema, default=str))
    return tokens


def estimate_tokens_detailed(
    prompt: llm.Prompt, model: Any = None
) -> tuple[int, str]:
    """Return ``(tokens, source)`` where source is ``"heuristic"`` or the
    provider name when an exact-count adapter produced the figure.

    Adapter failures fall through to the local heuristic, but unlike the
    v0 behaviour they are *not* silent: a one-line notice goes to stderr
    so the user can see when exact mode is actually taking effect and
    when it has silently degraded. Gating is a trust tool; invisible
    fallback is how trust erodes.

    When ``LLM_CONFIRM_TOKENS_DRIFT_WARN`` is set and exact mode succeeds,
    the function compares the exact count to the heuristic and writes a
    one-line warning to stderr if they diverge beyond that percentage.
    The heuristic is calibrated against today's provider formulas; if a
    provider changes how it charges images, PDFs, or tools, the
    heuristic will drift silently until this warning surfaces it.
    """
    if model is not None and _exact_mode():
        from . import _adapters

        for adapter in _adapters.iter_adapters():
            if not adapter.matches(model):
                continue
            name = type(adapter).__name__.removesuffix("Adapter").lower() or "adapter"
            try:
                exact = adapter.count(prompt, model)
                _maybe_warn_drift(exact, prompt, model, name)
                return (exact, name)
            except Exception as exc:
                sys.stderr.write(
                    f"llm-confirm-tokens: {name} exact-count failed "
                    f"({type(exc).__name__}: {exc}); using heuristic.\n"
                )
                break
    return (count_prompt_tokens(prompt, model), "heuristic")


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


def _maybe_warn_drift(
    exact: int, prompt: llm.Prompt, model: Any, provider: str
) -> None:
    """Warn to stderr if heuristic deviates from ``exact`` by ≥ threshold.

    Provides a canary for silent drift: if a provider changes its image
    tile formula or starts billing tools differently, users won't
    discover it from the gate itself (which now reports an exact count)
    — only the drift warning will flag that the heuristic needs
    re-tuning. Opt-in via env var so the extra heuristic pass has no
    effect in the default configuration.
    """
    threshold = _drift_threshold_pct()
    if threshold is None or exact <= 0:
        return
    try:
        heuristic = count_prompt_tokens(prompt, model)
    except Exception:
        return
    delta_pct = abs(exact - heuristic) / exact * 100
    if delta_pct < threshold:
        return
    direction = "under" if heuristic < exact else "over"
    sys.stderr.write(
        f"llm-confirm-tokens: heuristic {heuristic:,} {direction}-counts "
        f"vs {provider} {exact:,} by {delta_pct:.0f}% — local estimates "
        f"are a best guess, not billing-grade.\n"
    )


def estimate_tokens(prompt: llm.Prompt, model: Any = None) -> int:
    """Return a pre-flight token estimate for ``prompt``.

    Thin wrapper over :func:`estimate_tokens_detailed` for callers that
    only want the number. The gate uses the detailed form so it can
    indicate in the confirmation prompt whether the figure is exact.
    """
    return estimate_tokens_detailed(prompt, model)[0]


def _ask_via_tty(tokens: int, source: str = "heuristic") -> bool:
    """Prompt the user for a proceed/cancel decision and return True to proceed.

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
    prefix = "~" if source == "heuristic" else ""
    suffix = "" if source == "heuristic" else f" ({source})"
    message = f"Total tokens: {prefix}{tokens:,}{suffix}. Proceed? [Y/n]: "

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
        f"llm-confirm-tokens: {prefix}{tokens:,} tokens, but no tty available "
        "to confirm. Set LLM_CONFIRM_TOKENS_YES=1 to auto-approve in "
        "non-interactive environments.\n"
    )
    return False


class ConfirmTokensGate:
    """Prompt gate that confirms with the user before large prompts are sent.

    Inject ``tokens_fn`` and ``ask`` for tests — in production the defaults
    (exact counts for providers that support them, heuristic otherwise, and
    platform-appropriate tty prompting) are used.

    ``tokens_fn`` receives ``(prompt, model)``; legacy callers that pass a
    single-argument function are still accepted. ``ask`` receives
    ``(tokens, source)`` where source is ``"heuristic"`` or the provider
    adapter name; legacy single-argument ``ask`` callables are also
    accepted so in-process test code doesn't have to change.
    """

    def __init__(
        self,
        *,
        threshold: int = 0,
        tokens_fn: Callable[..., int] | None = None,
        ask: Callable[..., bool] | None = None,
    ) -> None:
        self.threshold = threshold
        self._tokens_fn = tokens_fn
        self._ask = ask or _ask_via_tty

    def _count(self, prompt: llm.Prompt, model: Any) -> tuple[int, str]:
        """Return ``(tokens, source)``.

        When no ``tokens_fn`` was injected, we call
        :func:`estimate_tokens_detailed` directly so the true source
        ("heuristic", "anthropic", "gemini", "openai") flows through
        to the confirmation message. Injected counters that return
        just an int (typical for tests) are labelled "heuristic" by
        convention; injected counters can return a ``(tokens, source)``
        tuple if they want to control the label.
        """
        if self._tokens_fn is None:
            return estimate_tokens_detailed(prompt, model)
        try:
            result = self._tokens_fn(prompt, model)
        except TypeError:
            result = self._tokens_fn(prompt)
        if isinstance(result, tuple):
            return result  # type: ignore[return-value]
        return (int(result), "heuristic")

    def _invoke_ask(self, tokens: int, source: str) -> bool:
        try:
            return self._ask(tokens, source)
        except TypeError:
            return self._ask(tokens)

    def check(self, prompt: llm.Prompt, model: Any) -> None:
        tokens, source = self._count(prompt, model)
        if tokens < self.threshold:
            return
        if _assume_yes():
            return
        if not self._invoke_ask(tokens, source):
            raise llm.CancelPrompt(f"user declined {tokens:,} token prompt")


@hookimpl
def register_prompt_gates(register: Any) -> None:
    """Register the confirm-tokens gate when the plugin is enabled.

    Keyed on the ``LLM_CONFIRM_TOKENS`` env var so the plugin is a no-op
    for users who have installed it but not opted in.
    """
    if not _is_enabled():
        return
    register(ConfirmTokensGate(threshold=_threshold()))
