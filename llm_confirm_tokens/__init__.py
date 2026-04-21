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
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import llm
from llm import hookimpl

__all__ = [
    "ConfirmTokensGate",
    "count_prompt_tokens",
    "register_prompt_gates",
]


_TRUTHY = ("1", "true", "yes", "on")


def _is_enabled() -> bool:
    return os.environ.get("LLM_CONFIRM_TOKENS", "").strip().lower() in _TRUTHY


def _assume_yes() -> bool:
    return os.environ.get("LLM_CONFIRM_TOKENS_YES", "").strip().lower() in _TRUTHY


def _threshold() -> int:
    raw = os.environ.get("LLM_CONFIRM_TOKENS_THRESHOLD", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


# Per-media token constants, calibrated against Gemini 1.5/2.x documented
# pricing (image and PDF pages both cost ~258 input tokens). These are
# directional — different providers tokenise images differently (OpenAI
# charges per tile, Anthropic per image region) — but 258 per image /
# per PDF page is a defensible default for the "is this about to cost
# me a lot?" question the plugin is trying to answer.
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
        return lambda text: len(enc.encode(text))
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


def _count_attachment_tokens(count: Callable[[str], int], a: Any) -> int:
    data = _attachment_bytes(a)
    mime = _detect_mime(a, data)
    if mime.startswith("image/"):
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


def count_prompt_tokens(prompt: llm.Prompt) -> int:
    """Estimate the full token cost of ``prompt`` before it is sent.

    Covers the text body (system, user prompt, fragments), attachments
    (text attachments decoded and tokenised; binary or URL-only
    attachments charged a flat ``_BINARY_ATTACHMENT_TOKENS``), tool
    schemas, tool results from prior turns, and any structured-output
    JSON schema. The number is an estimate — provider-side tokenisers
    disagree on edge cases — but it catches the common "I forgot I
    attached a 4MB PDF" failure mode.
    """
    count = _make_counter()
    tokens = count(_prompt_text(prompt))
    for a in getattr(prompt, "attachments", []) or []:
        tokens += _count_attachment_tokens(count, a)
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


def _ask_via_tty(tokens: int) -> bool:
    """Prompt the user on ``/dev/tty`` and return True to proceed.

    Falls back to stderr/stdin when ``/dev/tty`` is not available
    (e.g. CI, sandboxed scripts). If no interactive input is possible
    at all, returns True so non-interactive scripts are not blocked —
    users who want strict blocking should gate their scripts on the
    env var directly.
    """
    message = f"Total tokens: {tokens:,}. Proceed? [Y/n]: "
    try:
        with open("/dev/tty", "r+") as tty:
            tty.write(message)
            tty.flush()
            answer = tty.readline()
    except OSError:
        if not sys.stdin.isatty():
            sys.stderr.write(f"llm-confirm-tokens: {tokens:,} tokens (no tty; proceeding)\n")
            return True
        sys.stderr.write(message)
        sys.stderr.flush()
        answer = sys.stdin.readline()
    answer = (answer or "").strip().lower()
    return answer in ("", "y", "yes")


class ConfirmTokensGate:
    """Prompt gate that confirms with the user before large prompts are sent.

    Inject ``tokens_fn`` and ``ask`` for tests — in production the defaults
    (tiktoken-based counting and ``/dev/tty`` prompting) are used.
    """

    def __init__(
        self,
        *,
        threshold: int = 0,
        tokens_fn: Callable[[llm.Prompt], int] | None = None,
        ask: Callable[[int], bool] | None = None,
    ) -> None:
        self.threshold = threshold
        self._tokens_fn = tokens_fn or count_prompt_tokens
        self._ask = ask or _ask_via_tty

    def check(self, prompt: llm.Prompt, model: Any) -> None:
        tokens = self._tokens_fn(prompt)
        if tokens < self.threshold:
            return
        if _assume_yes():
            return
        if not self._ask(tokens):
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
