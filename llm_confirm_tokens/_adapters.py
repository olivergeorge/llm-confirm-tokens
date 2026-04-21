"""Provider-native token-count adapters.

Adapters are opt-in via ``LLM_CONFIRM_TOKENS_EXACT=1`` and discovered by
:func:`iter_adapters`. Each adapter decides whether it matches the model
in play and, if so, turns an :class:`llm.Prompt` into the provider's
count-tokens request. Failures fall through to the local heuristic in
:func:`llm_confirm_tokens.estimate_tokens`, so unavailable keys, network
flakes, or SDK drift never turn into a plugin crash.

Adding a provider is a new module + one entry in :func:`iter_adapters`.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Iterable
from typing import Any, Protocol


class Adapter(Protocol):
    def matches(self, model: Any) -> bool: ...
    def count(self, prompt: Any, model: Any) -> int: ...


def iter_adapters() -> Iterable[Adapter]:
    """Adapters in priority order. Order only matters if a model could
    plausibly match two providers, which is currently impossible."""
    return (AnthropicAdapter(),)


def _get_anthropic_key() -> str | None:
    """Resolve an Anthropic key from the environment or llm's keyring.

    Checked in this order: ``ANTHROPIC_API_KEY`` environment variable,
    then llm's stored key aliases (``claude`` / ``anthropic``). Any
    failure returns ``None`` so the caller falls back to the heuristic.
    """
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return env_key
    try:
        import llm

        for alias in ("claude", "anthropic"):
            try:
                return llm.get_key("", alias, "ANTHROPIC_API_KEY")
            except Exception:
                continue
    except Exception:
        pass
    return None


def _attachment_content_block(a: Any) -> dict | None:
    """Turn an llm.Attachment into the content block count_tokens expects.

    Returns ``None`` for attachments we can't represent without a network
    fetch (URL-only) or that we don't have a bytes source for — the
    caller drops them and the adapter falls back to the heuristic if
    that leaves the message empty.
    """
    mime = getattr(a, "type", None)
    if not mime:
        path = getattr(a, "path", None)
        if path:
            try:
                from llm.utils import mimetype_from_path

                mime = mimetype_from_path(path) or ""
            except Exception:
                mime = ""
    content = getattr(a, "content", None)
    if content is None:
        path = getattr(a, "path", None)
        if path:
            try:
                from pathlib import Path

                content = Path(path).read_bytes()
            except OSError:
                content = None
    if content is None:
        return None
    data_b64 = base64.b64encode(content).decode("ascii")
    if mime and mime.startswith("image/"):
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": data_b64},
        }
    if mime == "application/pdf":
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": data_b64,
            },
        }
    try:
        return {"type": "text", "text": content.decode("utf-8")}
    except UnicodeDecodeError:
        return None


def _tool_to_anthropic(tool: Any) -> dict:
    return {
        "name": getattr(tool, "name", "tool"),
        "description": getattr(tool, "description", "") or "",
        "input_schema": getattr(tool, "input_schema", {}) or {"type": "object"},
    }


class AnthropicAdapter:
    """Exact token counts via Anthropic's free /v1/messages/count_tokens.

    Matches any model whose ``model_id`` looks Claude-shaped (the
    ``claude-`` prefix) or whose implementing class comes from
    ``llm_anthropic``. Builds a minimal Messages-API-compatible payload
    — system prompt, user content (text + fragments + text/image/PDF
    attachments), and tools — then returns ``response.input_tokens``.

    Tool results from prior turns are counted heuristically by the
    caller: Anthropic's message format requires alternating assistant/
    user turns to express tool use, and mis-ordering them would cause
    count_tokens to reject the request. The over-count is small in
    practice.
    """

    def matches(self, model: Any) -> bool:
        mid = getattr(model, "model_id", "") or ""
        if mid.startswith("claude-") or "anthropic/" in mid or "claude/" in mid:
            return True
        return (type(model).__module__ or "").startswith("llm_anthropic")

    def _model_id(self, model: Any) -> str:
        mid = getattr(model, "model_id", "") or ""
        for prefix in ("anthropic/", "claude/"):
            if mid.startswith(prefix):
                return mid[len(prefix) :]
        return mid

    def count(self, prompt: Any, model: Any) -> int:
        from anthropic import Anthropic

        api_key = _get_anthropic_key()
        client = Anthropic(api_key=api_key) if api_key else Anthropic()

        user_blocks: list[dict] = []
        body = getattr(prompt, "prompt", None) or getattr(prompt, "_prompt", None)
        if body:
            user_blocks.append({"type": "text", "text": body})
        for f in getattr(prompt, "fragments", []) or []:
            user_blocks.append({"type": "text", "text": str(f)})
        for a in getattr(prompt, "attachments", []) or []:
            block = _attachment_content_block(a)
            if block is not None:
                user_blocks.append(block)

        if not user_blocks:
            user_blocks = [{"type": "text", "text": ""}]

        kwargs: dict[str, Any] = {
            "model": self._model_id(model),
            "messages": [{"role": "user", "content": user_blocks}],
        }

        system_parts: list[str] = []
        system = getattr(prompt, "system", None)
        if system:
            system_parts.append(system)
        for sf in getattr(prompt, "system_fragments", []) or []:
            system_parts.append(str(sf))
        if system_parts:
            kwargs["system"] = "\n".join(system_parts)

        tools = getattr(prompt, "tools", []) or []
        if tools:
            kwargs["tools"] = [_tool_to_anthropic(t) for t in tools]

        response = client.messages.count_tokens(**kwargs)
        return int(response.input_tokens)
