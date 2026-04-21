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
    return (AnthropicAdapter(), GeminiAdapter(), OpenAIAdapter())


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


def _get_gemini_key() -> str | None:
    """Resolve a Gemini API key from the environment or llm's keyring."""
    for env in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(env)
        if v:
            return v
    try:
        import llm

        for alias in ("gemini", "google"):
            try:
                return llm.get_key("", alias, "GEMINI_API_KEY")
            except Exception:
                continue
    except Exception:
        pass
    return None


def _gemini_part_from_attachment(a: Any) -> dict | None:
    """Turn an llm.Attachment into a Gemini Part dict.

    URL-only attachments are skipped (no network fetches from the
    plugin). Returns ``None`` to signal "drop this attachment and let
    the caller proceed without it" — a small under-count is preferable
    to a surprise HTTP call.
    """
    mime = getattr(a, "type", None)
    content = getattr(a, "content", None)
    path = getattr(a, "path", None)
    if not mime and path:
        try:
            from llm.utils import mimetype_from_path

            mime = mimetype_from_path(path) or ""
        except Exception:
            mime = ""
    if content is None and path:
        try:
            from pathlib import Path

            content = Path(path).read_bytes()
        except OSError:
            content = None
    if content is None:
        return None
    if mime and (mime.startswith("image/") or mime == "application/pdf"):
        return {
            "inline_data": {
                "mime_type": mime,
                "data": base64.b64encode(content).decode("ascii"),
            }
        }
    try:
        return {"text": content.decode("utf-8")}
    except UnicodeDecodeError:
        return None


class GeminiAdapter:
    """Exact token counts via ``google.genai.Client.models.count_tokens``.

    Matches any model whose ``model_id`` looks Gemini-shaped (the
    ``gemini-`` prefix, or an ``llm-gemini``-style ``gemini/…`` alias)
    or whose implementing class comes from ``llm_gemini``. Builds a
    single user Content with text + fragments + text/image/PDF
    attachments and returns ``response.total_tokens``.

    System prompt is prepended as a leading user-role text part rather
    than routed through ``CountTokensConfig.system_instruction``. The
    latter would match billing more tightly (~5% closer), but the
    ``count_tokens`` endpoint rejects ``system_instruction`` for some
    Gemini models (e.g. ``gemini-flash-lite`` raises ``ValueError:
    system_instruction parameter is not supported in Gemini API`` even
    when the same model accepts it on ``generate_content``). Inlining
    keeps the adapter model-agnostic at the cost of an envelope
    over-count that errs in the safe direction for a gating tool.

    Tools and tool results are not included: the Gemini tool-calling
    format requires a multi-turn message history whose invariants
    count_tokens validates, and the over-count risk isn't worth the
    adapter complexity. Callers that need exact counts for tool-heavy
    prompts can extend this adapter.
    """

    def matches(self, model: Any) -> bool:
        mid = getattr(model, "model_id", "") or ""
        if mid.startswith("gemini-") or mid.startswith("gemini/") or "/gemini-" in mid:
            return True
        return (type(model).__module__ or "").startswith("llm_gemini")

    def _model_id(self, model: Any) -> str:
        mid = getattr(model, "model_id", "") or ""
        for prefix in ("gemini/", "google/"):
            if mid.startswith(prefix):
                return mid[len(prefix) :]
        return mid

    def count(self, prompt: Any, model: Any) -> int:
        from google import genai

        api_key = _get_gemini_key()
        client = genai.Client(api_key=api_key) if api_key else genai.Client()

        parts: list[dict] = []
        # System prompt / fragments go first as leading text parts. See
        # class docstring for why this isn't routed via CountTokensConfig.
        system = getattr(prompt, "system", None)
        if system:
            parts.append({"text": system})
        for sf in getattr(prompt, "system_fragments", []) or []:
            parts.append({"text": str(sf)})

        body = getattr(prompt, "prompt", None) or getattr(prompt, "_prompt", None)
        if body:
            parts.append({"text": body})
        for f in getattr(prompt, "fragments", []) or []:
            parts.append({"text": str(f)})
        for a in getattr(prompt, "attachments", []) or []:
            part = _gemini_part_from_attachment(a)
            if part is not None:
                parts.append(part)

        if not parts:
            parts = [{"text": ""}]

        contents = [{"role": "user", "parts": parts}]
        response = client.models.count_tokens(
            model=self._model_id(model),
            contents=contents,
        )
        return int(response.total_tokens)


def _get_openai_key() -> str | None:
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    try:
        import llm

        for alias in ("openai",):
            try:
                return llm.get_key("", alias, "OPENAI_API_KEY")
            except Exception:
                continue
    except Exception:
        pass
    return None


def _openai_input_part(a: Any) -> dict | None:
    """Turn an llm.Attachment into a Responses-API ``input_*`` content part.

    Images become ``input_image`` with a base64 data URL; PDFs become
    ``input_file`` with a base64 ``file_data`` data URL; text decodes
    into ``input_text``. URL-only attachments without content are
    dropped — the adapter never fetches on behalf of the counter.
    """
    mime = getattr(a, "type", None)
    content = getattr(a, "content", None)
    path = getattr(a, "path", None)
    if not mime and path:
        try:
            from llm.utils import mimetype_from_path

            mime = mimetype_from_path(path) or ""
        except Exception:
            mime = ""
    if content is None and path:
        try:
            from pathlib import Path

            content = Path(path).read_bytes()
        except OSError:
            content = None
    if content is None:
        return None
    if mime and mime.startswith("image/"):
        data_url = f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}"
        return {"type": "input_image", "image_url": data_url}
    if mime == "application/pdf":
        data_url = f"data:application/pdf;base64,{base64.b64encode(content).decode('ascii')}"
        filename = (
            getattr(a, "path", None) or "").rsplit("/", 1)[-1] or "attachment.pdf"
        return {
            "type": "input_file",
            "filename": filename,
            "file_data": data_url,
        }
    try:
        return {"type": "input_text", "text": content.decode("utf-8")}
    except UnicodeDecodeError:
        return None


class OpenAIAdapter:
    """Exact token counts via OpenAI's ``/v1/responses/input_tokens``.

    Uses the Responses-API preflight counter: the token count reflects
    the model's full processed input (text, images, PDFs, and tools),
    and the endpoint is free. Matches any model whose ``model_id``
    looks OpenAI-shaped (``gpt-``, ``chatgpt-``, ``o1``, ``o3``,
    ``o4``) or whose implementing class comes from ``llm`` core's
    default plugins.

    The adapter targets ``client.responses.input_tokens.count``; if the
    installed ``openai`` SDK predates that resource the call raises and
    the gate falls back to the local heuristic, same as any other
    adapter error. OpenAI's tokeniser for chat-completions models is
    the same as for Responses-API models, so the count stays accurate
    even when the actual prompt is ultimately sent via chat
    completions rather than ``/responses``.
    """

    _MODEL_PREFIXES = ("gpt-", "chatgpt-", "o1", "o3", "o4")

    def matches(self, model: Any) -> bool:
        mid = getattr(model, "model_id", "") or ""
        if any(mid.startswith(p) for p in self._MODEL_PREFIXES):
            return True
        if mid.startswith("openai/") or "/gpt-" in mid:
            return True
        mod = type(model).__module__ or ""
        return mod.startswith("llm.default_plugins.openai_models")

    def _model_id(self, model: Any) -> str:
        mid = getattr(model, "model_id", "") or ""
        for prefix in ("openai/",):
            if mid.startswith(prefix):
                return mid[len(prefix) :]
        return mid

    def count(self, prompt: Any, model: Any) -> int:
        from openai import OpenAI

        api_key = _get_openai_key()
        client = OpenAI(api_key=api_key) if api_key else OpenAI()

        user_content: list[dict] = []
        body = getattr(prompt, "prompt", None) or getattr(prompt, "_prompt", None)
        if body:
            user_content.append({"type": "input_text", "text": body})
        for f in getattr(prompt, "fragments", []) or []:
            user_content.append({"type": "input_text", "text": str(f)})
        for a in getattr(prompt, "attachments", []) or []:
            part = _openai_input_part(a)
            if part is not None:
                user_content.append(part)
        if not user_content:
            user_content = [{"type": "input_text", "text": ""}]

        kwargs: dict[str, Any] = {
            "model": self._model_id(model),
            "input": [{"role": "user", "content": user_content}],
        }

        system_parts: list[str] = []
        system = getattr(prompt, "system", None)
        if system:
            system_parts.append(system)
        for sf in getattr(prompt, "system_fragments", []) or []:
            system_parts.append(str(sf))
        if system_parts:
            # Responses API calls the system prompt "instructions".
            kwargs["instructions"] = "\n".join(system_parts)

        response = client.responses.input_tokens.count(**kwargs)
        return int(response.input_tokens)
