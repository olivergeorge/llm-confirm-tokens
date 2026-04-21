"""Tests for the opt-in Anthropic exact-count adapter.

The adapter reaches for the ``anthropic`` SDK at call time, so these
tests inject a fake ``anthropic`` module into ``sys.modules`` before
constructing the adapter. This keeps the test environment free of a
real SDK dependency and — more importantly — proves the plugin never
touches the network when the SDK isn't present.
"""

from __future__ import annotations

import base64
import sys
import types

import pytest

from llm_confirm_tokens import ConfirmTokensGate, estimate_tokens
from llm_confirm_tokens._adapters import AnthropicAdapter


class _FakeCountResponse:
    def __init__(self, tokens: int):
        self.input_tokens = tokens


class _FakeMessages:
    def __init__(self, recorder: list[dict]):
        self._recorder = recorder

    def count_tokens(self, **kwargs):
        self._recorder.append(kwargs)
        return _FakeCountResponse(tokens=kwargs.get("_force_tokens", 4242))


class _FakeAnthropicClient:
    def __init__(self, recorder: list[dict], **client_kwargs):
        self.client_kwargs = client_kwargs
        self.messages = _FakeMessages(recorder)


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Install a fake ``anthropic`` module on sys.modules and return the
    list of count_tokens kwargs it sees, so tests can assert on the
    payload that was built.
    """
    recorder: list[dict] = []

    fake_module = types.ModuleType("anthropic")

    def Anthropic(**kwargs):  # noqa: N802 — matches SDK's casing
        return _FakeAnthropicClient(recorder, **kwargs)

    fake_module.Anthropic = Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return recorder


class _FakeClaude:
    """Duck-typed llm.Model with a claude-shaped model_id."""

    def __init__(self, model_id="claude-sonnet-4-5"):
        self.model_id = model_id


class _FakePrompt:
    def __init__(
        self,
        prompt="",
        system=None,
        fragments=(),
        system_fragments=(),
        attachments=(),
        tools=(),
    ):
        self.prompt = prompt
        self.system = system
        self.fragments = list(fragments)
        self.system_fragments = list(system_fragments)
        self.attachments = list(attachments)
        self.tools = list(tools)
        self.tool_results = []
        self.schema = None


class _FakeAttachment:
    def __init__(self, *, content=None, path=None, url=None, type=None):
        self.content = content
        self.path = path
        self.url = url
        self.type = type


def test_matches_recognises_claude_model_ids():
    adapter = AnthropicAdapter()
    assert adapter.matches(_FakeClaude("claude-sonnet-4-5"))
    assert adapter.matches(_FakeClaude("claude-3-5-sonnet-20241022"))
    assert adapter.matches(_FakeClaude("anthropic/claude-opus-4"))


def test_matches_rejects_non_claude_models():
    adapter = AnthropicAdapter()
    assert not adapter.matches(_FakeClaude("gemini-3-flash-preview"))
    assert not adapter.matches(_FakeClaude("gpt-4o"))


def test_adapter_sends_text_prompt_and_returns_input_tokens(fake_anthropic):
    adapter = AnthropicAdapter()
    tokens = adapter.count(_FakePrompt("hello world"), _FakeClaude())
    assert tokens == 4242  # the fake always returns 4242
    assert len(fake_anthropic) == 1
    payload = fake_anthropic[0]
    assert payload["model"] == "claude-sonnet-4-5"
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello world"}]}
    ]
    assert "system" not in payload
    assert "tools" not in payload


def test_adapter_strips_provider_prefix_from_model_id(fake_anthropic):
    adapter = AnthropicAdapter()
    adapter.count(_FakePrompt("hi"), _FakeClaude("anthropic/claude-opus-4"))
    assert fake_anthropic[0]["model"] == "claude-opus-4"


def test_adapter_includes_system_and_fragments(fake_anthropic):
    adapter = AnthropicAdapter()
    adapter.count(
        _FakePrompt(
            "hello",
            system="you are helpful",
            system_fragments=["extra system instruction"],
            fragments=["retrieved context"],
        ),
        _FakeClaude(),
    )
    payload = fake_anthropic[0]
    assert payload["system"] == "you are helpful\nextra system instruction"
    assert payload["messages"][0]["content"] == [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "retrieved context"},
    ]


def test_adapter_encodes_image_attachment_as_base64(fake_anthropic):
    image_bytes = b"\xff\xd8\xff" + b"\x00" * 100
    adapter = AnthropicAdapter()
    adapter.count(
        _FakePrompt(
            "describe",
            attachments=[_FakeAttachment(content=image_bytes, type="image/jpeg")],
        ),
        _FakeClaude(),
    )
    content = fake_anthropic[0]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    image_block = content[1]
    assert image_block["type"] == "image"
    assert image_block["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(image_block["source"]["data"]) == image_bytes


def test_adapter_encodes_pdf_attachment_as_document(fake_anthropic):
    pdf_bytes = b"%PDF-1.4\n%%EOF"
    adapter = AnthropicAdapter()
    adapter.count(
        _FakePrompt(
            "summarise",
            attachments=[_FakeAttachment(content=pdf_bytes, type="application/pdf")],
        ),
        _FakeClaude(),
    )
    content = fake_anthropic[0]["messages"][0]["content"]
    pdf_block = content[1]
    assert pdf_block["type"] == "document"
    assert pdf_block["source"]["media_type"] == "application/pdf"
    assert base64.b64decode(pdf_block["source"]["data"]) == pdf_bytes


def test_adapter_serialises_tools(fake_anthropic):
    class _Tool:
        name = "search"
        description = "search the index"
        input_schema = {"type": "object", "properties": {"q": {"type": "string"}}}

    adapter = AnthropicAdapter()
    adapter.count(_FakePrompt("hi", tools=[_Tool()]), _FakeClaude())
    tools = fake_anthropic[0]["tools"]
    assert tools == [
        {
            "name": "search",
            "description": "search the index",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
        }
    ]


def test_adapter_drops_url_only_attachments(fake_anthropic):
    """URL attachments without content must not trigger a network fetch.

    The adapter's job is to never surprise users with pre-flight HTTP
    traffic; URL-only attachments are silently omitted from the count
    payload, so the over/undercount is at most the heuristic fallback.
    """
    adapter = AnthropicAdapter()
    adapter.count(
        _FakePrompt(
            "look at this",
            attachments=[_FakeAttachment(url="https://example.com/img.jpg")],
        ),
        _FakeClaude(),
    )
    content = fake_anthropic[0]["messages"][0]["content"]
    assert content == [{"type": "text", "text": "look at this"}]


def test_estimate_tokens_uses_adapter_when_exact_mode_on(
    fake_anthropic, monkeypatch
):
    monkeypatch.setenv("LLM_CONFIRM_TOKENS_EXACT", "1")
    n = estimate_tokens(_FakePrompt("hi"), _FakeClaude())
    assert n == 4242
    assert len(fake_anthropic) == 1


def test_estimate_tokens_ignores_adapter_without_exact_mode(
    fake_anthropic, monkeypatch
):
    monkeypatch.delenv("LLM_CONFIRM_TOKENS_EXACT", raising=False)
    estimate_tokens(_FakePrompt("hi"), _FakeClaude())
    # Adapter must not have been called — heuristic only.
    assert fake_anthropic == []


def test_estimate_tokens_falls_back_to_heuristic_on_adapter_error(
    monkeypatch,
):
    """A failing adapter must not break gating — heuristic kicks in silently."""
    monkeypatch.setenv("LLM_CONFIRM_TOKENS_EXACT", "1")

    fake_module = types.ModuleType("anthropic")

    def Anthropic(**kwargs):  # noqa: N802
        class _Broken:
            class messages:
                @staticmethod
                def count_tokens(**_):
                    raise RuntimeError("API down")

        return _Broken()

    fake_module.Anthropic = Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    # Heuristic has to produce *some* positive number for a non-empty prompt.
    n = estimate_tokens(_FakePrompt("hi"), _FakeClaude())
    assert n > 0


def test_gate_uses_exact_count_via_adapter(fake_anthropic, monkeypatch):
    """End-to-end: enabling exact mode makes the gate prompt with the
    adapter's number, not the heuristic."""
    monkeypatch.setenv("LLM_CONFIRM_TOKENS_EXACT", "1")
    asked: list[int] = []

    gate = ConfirmTokensGate(threshold=0, ask=lambda n: asked.append(n) or True)
    gate.check(_FakePrompt("hi"), _FakeClaude())
    assert asked == [4242]


def test_gate_legacy_single_arg_tokens_fn_still_works():
    """Back-compat: callers passing a one-arg tokens_fn continue to work."""
    gate = ConfirmTokensGate(
        threshold=0,
        tokens_fn=lambda p: 7,  # ignores model
        ask=lambda n: True,
    )
    gate.check(_FakePrompt("hi"), model=None)  # must not raise
