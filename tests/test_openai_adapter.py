"""Tests for the opt-in OpenAI exact-count adapter.

Uses the same fake-SDK-in-sys.modules pattern as the other adapter
tests. The fixture wires up ``openai.OpenAI`` → a stub with a
``.responses.input_tokens.count`` chain so tests can assert on the
payload the adapter built.
"""

from __future__ import annotations

import base64
import sys
import types

import pytest

from llm_confirm_tokens import ConfirmTokensGate, estimate_tokens
from llm_confirm_tokens._adapters import OpenAIAdapter


class _FakeCountResponse:
    def __init__(self, tokens: int):
        self.input_tokens = tokens


class _FakeInputTokens:
    def __init__(self, recorder: list[dict]):
        self._recorder = recorder

    def count(self, **kwargs):
        self._recorder.append(kwargs)
        return _FakeCountResponse(tokens=kwargs.get("_force_tokens", 314))


class _FakeResponses:
    def __init__(self, recorder: list[dict]):
        self.input_tokens = _FakeInputTokens(recorder)


class _FakeOpenAI:
    def __init__(self, recorder: list[dict], **client_kwargs):
        self.client_kwargs = client_kwargs
        self.responses = _FakeResponses(recorder)


@pytest.fixture
def fake_openai(monkeypatch):
    recorder: list[dict] = []
    fake_module = types.ModuleType("openai")

    def OpenAI(**kwargs):  # noqa: N802
        return _FakeOpenAI(recorder, **kwargs)

    fake_module.OpenAI = OpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return recorder


class _FakeGPT:
    def __init__(self, model_id="gpt-4o-mini"):
        self.model_id = model_id


class _FakePrompt:
    def __init__(
        self,
        prompt="",
        system=None,
        fragments=(),
        system_fragments=(),
        attachments=(),
    ):
        self.prompt = prompt
        self.system = system
        self.fragments = list(fragments)
        self.system_fragments = list(system_fragments)
        self.attachments = list(attachments)
        self.tools = []
        self.tool_results = []
        self.schema = None


class _FakeAttachment:
    def __init__(self, *, content=None, path=None, url=None, type=None):
        self.content = content
        self.path = path
        self.url = url
        self.type = type


def test_matches_recognises_openai_model_ids():
    adapter = OpenAIAdapter()
    ids = (
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "chatgpt-4o-latest",
        "o1-mini",
        "o3",
        "o4-mini",
    )
    for mid in ids:
        assert adapter.matches(_FakeGPT(mid)), mid


def test_matches_rejects_non_openai_models():
    adapter = OpenAIAdapter()
    for mid in ("claude-sonnet-4-5", "gemini-3-flash-preview"):
        assert not adapter.matches(_FakeGPT(mid))


def test_adapter_sends_text_prompt_and_returns_input_tokens(fake_openai):
    adapter = OpenAIAdapter()
    tokens = adapter.count(_FakePrompt("hello world"), _FakeGPT())
    assert tokens == 314
    payload = fake_openai[0]
    assert payload["model"] == "gpt-4o-mini"
    assert payload["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hello world"}],
        }
    ]
    assert "instructions" not in payload


def test_adapter_maps_system_to_instructions(fake_openai):
    """The Responses API calls the system prompt ``instructions``."""
    adapter = OpenAIAdapter()
    adapter.count(
        _FakePrompt(
            "hi",
            system="you are terse",
            system_fragments=["additional rules"],
        ),
        _FakeGPT(),
    )
    assert fake_openai[0]["instructions"] == "you are terse\nadditional rules"


def test_adapter_encodes_image_as_input_image_data_url(fake_openai):
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    adapter = OpenAIAdapter()
    adapter.count(
        _FakePrompt(
            "describe",
            attachments=[_FakeAttachment(content=png, type="image/png")],
        ),
        _FakeGPT(),
    )
    content = fake_openai[0]["input"][0]["content"]
    image = next(p for p in content if p["type"] == "input_image")
    assert image["image_url"].startswith("data:image/png;base64,")
    b64 = image["image_url"].split("base64,", 1)[1]
    assert base64.b64decode(b64) == png


def test_adapter_encodes_pdf_as_input_file_data_url(fake_openai):
    pdf = b"%PDF-1.4\nmini"
    adapter = OpenAIAdapter()
    adapter.count(
        _FakePrompt(
            "summarise",
            attachments=[
                _FakeAttachment(content=pdf, path="docs/report.pdf", type="application/pdf")
            ],
        ),
        _FakeGPT(),
    )
    content = fake_openai[0]["input"][0]["content"]
    file_part = next(p for p in content if p["type"] == "input_file")
    assert file_part["filename"] == "report.pdf"
    assert file_part["file_data"].startswith("data:application/pdf;base64,")
    b64 = file_part["file_data"].split("base64,", 1)[1]
    assert base64.b64decode(b64) == pdf


def test_adapter_drops_url_only_attachments(fake_openai):
    adapter = OpenAIAdapter()
    adapter.count(
        _FakePrompt(
            "look",
            attachments=[_FakeAttachment(url="https://example.com/img.jpg")],
        ),
        _FakeGPT(),
    )
    content = fake_openai[0]["input"][0]["content"]
    assert all(p["type"] == "input_text" for p in content)


def test_adapter_strips_provider_prefix(fake_openai):
    adapter = OpenAIAdapter()
    adapter.count(_FakePrompt("hi"), _FakeGPT("openai/gpt-4.1"))
    assert fake_openai[0]["model"] == "gpt-4.1"


def test_estimate_tokens_uses_openai_adapter_when_exact_mode_on(
    fake_openai, monkeypatch
):
    monkeypatch.setenv("LLM_CONFIRM_TOKENS_EXACT", "1")
    n = estimate_tokens(_FakePrompt("hi"), _FakeGPT())
    assert n == 314


def test_estimate_tokens_falls_back_when_openai_sdk_missing_responses(monkeypatch):
    """Older openai SDKs without .responses.input_tokens must fall back."""
    monkeypatch.setenv("LLM_CONFIRM_TOKENS_EXACT", "1")
    fake_module = types.ModuleType("openai")

    class _OldClient:
        def __init__(self, **_):
            self.responses = types.SimpleNamespace()  # no input_tokens

    fake_module.OpenAI = _OldClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    n = estimate_tokens(_FakePrompt("hi"), _FakeGPT())
    assert n > 0  # heuristic fallback


def test_gate_uses_openai_exact_count_via_adapter(fake_openai, monkeypatch):
    monkeypatch.setenv("LLM_CONFIRM_TOKENS_EXACT", "1")
    asked: list[int] = []
    gate = ConfirmTokensGate(threshold=0, ask=lambda n: asked.append(n) or True)
    gate.check(_FakePrompt("hi"), _FakeGPT())
    assert asked == [314]
