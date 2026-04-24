"""Tests for the opt-in Gemini exact-count adapter.

As with the Anthropic adapter tests, we inject a fake ``google.genai``
module into ``sys.modules`` so the tests do not require the real SDK
and never touch the network. Asserting on the payload the fake client
receives is the specification of how the adapter builds its
``count_tokens`` request.
"""

from __future__ import annotations

import base64
import sys
import types

import pytest

from llm_confirm_tokens import ConfirmTokensGate, estimate_tokens
from llm_confirm_tokens._adapters import GeminiAdapter


class _FakeCountResponse:
    def __init__(self, tokens: int):
        self.total_tokens = tokens


class _FakeModelsNamespace:
    def __init__(self, recorder: list[dict]):
        self._recorder = recorder

    def count_tokens(self, **kwargs):
        self._recorder.append(kwargs)
        return _FakeCountResponse(tokens=kwargs.get("_force_tokens", 9001))


class _FakeGenAIClient:
    def __init__(self, recorder: list[dict], **client_kwargs):
        self.client_kwargs = client_kwargs
        self.models = _FakeModelsNamespace(recorder)


@pytest.fixture
def fake_genai(monkeypatch):
    """Install fake ``google.genai`` and ``google`` modules, and return the
    list of count_tokens kwargs observed — i.e. the payload the adapter
    built from the llm.Prompt."""
    recorder: list[dict] = []

    fake_genai_module = types.ModuleType("google.genai")

    def Client(**kwargs):  # noqa: N802 — matches SDK's casing
        return _FakeGenAIClient(recorder, **kwargs)

    fake_genai_module.Client = Client  # type: ignore[attr-defined]

    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_module)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return recorder


class _FakeGemini:
    def __init__(self, model_id="gemini-3-flash-preview"):
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


def test_matches_recognises_gemini_model_ids():
    adapter = GeminiAdapter()
    assert adapter.matches(_FakeGemini("gemini-3-flash-preview"))
    assert adapter.matches(_FakeGemini("gemini-2.0-flash"))
    assert adapter.matches(_FakeGemini("gemini/gemini-1.5-pro"))


def test_matches_rejects_non_gemini_models():
    adapter = GeminiAdapter()
    assert not adapter.matches(_FakeGemini("claude-sonnet-4-5"))
    assert not adapter.matches(_FakeGemini("gpt-4o"))


def test_adapter_strips_provider_prefix_from_model_id(fake_genai):
    adapter = GeminiAdapter()
    adapter.count(_FakePrompt("hi"), _FakeGemini("gemini/gemini-1.5-pro"))
    assert fake_genai[0]["model"] == "gemini-1.5-pro"


def test_adapter_sends_text_prompt_and_returns_total_tokens(fake_genai):
    adapter = GeminiAdapter()
    tokens = adapter.count(_FakePrompt("hello world"), _FakeGemini())
    assert tokens == 9001
    assert len(fake_genai) == 1
    payload = fake_genai[0]
    assert payload["model"] == "gemini-3-flash-preview"
    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": "hello world"}]}
    ]


def test_adapter_prepends_system_as_leading_text_parts(fake_genai):
    """System prompt and system_fragments go in as leading user-role text
    parts — one part each, not a pre-joined block — because
    ``CountTokensConfig.system_instruction`` is rejected by several
    Gemini models (e.g. ``gemini-flash-lite``) even when the same
    models accept it on ``generate_content``. Inlining keeps the
    adapter model-agnostic.
    """
    adapter = GeminiAdapter()
    adapter.count(
        _FakePrompt(
            "user request",
            system="you are helpful",
            system_fragments=["extra system"],
            fragments=["retrieved context"],
        ),
        _FakeGemini(),
    )
    payload = fake_genai[0]
    # The config kwarg must not be set — otherwise the flash-lite class
    # of models rejects the request at the SDK boundary.
    assert payload.get("config") is None
    parts = payload["contents"][0]["parts"]
    assert parts[0] == {"text": "you are helpful"}
    assert parts[1] == {"text": "extra system"}
    assert {"text": "user request"} in parts
    assert {"text": "retrieved context"} in parts


def test_adapter_encodes_image_as_inline_data(fake_genai):
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"\xaa" * 50
    adapter = GeminiAdapter()
    adapter.count(
        _FakePrompt(
            "describe",
            attachments=[_FakeAttachment(content=image_bytes, type="image/png")],
        ),
        _FakeGemini(),
    )
    parts = fake_genai[0]["contents"][0]["parts"]
    inline = next(p for p in parts if "inline_data" in p)
    assert inline["inline_data"]["mime_type"] == "image/png"
    assert base64.b64decode(inline["inline_data"]["data"]) == image_bytes


def test_adapter_encodes_pdf_as_inline_data(fake_genai):
    pdf_bytes = b"%PDF-1.4\nminimal"
    adapter = GeminiAdapter()
    adapter.count(
        _FakePrompt(
            "summarise",
            attachments=[_FakeAttachment(content=pdf_bytes, type="application/pdf")],
        ),
        _FakeGemini(),
    )
    parts = fake_genai[0]["contents"][0]["parts"]
    inline = next(p for p in parts if "inline_data" in p)
    assert inline["inline_data"]["mime_type"] == "application/pdf"
    assert base64.b64decode(inline["inline_data"]["data"]) == pdf_bytes


def test_adapter_encodes_audio_as_inline_data(fake_genai):
    """Audio attachments must be passed through to count_tokens as
    inline_data — dropping them (the previous behaviour) silently
    under-counted Gemini prompts by ~10× on real-world voice memos.
    """
    audio_bytes = b"\x00\x00\x00\x20ftypM4A " + b"\xaa" * 200
    adapter = GeminiAdapter()
    adapter.count(
        _FakePrompt(
            "transcribe",
            attachments=[_FakeAttachment(content=audio_bytes, type="audio/mp4")],
        ),
        _FakeGemini(),
    )
    parts = fake_genai[0]["contents"][0]["parts"]
    inline = next(p for p in parts if "inline_data" in p)
    assert inline["inline_data"]["mime_type"] == "audio/mp4"
    assert base64.b64decode(inline["inline_data"]["data"]) == audio_bytes


def test_adapter_encodes_video_as_inline_data(fake_genai):
    """Same widening as audio — video bills heavily and must reach the
    count_tokens endpoint rather than being silently dropped.
    """
    video_bytes = b"\x00\x00\x00\x20ftypisom" + b"\xbb" * 200
    adapter = GeminiAdapter()
    adapter.count(
        _FakePrompt(
            "summarise",
            attachments=[_FakeAttachment(content=video_bytes, type="video/mp4")],
        ),
        _FakeGemini(),
    )
    parts = fake_genai[0]["contents"][0]["parts"]
    inline = next(p for p in parts if "inline_data" in p)
    assert inline["inline_data"]["mime_type"] == "video/mp4"


def test_adapter_drops_url_only_attachments(fake_genai):
    """URL attachments without content are silently dropped — no HEAD or
    GET is performed just to compute a token estimate."""
    adapter = GeminiAdapter()
    adapter.count(
        _FakePrompt(
            "describe",
            attachments=[_FakeAttachment(url="https://example.com/image.jpg")],
        ),
        _FakeGemini(),
    )
    parts = fake_genai[0]["contents"][0]["parts"]
    assert all("inline_data" not in p for p in parts)


def test_estimate_tokens_uses_gemini_adapter_when_exact_mode_on(
    fake_genai, monkeypatch
):
    monkeypatch.setenv("LLM_CONFIRM_TOKENS_EXACT", "1")
    n = estimate_tokens(_FakePrompt("hi"), _FakeGemini())
    assert n == 9001
    assert len(fake_genai) == 1


def test_estimate_tokens_falls_back_to_heuristic_on_gemini_error(monkeypatch):
    """SDK errors inside the Gemini adapter must not fail gating."""
    monkeypatch.setenv("LLM_CONFIRM_TOKENS_EXACT", "1")

    fake_module = types.ModuleType("google.genai")

    def Client(**kwargs):  # noqa: N802
        class _Broken:
            class models:
                @staticmethod
                def count_tokens(**_):
                    raise RuntimeError("Gemini API unreachable")

        return _Broken()

    fake_module.Client = Client  # type: ignore[attr-defined]
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_module)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    n = estimate_tokens(_FakePrompt("hi"), _FakeGemini())
    assert n > 0  # heuristic kicked in


def test_gate_uses_gemini_exact_count_via_adapter(fake_genai, monkeypatch):
    monkeypatch.setenv("LLM_CONFIRM_TOKENS_EXACT", "1")
    asked: list[int] = []
    gate = ConfirmTokensGate(threshold=0, ask=lambda n: asked.append(n) or True)
    gate.check(_FakePrompt("hi"), _FakeGemini())
    assert asked == [9001]


def test_gemini_adapter_ignored_when_anthropic_model_in_play(
    fake_genai, monkeypatch
):
    """Regression guard: matching runs in adapter order, so an
    Anthropic-shaped model must not accidentally hit the Gemini
    client. We ensure the Gemini client's recorder stayed empty."""
    monkeypatch.setenv("LLM_CONFIRM_TOKENS_EXACT", "1")
    # Also install a fake anthropic so the Anthropic adapter is chosen.
    fake_anthropic = types.ModuleType("anthropic")

    class _Resp:
        input_tokens = 11

    class _Msgs:
        def count_tokens(self, **_):
            return _Resp()

    class _Cli:
        def __init__(self, **_):
            self.messages = _Msgs()

    fake_anthropic.Anthropic = _Cli  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    n = estimate_tokens(_FakePrompt("hi"), type("M", (), {"model_id": "claude-sonnet-4-5"})())
    assert n == 11
    assert fake_genai == []


class _FakeResponse:
    def __init__(self, prompt, output=""):
        self.prompt = prompt
        self._chunks = [output] if output else []


class _FakeConversation:
    def __init__(self, responses=()):
        self.responses = list(responses)


def test_adapter_builds_multi_turn_contents_with_model_role(fake_genai):
    """Prior turns replay with Gemini's ``model`` role for assistant output."""
    adapter = GeminiAdapter()
    prior = _FakeResponse(_FakePrompt("first"), output="first reply")
    adapter.count(
        _FakePrompt("second"),
        _FakeGemini(),
        conversation=_FakeConversation([prior]),
    )
    contents = fake_genai[0]["contents"]
    assert contents == [
        {"role": "user", "parts": [{"text": "first"}]},
        {"role": "model", "parts": [{"text": "first reply"}]},
        {"role": "user", "parts": [{"text": "second"}]},
    ]


def test_adapter_system_stays_on_current_turn_only(fake_genai):
    """System prompt inlines into the current turn — not replayed per prior turn."""
    adapter = GeminiAdapter()
    prior = _FakeResponse(_FakePrompt("earlier"), output="earlier reply")
    adapter.count(
        _FakePrompt("now", system="you are helpful"),
        _FakeGemini(),
        conversation=_FakeConversation([prior]),
    )
    contents = fake_genai[0]["contents"]
    # The prior user turn must not carry the system text.
    assert contents[0] == {"role": "user", "parts": [{"text": "earlier"}]}
    # The current user turn gets system prepended to its parts.
    assert contents[-1] == {
        "role": "user",
        "parts": [{"text": "you are helpful"}, {"text": "now"}],
    }
