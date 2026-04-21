"""Unit tests for ConfirmTokensGate and the hookspec plumbing.

Exercises the gate directly with injected ``tokens_fn`` / ``ask`` rather
than monkey-patching ``/dev/tty``, so the tests work identically in CI
and on a developer laptop.
"""

from __future__ import annotations

import llm
import pytest

from llm_confirm_tokens import (
    ConfirmTokensGate,
    count_prompt_tokens,
    register_prompt_gates,
)


class _FakePrompt:
    """Duck-typed llm.Prompt substitute: only the attrs the plugin reads."""

    def __init__(
        self,
        prompt="",
        system=None,
        fragments=(),
        system_fragments=(),
        attachments=(),
        tools=(),
        tool_results=(),
        schema=None,
    ):
        self.prompt = prompt
        self.system = system
        self.fragments = list(fragments)
        self.system_fragments = list(system_fragments)
        self.attachments = list(attachments)
        self.tools = list(tools)
        self.tool_results = list(tool_results)
        self.schema = schema


class _FakeAttachment:
    def __init__(self, *, content=None, path=None, url=None):
        self.content = content
        self.path = path
        self.url = url


def test_gate_under_threshold_returns_without_asking():
    """Counts below threshold must not prompt the user at all."""
    asked = []
    gate = ConfirmTokensGate(
        threshold=100,
        tokens_fn=lambda p: 10,
        ask=lambda n: asked.append(n) or True,
    )
    gate.check(_FakePrompt("hi"), model=None)
    assert asked == []


def test_gate_at_threshold_asks_and_proceeds_on_yes():
    """At or above threshold, ask the user; 'yes' allows the prompt."""
    gate = ConfirmTokensGate(
        threshold=50,
        tokens_fn=lambda p: 50,
        ask=lambda n: True,
    )
    gate.check(_FakePrompt("hi"), model=None)  # must not raise


def test_gate_raises_cancelprompt_on_no():
    """A negative answer raises CancelPrompt and includes the token count."""
    gate = ConfirmTokensGate(
        threshold=0,
        tokens_fn=lambda p: 12345,
        ask=lambda n: False,
    )
    with pytest.raises(llm.CancelPrompt, match="12,345"):
        gate.check(_FakePrompt("hi"), model=None)


def test_gate_assume_yes_env_bypasses_prompt(monkeypatch):
    """LLM_CONFIRM_TOKENS_YES=1 short-circuits asking — useful in scripts."""
    monkeypatch.setenv("LLM_CONFIRM_TOKENS_YES", "1")
    asked = []
    gate = ConfirmTokensGate(
        threshold=0,
        tokens_fn=lambda p: 999,
        ask=lambda n: asked.append(n) or False,  # would deny if consulted
    )
    gate.check(_FakePrompt("hi"), model=None)
    assert asked == []


def test_register_prompt_gates_is_noop_when_disabled(monkeypatch):
    """Without LLM_CONFIRM_TOKENS set, the plugin registers no gates."""
    monkeypatch.delenv("LLM_CONFIRM_TOKENS", raising=False)
    registered: list[object] = []
    register_prompt_gates(register=registered.append)
    assert registered == []


def test_register_prompt_gates_registers_when_enabled(monkeypatch):
    """With LLM_CONFIRM_TOKENS=1 a ConfirmTokensGate is registered."""
    monkeypatch.setenv("LLM_CONFIRM_TOKENS", "1")
    monkeypatch.setenv("LLM_CONFIRM_TOKENS_THRESHOLD", "42")
    registered: list[object] = []
    register_prompt_gates(register=registered.append)
    assert len(registered) == 1
    gate = registered[0]
    assert isinstance(gate, ConfirmTokensGate)
    assert gate.threshold == 42


def test_count_prompt_tokens_flattens_system_and_fragments():
    """The counter reads system + fragments so gating reflects the full payload."""
    prompt = _FakePrompt(
        prompt="hello",
        system="you are a helpful assistant",
        fragments=["some retrieved document"],
        system_fragments=["an additional system fragment"],
    )
    n = count_prompt_tokens(prompt)
    # Bare text alone would be ~1 token. Including everything must be more.
    assert n > count_prompt_tokens(_FakePrompt("hello"))


def test_count_prompt_tokens_includes_text_attachment_content(tmp_path):
    """Text attachments on disk are decoded and tokenised — the README.md case."""
    file = tmp_path / "notes.md"
    file.write_text("The quick brown fox jumps over the lazy dog. " * 100)

    bare = count_prompt_tokens(_FakePrompt("hi"))
    with_attachment = count_prompt_tokens(
        _FakePrompt("hi", attachments=[_FakeAttachment(path=str(file))])
    )
    # The attachment adds hundreds of tokens; "hi" alone is ~1.
    assert with_attachment - bare > 100


def test_count_prompt_tokens_handles_inline_byte_attachment():
    """Attachments whose bytes are in-memory are tokenised from .content."""
    prompt = _FakePrompt(
        "hi",
        attachments=[_FakeAttachment(content=b"a" * 4000)],
    )
    assert count_prompt_tokens(prompt) > count_prompt_tokens(_FakePrompt("hi"))


def test_count_prompt_tokens_image_attachment_uses_fixed_per_image_cost():
    """Image attachments cost a fixed per-image estimate, not a bytes heuristic.

    Gemini charges ~258 tokens per image regardless of resolution, so a large
    JPEG should not inflate the estimate as if it were bytes of text.
    """

    class _FakeImage:
        def __init__(self, content):
            self.content = content
            self.path = None
            self.url = None
            self.type = "image/jpeg"

    big_image = b"\xff\xd8\xff\xe0" + b"\xaa" * 100_000
    prompt = _FakePrompt("hi", attachments=[_FakeImage(big_image)])
    n = count_prompt_tokens(prompt)
    bare = count_prompt_tokens(_FakePrompt("hi"))
    # The image must cost around one image's worth of tokens (~250), not
    # ~25000 that a naive bytes//4 would produce.
    assert 150 < (n - bare) < 400


def test_count_prompt_tokens_pdf_attachment_scales_with_pages():
    """PDFs cost per page — a 10-page PDF must estimate ~10 * per-page tokens."""

    class _FakePdf:
        def __init__(self, content):
            self.content = content
            self.path = None
            self.url = None
            self.type = "application/pdf"

    # Synthesise ten `/Type /Page` markers so the page counter picks 10.
    pdf_bytes = b"%PDF-1.4\n" + (b"/Type /Page\n" * 10) + b"%%EOF"
    prompt = _FakePrompt("hi", attachments=[_FakePdf(pdf_bytes)])
    n = count_prompt_tokens(prompt)
    bare = count_prompt_tokens(_FakePrompt("hi"))
    # 10 pages * 258 tokens/page ≈ 2580. Allow a band around it.
    assert 2000 < (n - bare) < 3500


def test_count_prompt_tokens_unknown_binary_falls_back():
    """When nothing about the blob is recognisable, use the flat-binary number."""
    prompt = _FakePrompt(
        "hi",
        attachments=[_FakeAttachment(content=b"\x01\x02\x03\x04" * 3000)],
    )
    n = count_prompt_tokens(prompt)
    bare = count_prompt_tokens(_FakePrompt("hi"))
    # Flat 300 band: bytes // 4 would yield thousands; the fallback keeps it bounded.
    assert 200 < (n - bare) < 500


def test_count_prompt_tokens_url_attachment_does_not_fetch(monkeypatch):
    """URL-only attachments must not trigger any HTTP call from the counter."""
    import httpx

    def _fail(*_a, **_kw):
        raise AssertionError("counter must not fetch URLs")

    monkeypatch.setattr(httpx, "get", _fail)
    monkeypatch.setattr(httpx, "head", _fail)
    prompt = _FakePrompt(
        "hi",
        attachments=[_FakeAttachment(url="https://example.com/photo.jpg")],
    )
    count_prompt_tokens(prompt)  # must not raise


def test_count_prompt_tokens_missing_file_uses_flat_estimate(tmp_path):
    """A path that doesn't exist falls back rather than crashing the gate."""
    prompt = _FakePrompt(
        "hi",
        attachments=[_FakeAttachment(path=str(tmp_path / "does-not-exist"))],
    )
    assert count_prompt_tokens(prompt) > count_prompt_tokens(_FakePrompt("hi"))


def test_counter_survives_strings_that_contain_special_tokens():
    """Strings with ``<|endoftext|>`` etc must not crash count_prompt_tokens.

    tiktoken raises ValueError by default on these strings; we pass
    ``disallowed_special=()`` so user prompts that mention GPT internals
    can still be counted. Regression guard for the Gemini review finding.
    """
    prompt = _FakePrompt("help me understand <|endoftext|> and <|im_start|>")
    count_prompt_tokens(prompt)  # must not raise


def test_count_prompt_tokens_includes_tools_and_schema():
    """Tool JSON schemas and structured-output schemas add to the estimate."""

    class _FakeTool:
        name = "search"
        description = "search the index for matching documents"
        input_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    class _FakeToolResult:
        output = "result row 1, result row 2, result row 3"

    bare = count_prompt_tokens(_FakePrompt("hi"))
    with_tools = count_prompt_tokens(
        _FakePrompt(
            "hi",
            tools=[_FakeTool()],
            tool_results=[_FakeToolResult()],
            schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        )
    )
    assert with_tools > bare
