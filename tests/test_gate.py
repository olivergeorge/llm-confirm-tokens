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


def test_count_prompt_tokens_image_attachment_falls_back_without_dimensions():
    """Image bytes we can't parse for dimensions fall back to the flat cost.

    A fake JPEG with no Start-Of-Frame marker is unparseable, so the heuristic
    can't apply a per-provider tile formula and uses the ``_IMAGE_TOKENS``
    baseline (258). The key invariant is that we don't fall through to the
    bytes-as-text path (~25 000 tokens) for 100 KB of opaque bytes.
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
    assert 150 < (n - bare) < 400


def _png_with_dims(width: int, height: int) -> bytes:
    """Return the smallest byte string that parses as PNG with given dims."""
    import struct

    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0dIHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
        + b"\x00\x00\x00\x00"
    )


class _ModelStub:
    def __init__(self, model_id: str):
        self.model_id = model_id


def test_image_small_under_384_uses_flat_cost():
    """Images ≤ 384 on both sides don't tile on any provider — flat 258."""

    class _FakeImage:
        def __init__(self, content):
            self.content = content
            self.path = None
            self.url = None
            self.type = "image/png"

    prompt = _FakePrompt("hi", attachments=[_FakeImage(_png_with_dims(300, 300))])
    bare = count_prompt_tokens(_FakePrompt("hi"))
    # Gemini (default), Anthropic, OpenAI should all sit near 258 for a
    # tiny image: Gemini's flat cost, Anthropic's 300*300/750=120, OpenAI's
    # 85+170=255 for a single sub-512 tile.
    for model in (None, _ModelStub("claude-3-5-sonnet"), _ModelStub("gpt-4o")):
        n = count_prompt_tokens(prompt, model)
        assert 100 <= (n - bare) <= 300, (model, n - bare)


def test_image_large_applies_gemini_tiling_by_default():
    """A retina-screenshot-sized PNG gets Gemini's tile formula (≫ 258)."""

    class _FakeImage:
        def __init__(self, content):
            self.content = content
            self.path = None
            self.url = None
            self.type = "image/png"

    # 1024×768 → tile_size=512, tiles=2×2=4, tokens=4*258=1032.
    prompt = _FakePrompt("hi", attachments=[_FakeImage(_png_with_dims(1024, 768))])
    bare = count_prompt_tokens(_FakePrompt("hi"))
    n = count_prompt_tokens(prompt)  # no model → gemini default
    assert 900 <= (n - bare) <= 1100


def test_image_large_uses_anthropic_pixel_rate_for_claude():
    """Claude models are scored at ~(W×H)/750, not Gemini tile tokens."""

    class _FakeImage:
        def __init__(self, content):
            self.content = content
            self.path = None
            self.url = None
            self.type = "image/png"

    # 1024×768 → (1024*768)/750 ≈ 1048 tokens on Anthropic.
    prompt = _FakePrompt("hi", attachments=[_FakeImage(_png_with_dims(1024, 768))])
    bare = count_prompt_tokens(_FakePrompt("hi"))
    n = count_prompt_tokens(prompt, _ModelStub("claude-3-5-sonnet"))
    assert 950 <= (n - bare) <= 1150


def test_image_large_uses_openai_tile_formula_for_gpt():
    """GPT-4o models are scored at 85 base + 170 per 512×512 high-detail tile."""

    class _FakeImage:
        def __init__(self, content):
            self.content = content
            self.path = None
            self.url = None
            self.type = "image/png"

    # 1024×768 scaled so shortest side is 768 → 1024×768 (already). Tiles
    # = ceil(1024/512) * ceil(768/512) = 2*2 = 4. Tokens = 85 + 170*4 = 765.
    prompt = _FakePrompt("hi", attachments=[_FakeImage(_png_with_dims(1024, 768))])
    bare = count_prompt_tokens(_FakePrompt("hi"))
    n = count_prompt_tokens(prompt, _ModelStub("gpt-4o"))
    assert 700 <= (n - bare) <= 850


def test_image_jpeg_dimensions_parse():
    """The JPEG SOF0 path extracts width/height without a PIL dependency."""
    import struct

    # SOI, APP0 (minimal), SOF0 with 800×600, then EOI. SOF0 payload:
    # length=17, precision=8, H=600, W=800, Nf=3, components(9 bytes).
    jpeg = (
        b"\xff\xd8"
        + b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        + b"\xff\xc0\x00\x11\x08"
        + struct.pack(">HH", 600, 800)
        + b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
        + b"\xff\xd9"
    )

    class _FakeImage:
        def __init__(self, content):
            self.content = content
            self.path = None
            self.url = None
            self.type = "image/jpeg"

    prompt = _FakePrompt("hi", attachments=[_FakeImage(jpeg)])
    bare = count_prompt_tokens(_FakePrompt("hi"))
    # 800×600 Gemini: tile_size=400, tiles=2×2=4, 1032 tokens.
    n = count_prompt_tokens(prompt)
    assert 900 <= (n - bare) <= 1100


def test_drift_warn_fires_when_heuristic_diverges(monkeypatch, capsys):
    """LLM_CONFIRM_TOKENS_DRIFT_WARN logs a stderr notice on significant drift."""
    from llm_confirm_tokens import _maybe_warn_drift

    monkeypatch.setenv("LLM_CONFIRM_TOKENS_DRIFT_WARN", "10")
    _maybe_warn_drift(actual=1000, heuristic=100, source="anthropic")
    err = capsys.readouterr().err
    assert "heuristic" in err and "anthropic" in err and "best guess" in err


def test_drift_warn_silent_when_unset(monkeypatch, capsys):
    """With the env var unset, drift comparison is silent (opt-in only)."""
    from llm_confirm_tokens import _maybe_warn_drift

    monkeypatch.delenv("LLM_CONFIRM_TOKENS_DRIFT_WARN", raising=False)
    _maybe_warn_drift(actual=1000, heuristic=100, source="anthropic")
    assert capsys.readouterr().err == ""


def test_drift_warn_silent_when_within_threshold(monkeypatch, capsys):
    """No notice when heuristic is within the configured percentage band."""
    from llm_confirm_tokens import _maybe_warn_drift

    monkeypatch.setenv("LLM_CONFIRM_TOKENS_DRIFT_WARN", "50")
    _maybe_warn_drift(actual=100, heuristic=90, source="anthropic")
    assert capsys.readouterr().err == ""


def test_estimate_tokens_detailed_stashes_heuristic_on_prompt(monkeypatch):
    """The public counter records the heuristic on the prompt for later compare."""
    from llm_confirm_tokens import estimate_tokens_detailed

    monkeypatch.delenv("LLM_CONFIRM_TOKENS_EXACT", raising=False)
    prompt = _FakePrompt("hello there")
    estimate_tokens_detailed(prompt, model=None)
    assert getattr(prompt, "_confirm_tokens_heuristic", None) is not None


def test_after_log_to_db_warns_on_post_response_drift(monkeypatch, capsys):
    """Billed vs heuristic drift is reported via the after_log_to_db hook."""
    from llm_confirm_tokens import after_log_to_db

    monkeypatch.setenv("LLM_CONFIRM_TOKENS_DRIFT_WARN", "25")
    prompt = _FakePrompt("hello")
    prompt._confirm_tokens_heuristic = 258  # pre-flight estimate
    prompt._confirm_tokens_model_id = "gemini-flash-latest"

    class _Response:
        input_tokens = 1080  # what the provider actually billed

    _Response.prompt = prompt  # type: ignore[attr-defined]
    after_log_to_db(_Response(), db=None)
    err = capsys.readouterr().err
    assert "heuristic 258" in err and "under-counts" in err
    assert "gemini-flash-latest billed" in err


def test_after_log_to_db_silent_without_stash(monkeypatch, capsys):
    """Without a pre-flight stash (gate didn't run), the hook is a no-op."""
    from llm_confirm_tokens import after_log_to_db

    monkeypatch.setenv("LLM_CONFIRM_TOKENS_DRIFT_WARN", "10")

    class _Response:
        input_tokens = 9999
        prompt = _FakePrompt("hi")  # no stash attribute set

    after_log_to_db(_Response(), db=None)
    assert capsys.readouterr().err == ""


def test_after_log_to_db_silent_when_threshold_unset(monkeypatch, capsys):
    """Without the opt-in env var, the post-response path stays quiet."""
    from llm_confirm_tokens import after_log_to_db

    monkeypatch.delenv("LLM_CONFIRM_TOKENS_DRIFT_WARN", raising=False)
    prompt = _FakePrompt("hello")
    prompt._confirm_tokens_heuristic = 10
    prompt._confirm_tokens_model_id = "claude-3-5-sonnet"

    class _Response:
        input_tokens = 10000

    _Response.prompt = prompt  # type: ignore[attr-defined]
    after_log_to_db(_Response(), db=None)
    assert capsys.readouterr().err == ""


def test_count_prompt_tokens_pdf_regex_wins_over_byte_fallback():
    """When the regex sees real pages, the byte-size fallback must not inflate.

    Regression: a 24-page 7MB PDF previously estimated at 146 pages because
    ``max(regex, bytes // 50_000)`` let the byte floor override an accurate
    page count on content-rich files. The composition is now
    ``regex or bytes``, so a readable PDF uses its true page count.
    """

    class _FakePdf:
        def __init__(self, content):
            self.content = content
            self.path = None
            self.url = None
            self.type = "application/pdf"

    # 5 Page markers in a 2MB PDF. Byte estimate would be 2_000_000 // 50_000
    # = 40 pages; true count is 5. Pre-fix would have picked 40.
    pdf_bytes = (
        b"%PDF-1.4\n"
        + (b"/Type /Page\n" * 5)
        + b"\x00" * 2_000_000
        + b"%%EOF"
    )
    prompt = _FakePrompt("hi", attachments=[_FakePdf(pdf_bytes)])
    n = count_prompt_tokens(prompt)
    bare = count_prompt_tokens(_FakePrompt("hi"))
    # 5 pages * 258 tokens ≈ 1290. The old 40-page byte floor would give
    # ~10 320; the band below locks out that regression.
    assert 1000 < (n - bare) < 2000


def test_count_prompt_tokens_pdf_byte_fallback_when_regex_blind():
    """Compressed PDFs (regex finds nothing) still get a bytes-based estimate."""

    class _FakePdf:
        def __init__(self, content):
            self.content = content
            self.path = None
            self.url = None
            self.type = "application/pdf"

    # No Page markers — simulates a PDF with object streams.
    pdf_bytes = b"%PDF-1.4\n" + b"\x00" * 500_000 + b"%%EOF"
    prompt = _FakePrompt("hi", attachments=[_FakePdf(pdf_bytes)])
    n = count_prompt_tokens(prompt)
    bare = count_prompt_tokens(_FakePrompt("hi"))
    # 500_000 // 50_000 = 10 pages * 258 ≈ 2580. Must be non-zero since we
    # know there's a PDF attached even if we couldn't read the page count.
    assert 2000 < (n - bare) < 3500


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
