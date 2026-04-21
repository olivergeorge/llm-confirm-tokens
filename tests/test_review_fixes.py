"""Tests that pin the behaviour changes from the Gemini code review.

These are kept in their own file so the review motivation stays
readable: each test corresponds to a specific review finding that the
gate used to get wrong.
"""

from __future__ import annotations

import sys
import types

import llm
import pytest

from llm_confirm_tokens import (
    ConfirmTokensGate,
    _ask_via_tty,
    estimate_tokens,
    estimate_tokens_detailed,
)


class _FakePrompt:
    def __init__(self, prompt="hello"):
        self.prompt = prompt
        self.system = None
        self.fragments = []
        self.system_fragments = []
        self.attachments = []
        self.tools = []
        self.tool_results = []
        self.schema = None


class _FakeClaude:
    model_id = "claude-sonnet-4-5"


def test_no_tty_fails_closed_not_open(monkeypatch, capsys):
    """No-tty environments must *decline* rather than auto-approve.

    The v0 behaviour was to print a warning and return True, which
    defeated the point of installing the plugin on a headless box.
    Fail-closed + a clear stderr note about LLM_CONFIRM_TOKENS_YES is
    the safer default; scripts that actually want auto-approval set
    the env var explicitly.
    """

    def _always_oserror(*_a, **_kw):
        raise OSError("no tty")

    monkeypatch.setattr("builtins.open", _always_oserror)
    proceed = _ask_via_tty(1234)
    assert proceed is False
    err = capsys.readouterr().err
    assert "no tty" in err
    assert "LLM_CONFIRM_TOKENS_YES" in err


def test_no_tty_gate_raises_cancelprompt(monkeypatch):
    """End-to-end: the gate converts the False answer into CancelPrompt."""

    def _always_oserror(*_a, **_kw):
        raise OSError("no tty")

    monkeypatch.setattr("builtins.open", _always_oserror)
    gate = ConfirmTokensGate(threshold=0, tokens_fn=lambda p, m: 500)
    with pytest.raises(llm.CancelPrompt):
        gate.check(_FakePrompt(), model=None)


def test_ask_message_humanises_heuristic_with_estimate_label(monkeypatch):
    """Heuristic counts show a rounded number and an ``(estimate)`` tag
    so the user can tell an estimate from an exact provider count at a
    glance without confusing false precision."""
    written: list[str] = []

    class _FakeTTY:
        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def write(self, s):
            written.append(s)

        def flush(self):
            pass

        def readline(self):
            return "y\n"

    monkeypatch.setattr("builtins.open", lambda *a, **k: _FakeTTY())
    _ask_via_tty(1234, source="heuristic")
    combined = "".join(written)
    assert "1.2k input tokens (estimate)" in combined
    # Heuristic path must not show the raw four-digit number.
    assert "1,234" not in combined


def test_ask_message_keeps_exact_counts_precise(monkeypatch):
    """Exact counts stay at full precision and carry the provider name,
    so the user sees the gate is relying on the provider's tokeniser."""
    written: list[str] = []

    class _FakeTTY:
        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def write(self, s):
            written.append(s)

        def flush(self):
            pass

        def readline(self):
            return "y\n"

    monkeypatch.setattr("builtins.open", lambda *a, **k: _FakeTTY())
    _ask_via_tty(1234, source="anthropic")
    combined = "".join(written)
    assert "1,234 input tokens (anthropic)" in combined
    assert "estimate" not in combined


def test_adapter_failure_emits_stderr_warning(monkeypatch, capsys):
    """When an exact adapter fails, the user must see a clear stderr
    note so silent fallback doesn't undermine trust in the gate."""
    monkeypatch.setenv("LLM_CONFIRM_TOKENS_EXACT", "1")

    fake_module = types.ModuleType("anthropic")

    def _Anthropic(**_kw):
        class _Broken:
            class messages:
                @staticmethod
                def count_tokens(**_):
                    raise RuntimeError("api is sad today")

        return _Broken()

    fake_module.Anthropic = _Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    n = estimate_tokens(_FakePrompt(), _FakeClaude())
    assert n > 0  # heuristic fell through
    err = capsys.readouterr().err
    assert "anthropic exact-count failed" in err
    assert "RuntimeError" in err
    assert "using heuristic" in err


def test_detailed_estimator_returns_source_label_for_exact(monkeypatch):
    """The detailed API returns the provider name so the gate can show it."""
    monkeypatch.setenv("LLM_CONFIRM_TOKENS_EXACT", "1")

    fake_module = types.ModuleType("anthropic")

    class _Resp:
        input_tokens = 42

    class _Cli:
        def __init__(self, **_):
            self.messages = types.SimpleNamespace(count_tokens=lambda **_k: _Resp())

    fake_module.Anthropic = _Cli  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    low, high, source = estimate_tokens_detailed(_FakePrompt(), _FakeClaude())
    assert low == 42 and high == 42
    assert source == "anthropic"


def test_detailed_estimator_returns_heuristic_label_when_no_exact():
    """Without exact mode the source is ``heuristic`` — matches the prompt
    prefix shown to the user (``~N``)."""
    low, high, source = estimate_tokens_detailed(_FakePrompt(), model=None)
    assert low > 0 and high >= low
    assert source == "heuristic"


def test_ask_prefers_stdin_stderr_when_both_are_ttys(monkeypatch):
    """When both stdin and stderr are ttys, the prompt goes through
    the standard streams — no ``/dev/tty`` open, no seek trap."""
    writes: list[str] = []

    class _TTY:
        def isatty(self):
            return True

        def write(self, s):
            writes.append(s)

        def flush(self):
            pass

        def readline(self):
            return "y\n"

    monkeypatch.setattr(sys, "stdin", _TTY())
    monkeypatch.setattr(sys, "stderr", _TTY())

    def _fail_open(*_a, **_kw):
        raise AssertionError("should not have opened a device file")

    monkeypatch.setattr("builtins.open", _fail_open)

    assert _ask_via_tty(500, source="heuristic") is True
    combined = "".join(writes)
    assert "500 input tokens (estimate)" in combined


def test_ask_never_uses_readwrite_plus_mode(monkeypatch):
    """Regression guard for the macOS ``/dev/tty`` seek bug.

    Opening ``/dev/tty`` with ``'r+'`` triggers an implicit seek; macOS
    raises ``UnsupportedOperation: File or stream is not seekable``.
    This test stands in as a contract that the fallback path opens
    read and write handles separately, never with ``+``.
    """
    modes_used: list[str] = []

    class _NonTTY:
        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", _NonTTY())
    monkeypatch.setattr(sys, "stderr", _NonTTY())

    class _FakeTTY:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def write(self, s):
            pass

        def flush(self):
            pass

        def readline(self):
            return "y\n"

    def _fake_open(path, mode="r", *_a, **_kw):
        modes_used.append(mode)
        return _FakeTTY()

    monkeypatch.setattr("builtins.open", _fake_open)
    _ask_via_tty(100)
    assert modes_used  # fallback path was taken
    assert all("+" not in m for m in modes_used), modes_used
