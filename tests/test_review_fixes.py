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


def test_ask_message_marks_heuristic_with_tilde(monkeypatch):
    """Heuristic counts are displayed with a ``~`` prefix so the user
    can tell an estimate from an exact provider count."""
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
    assert any("~1,234" in w for w in written)
    assert not any("(heuristic)" in w for w in written)


def test_ask_message_marks_exact_with_provider(monkeypatch):
    """Exact counts get the provider name appended so the user sees
    the gate is relying on the provider's tokeniser, not the heuristic."""
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
    assert "1,234 (anthropic)" in combined
    assert "~" not in combined


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

    tokens, source = estimate_tokens_detailed(_FakePrompt(), _FakeClaude())
    assert tokens == 42
    assert source == "anthropic"


def test_detailed_estimator_returns_heuristic_label_when_no_exact():
    """Without exact mode the source is ``heuristic`` — matches the prompt
    prefix shown to the user (``~N``)."""
    tokens, source = estimate_tokens_detailed(_FakePrompt(), model=None)
    assert tokens > 0
    assert source == "heuristic"
