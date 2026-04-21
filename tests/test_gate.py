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

    def __init__(self, prompt="", system=None, fragments=(), system_fragments=()):
        self.prompt = prompt
        self.system = system
        self.fragments = list(fragments)
        self.system_fragments = list(system_fragments)


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
