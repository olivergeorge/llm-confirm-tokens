"""End-to-end integration: the plugin gate aborts a live llm prompt.

Registers a ConfirmTokensGate with a stubbed ``ask`` via the same hookspec
path the plugin uses in production, then runs a prompt against a mock
model and verifies CancelPrompt propagates and the model is not called.
"""

from __future__ import annotations

import llm
import pytest
from llm import hookimpl
from llm.plugins import pm

from llm_confirm_tokens import ConfirmTokensGate


class _MockModel(llm.Model):
    """Minimal sync model that records whether execute() was reached."""

    model_id = "confirm-tokens-mock"

    def __init__(self):
        self.execute_called = False

    def execute(self, prompt, stream, response, conversation):
        self.execute_called = True
        yield "ok"


def _register_gate(gate, *, name="ConfirmTokensTestGate"):
    class GatePlugin:
        __name__ = name

        @hookimpl
        def register_prompt_gates(self, register):
            register(gate)

    instance = GatePlugin()
    pm.register(instance, name=name)
    return instance


def test_gate_blocks_live_prompt_when_user_says_no():
    model = _MockModel()
    gate = ConfirmTokensGate(
        threshold=0,
        tokens_fn=lambda p: 500,
        ask=lambda n: False,
    )
    try:
        _register_gate(gate)
        response = model.prompt("hello")
        with pytest.raises(llm.CancelPrompt):
            response.text()
        assert model.execute_called is False
    finally:
        pm.unregister(name="ConfirmTokensTestGate")


def test_gate_allows_live_prompt_when_user_says_yes():
    model = _MockModel()
    gate = ConfirmTokensGate(
        threshold=0,
        tokens_fn=lambda p: 500,
        ask=lambda n: True,
    )
    try:
        _register_gate(gate)
        response = model.prompt("hello")
        assert response.text() == "ok"
        assert model.execute_called is True
    finally:
        pm.unregister(name="ConfirmTokensTestGate")
