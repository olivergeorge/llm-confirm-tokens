"""Test-suite fixtures.

Tests assume a clean ``LLM_CONFIRM_TOKENS_*`` environment so they
behave the same in CI and on a developer laptop where the user has
the plugin enabled in their shell. Without this autouse fixture, a
shell-level ``LLM_CONFIRM_TOKENS_YES=1`` makes the gate skip its
``ask`` callback entirely, and ``LLM_CONFIRM_TOKENS_DRIFT_WARN`` makes
adapter tests print drift notices that aren't part of the assertion.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_plugin_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every ``LLM_CONFIRM_TOKENS*`` env var before each test runs.

    Tests that need a flag set should use ``monkeypatch.setenv`` —
    pytest will undo it at teardown, so isolation between tests is
    preserved.
    """
    for key in list(os.environ):
        if key.startswith("LLM_CONFIRM_TOKENS"):
            monkeypatch.delenv(key, raising=False)
