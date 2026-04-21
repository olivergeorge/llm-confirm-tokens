"""llm-confirm-tokens: interactive token-count confirmation for llm prompts.

Registers a :class:`PromptGate` via the ``register_prompt_gates`` hookspec.
When enabled (``LLM_CONFIRM_TOKENS=1``) the gate counts tokens on the
resolved prompt and, if the total is at or above the configured threshold,
prints ``Total tokens: N. Proceed? [Y/n]:`` to ``/dev/tty`` and waits for
the user. Anything other than an empty response or ``y``/``yes`` raises
:class:`llm.CancelPrompt`, aborting the prompt before the upstream API is
called.

Opt-in rather than on-by-default so installing the plugin does not change
the behaviour of any existing script.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

import llm
from llm import hookimpl

__all__ = [
    "ConfirmTokensGate",
    "count_prompt_tokens",
    "register_prompt_gates",
]


_TRUTHY = ("1", "true", "yes", "on")


def _is_enabled() -> bool:
    return os.environ.get("LLM_CONFIRM_TOKENS", "").strip().lower() in _TRUTHY


def _assume_yes() -> bool:
    return os.environ.get("LLM_CONFIRM_TOKENS_YES", "").strip().lower() in _TRUTHY


def _threshold() -> int:
    raw = os.environ.get("LLM_CONFIRM_TOKENS_THRESHOLD", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _prompt_text(prompt: llm.Prompt) -> str:
    """Flatten the user-visible text payload of a prompt for token counting.

    Binary attachments are out of scope — plugins that want a precise
    image-token estimate can register their own gate. The heuristic here
    is deliberately rough so the "proceed?" signal is directional rather
    than authoritative.
    """
    parts: list[str] = []
    system = getattr(prompt, "system", None)
    if system:
        parts.append(system)
    for sf in getattr(prompt, "system_fragments", []) or []:
        parts.append(str(sf))
    body = getattr(prompt, "prompt", None) or getattr(prompt, "_prompt", None)
    if body:
        parts.append(body)
    for f in getattr(prompt, "fragments", []) or []:
        parts.append(str(f))
    return "\n".join(p for p in parts if p)


def count_prompt_tokens(prompt: llm.Prompt) -> int:
    """Return an estimated token count for ``prompt``.

    Uses tiktoken's ``cl100k_base`` encoding when available (accurate for
    OpenAI models and a reasonable approximation for others); falls back
    to ``max(1, len(text) // 4)`` when tiktoken is not installed so the
    plugin still works on any Python environment.
    """
    text = _prompt_text(prompt)
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _ask_via_tty(tokens: int) -> bool:
    """Prompt the user on ``/dev/tty`` and return True to proceed.

    Falls back to stderr/stdin when ``/dev/tty`` is not available
    (e.g. CI, sandboxed scripts). If no interactive input is possible
    at all, returns True so non-interactive scripts are not blocked —
    users who want strict blocking should gate their scripts on the
    env var directly.
    """
    message = f"Total tokens: {tokens:,}. Proceed? [Y/n]: "
    try:
        with open("/dev/tty", "r+") as tty:
            tty.write(message)
            tty.flush()
            answer = tty.readline()
    except OSError:
        if not sys.stdin.isatty():
            sys.stderr.write(f"llm-confirm-tokens: {tokens:,} tokens (no tty; proceeding)\n")
            return True
        sys.stderr.write(message)
        sys.stderr.flush()
        answer = sys.stdin.readline()
    answer = (answer or "").strip().lower()
    return answer in ("", "y", "yes")


class ConfirmTokensGate:
    """Prompt gate that confirms with the user before large prompts are sent.

    Inject ``tokens_fn`` and ``ask`` for tests — in production the defaults
    (tiktoken-based counting and ``/dev/tty`` prompting) are used.
    """

    def __init__(
        self,
        *,
        threshold: int = 0,
        tokens_fn: Callable[[llm.Prompt], int] | None = None,
        ask: Callable[[int], bool] | None = None,
    ) -> None:
        self.threshold = threshold
        self._tokens_fn = tokens_fn or count_prompt_tokens
        self._ask = ask or _ask_via_tty

    def check(self, prompt: llm.Prompt, model: Any) -> None:
        tokens = self._tokens_fn(prompt)
        if tokens < self.threshold:
            return
        if _assume_yes():
            return
        if not self._ask(tokens):
            raise llm.CancelPrompt(f"user declined {tokens:,} token prompt")


@hookimpl
def register_prompt_gates(register: Any) -> None:
    """Register the confirm-tokens gate when the plugin is enabled.

    Keyed on the ``LLM_CONFIRM_TOKENS`` env var so the plugin is a no-op
    for users who have installed it but not opted in.
    """
    if not _is_enabled():
        return
    register(ConfirmTokensGate(threshold=_threshold()))
