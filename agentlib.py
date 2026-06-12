"""Thin wrapper around the Cursor Python SDK shared by bootstrap.py and
orchestrator.py.

Keeps the SDK-specific concerns in one place: optional import with a friendly
message, API-key resolution, a persistent multi-turn agent context manager, and
a ``send + stream + wait`` helper that distinguishes the two failure modes the
SDK skill warns about (startup failure vs run failure).
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Iterator, Optional

DEFAULT_MODEL = os.environ.get("AUTO_RL_AGENT_MODEL", "composer-2.5")


class AgentUnavailable(RuntimeError):
    """Raised when the SDK or API key is missing; carries guidance text."""


def _import_sdk():
    try:
        import cursor_sdk  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise AgentUnavailable(
            "The Cursor Python SDK is not installed. Install it with:\n"
            "    uv pip install cursor-sdk   (or)   pip install cursor-sdk\n"
            f"(import error: {exc})"
        ) from exc
    from cursor_sdk import Agent, CursorAgentError, LocalAgentOptions

    return Agent, CursorAgentError, LocalAgentOptions


def require_api_key(explicit: Optional[str] = None) -> str:
    key = explicit or os.environ.get("CURSOR_API_KEY")
    if not key:
        raise AgentUnavailable(
            "CURSOR_API_KEY is not set. Get a key at "
            "https://cursor.com/dashboard/integrations and run:\n"
            '    export CURSOR_API_KEY="cursor_..."'
        )
    return key.strip()


def _message_text(message) -> str:
    """Best-effort extraction of assistant text from an SDK stream message."""
    if getattr(message, "type", None) != "assistant":
        return ""
    inner = getattr(message, "message", None)
    content = getattr(inner, "content", None) if inner is not None else None
    if not content:
        return ""
    out = []
    for block in content:
        if getattr(block, "type", None) == "text":
            out.append(getattr(block, "text", ""))
    return "".join(out)


@contextmanager
def open_agent(cwd, model: str = DEFAULT_MODEL, api_key: Optional[str] = None) -> Iterator["AgentSession"]:
    """Open a persistent local agent rooted at ``cwd`` for multi-turn use."""
    Agent, CursorAgentError, LocalAgentOptions = _import_sdk()
    key = require_api_key(api_key)
    with Agent.create(
        model=model,
        api_key=key,
        local=LocalAgentOptions(cwd=str(cwd)),
    ) as agent:
        yield AgentSession(agent, CursorAgentError)


class AgentSession:
    """Wraps a live agent; ``send`` streams output and returns the RunResult."""

    def __init__(self, agent, cursor_agent_error):
        self._agent = agent
        self._CursorAgentError = cursor_agent_error

    @property
    def agent_id(self) -> str:
        return getattr(self._agent, "agent_id", "?")

    def send(self, prompt: str, *, label: str = "", echo: bool = True):
        try:
            run = self._agent.send(prompt)
        except self._CursorAgentError as exc:  # did not start
            raise AgentUnavailable(
                f"agent run failed to start (auth/config/network): {exc}"
            ) from exc

        run_id = getattr(run, "id", "?")
        print(f"[agent] {label} run={run_id} agent={self.agent_id}", file=sys.stderr)
        try:
            for message in run.messages():
                if echo:
                    text = _message_text(message)
                    if text:
                        print(text, end="", flush=True)
        except self._CursorAgentError as exc:  # connection dropped mid-stream
            raise AgentUnavailable(f"agent stream failed: {exc}") from exc
        if echo:
            print()
        result = run.wait()
        return result


def result_failed(result) -> bool:
    """True if a *started* run ended in error (vs the finished happy path)."""
    return getattr(result, "status", None) == "error"
