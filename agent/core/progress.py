"""
Progress event emission for the reflexive agent pipeline.

Provides a ContextVar-based emitter so that stage progress (intent
classification, planning, tool execution, summarization) and structured
chart payloads can be surfaced to streaming clients as side-channel SSE
chunks WITHOUT changing the signature of run_reflexive_agent or its many
sub-handlers.

Design:
  - A ContextVar holds an optional async callback. Each asyncio task
    inherits a copy of the current context, so concurrent requests are
    isolated automatically -- no cross-request leakage.
  - emit_progress() and emit_chart() are no-ops when no emitter is set
    (non-streaming requests, tests, and all existing callers are unaffected).
  - Emission is best-effort: a failing callback never breaks the agent
    pipeline (graceful degradation per AGENTS.md).

Usage in main.py (streaming endpoint):
    queue = asyncio.Queue()
    def _on_progress(stage, label, detail=None):
        queue.put_nowait({"type": "progress", "stage": stage, "label": label, "detail": detail})
    def _on_chart(payload):
        queue.put_nowait({"type": "chart", "payload": payload})
    token = set_progress_emitter(_on_progress, _on_chart)
    try:
        answer = await run_agent(messages, auth_context)
    finally:
        reset_progress_emitter(token)
"""
import logging
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("hr_agent")

# Type alias for the progress callback.
# stage: short identifier (e.g. "intent", "planning", "executing", "summarizing")
# label: human-readable message for the UI status line
# detail: optional extra context (e.g. entity name, step index)
ProgressCallback = Callable[[str, str, Optional[str]], Any]

# Type alias for the chart callback.
# payload: Chart.js-shaped dict {type, title, labels, datasets}
ChartCallback = Callable[[Dict], Any]

_current_emitter: ContextVar[Optional[ProgressCallback]] = ContextVar(
    "hr_progress_emitter", default=None
)
_current_chart_emitter: ContextVar[Optional[ChartCallback]] = ContextVar(
    "hr_chart_emitter", default=None
)


def set_progress_emitter(
    progress_cb: Optional[ProgressCallback] = None,
    chart_cb: Optional[ChartCallback] = None,
) -> "contextvars.Token":
    """Install progress and/or chart callbacks in the current context.

    Returns a token to pass to reset_progress_emitter().
    """
    token = _current_emitter.set(progress_cb)
    if chart_cb is not None:
        _current_chart_emitter.set(chart_cb)
    return token


def reset_progress_emitter(token: "contextvars.Token") -> None:
    """Restore the previous emitter state (call in a finally block)."""
    _current_emitter.reset(token)


def emit_progress(stage: str, label: str, detail: Optional[str] = None) -> None:
    """Emit a progress event if an emitter is installed.

    Never raises -- progress is best-effort and must not affect the
    agent's correctness or result.
    """
    cb = _current_emitter.get()
    if cb is None:
        return
    try:
        cb(stage, label, detail)
    except Exception as exc:  # noqa: BLE001 - best-effort, never propagate
        logger.debug("emit_progress: callback failed for stage=%s: %s", stage, exc)


def emit_chart(payload: Dict) -> None:
    """Emit a structured chart payload (Chart.js shape) if a chart emitter is installed.

    The payload is {type, title, labels, datasets}. Only emitted for clients
    that opt in via the X-HR-Client header (custom UI). LibreChat and other
    clients receive the markdown artifact instead.

    Never raises -- best-effort.
    """
    cb = _current_chart_emitter.get()
    if cb is None:
        return
    try:
        cb(payload)
    except Exception as exc:  # noqa: BLE001 - best-effort, never propagate
        logger.debug("emit_chart: callback failed: %s", exc)
