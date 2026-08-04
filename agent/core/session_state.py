"""Lightweight session state for follow-up actions (DST-lite)."""
from typing import Dict, Any

_SESSION_STATE: Dict[str, Dict] = {}


def _get_session_key(auth_context: Any) -> str:
    if auth_context and auth_context.email:
        return auth_context.email.lower()
    if auth_context and auth_context.emp_id:
        return str(auth_context.emp_id)
    return "anonymous"


def load_session_state(auth_context: Any) -> Dict:
    key = _get_session_key(auth_context)
    return _SESSION_STATE.get(key, {})


def save_session_state(auth_context: Any, state: Dict):
    key = _get_session_key(auth_context)
    _SESSION_STATE[key] = state