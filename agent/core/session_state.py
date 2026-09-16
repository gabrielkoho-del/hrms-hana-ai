"""Lightweight session state for follow-up actions (DST-lite).

Supports:
- Standard session state (last_query, last_intent, etc.)
- Pending action tracking: when a field is mentioned but no value provided,
  the injector stores pending_field so the next turn can bind the value.
"""
from typing import Dict, Any, Optional, List

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


# =============================================================================
# Pending Action Tracking
# =============================================================================

def save_pending_action(
    auth_context: Any,
    action: str,
    entity: str,
    field: str,
    code_type: Optional[str] = None,
) -> None:
    """Save a pending action when field mentioned but no value provided.

    Args:
        auth_context: Auth context for session key
        action: Action type (e.g., "update", "create")
        entity: Entity name (e.g., "employee_general")
        field: Field name (e.g., "marital_status")
        code_type: Optional codesetup type for value resolution (e.g., "MARITAL_STATUS")
    """
    key = _get_session_key(auth_context)
    _SESSION_STATE[key] = _SESSION_STATE.get(key, {})
    _SESSION_STATE[key]["pending_action"] = {
        "action": action,
        "entity": entity,
        "field": field,
        "code_type": code_type,
    }


def load_pending_action(auth_context: Any) -> Optional[Dict]:
    """Load pending action if one exists for this session."""
    state = load_session_state(auth_context)
    return state.get("pending_action")


def clear_pending_action(auth_context: Any) -> None:
    """Clear pending action after it's been fulfilled or abandoned."""
    key = _get_session_key(auth_context)
    if key in _SESSION_STATE and "pending_action" in _SESSION_STATE[key]:
        del _SESSION_STATE[key]["pending_action"]


# =============================================================================
# Pending Leave Confirmation Tracking
# =============================================================================

def save_pending_leave_confirmation(
    auth_context: Any,
    entity: str,
    record: Dict[str, Any],
    summary: str,
    steps: List[Dict],
) -> None:
    """Store a leave application pending user confirmation.

    Called by the executor when a leave create_record is intercepted before
    execution. The stored payload is re-executed on the next turn when the
    user affirms.

    Args:
        auth_context: Auth context for session key
        entity: Entity name (e.g. "employee_leave_hd")
        record: The create_record data payload
        summary: Human-readable summary of the leave request
        steps: The full list of create steps (with $ref markers intact)
    """
    key = _get_session_key(auth_context)
    _SESSION_STATE[key] = _SESSION_STATE.get(key, {})
    _SESSION_STATE[key]["pending_leave_confirmation"] = {
        "entity": entity,
        "record": record,
        "summary": summary,
        "steps": steps,
    }


def load_pending_leave_confirmation(auth_context: Any) -> Optional[Dict]:
    """Load pending leave confirmation if one exists for this session."""
    state = load_session_state(auth_context)
    return state.get("pending_leave_confirmation")


def clear_pending_leave_confirmation(auth_context: Any) -> None:
    """Clear pending leave confirmation after it's been fulfilled or abandoned."""
    key = _get_session_key(auth_context)
    if key in _SESSION_STATE and "pending_leave_confirmation" in _SESSION_STATE[key]:
        del _SESSION_STATE[key]["pending_leave_confirmation"]


# =============================================================================
# Pending Update Confirmation Tracking
# =============================================================================

def save_pending_update_confirmation(
    auth_context: Any,
    entity: str,
    updates: Dict[str, Any],
    summary: str,
    steps: List[Dict],
) -> None:
    """Store a profile update pending user confirmation.

    Called by the profile-update confirmation gate when an implicit life-event
    statement (e.g. "I'm married now") is detected. The stored payload is
    re-executed on the next turn when the user affirms.

    Args:
        auth_context: Auth context for session key
        entity: Entity name (e.g. "employee_general")
        updates: field -> resolved value dict (e.g. {"marital_status": "M"})
        summary: Human-readable summary (e.g. "marital status to Married")
        steps: The full list of read+update steps (with $ref markers intact)
    """
    key = _get_session_key(auth_context)
    _SESSION_STATE[key] = _SESSION_STATE.get(key, {})
    _SESSION_STATE[key]["pending_update_confirmation"] = {
        "entity": entity,
        "updates": updates,
        "summary": summary,
        "steps": steps,
    }


def load_pending_update_confirmation(auth_context: Any) -> Optional[Dict]:
    """Load pending update confirmation if one exists for this session."""
    state = load_session_state(auth_context)
    return state.get("pending_update_confirmation")


def clear_pending_update_confirmation(auth_context: Any) -> None:
    """Clear pending update confirmation after it's been fulfilled or abandoned."""
    key = _get_session_key(auth_context)
    if key in _SESSION_STATE and "pending_update_confirmation" in _SESSION_STATE[key]:
        del _SESSION_STATE[key]["pending_update_confirmation"]