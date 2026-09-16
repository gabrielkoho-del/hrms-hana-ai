"""Leave action strategies for the unified action framework.

Each strategy delegates to the corresponding leave strategy module in this
package. Returns (steps, pending_slots) so the planner owns the ask-vs-act
decision, not the injector silencing itself.

For leave requests, the flow is:
1. leave_entitlement_guard -> injects read for entitlement check (BEFORE action injection)
2. leave_action_strategy -> injects create_record steps (header + detail)
3. leave_entitlement_updater -> handles entitlement adjustments

Strategy types:
- leave_action: Create leave request (header + detail with $ref chaining)
- entitlement_update: Adjust leave entitlement balance
"""
from typing import Any, Dict, List, Optional, Tuple

import logging

from agent.actions.base_injector import get_entity_config

logger = logging.getLogger("hr_agent")


def inject_leave_action(
    steps: List[Dict],
    user_query: str,
    auth_context: Any,
    cached_schema: Dict,
    cached_tools: Optional[List[Dict]] = None,
    tone_context: Optional[Dict] = None,
    tenant_id: Optional[str] = None,
    entity_name: Optional[str] = None,
    conversation_history: str = "",
) -> Tuple[List[Dict], List[Dict]]:
    """Inject leave action steps using the leave action strategy.

    Delegates to leave_action_strategy.inject_leave_action_step.
    When injection is skipped due to missing date or leave type, produces
    a pending_slot so the planner (not the injector) decides whether to ask.

    Returns (steps, pending_slots).
    """
    from agent.actions.strategies.leave_action_strategy import inject_leave_action_step as _inject

    # Resolve leave type codes from DAB via CodeResolver.
    # Uses the cached index (sync, zero-latency). The cache is refreshed by
    # scan_for_codes / get_reverse_index elsewhere in the request lifecycle;
    # if it is still empty here, the injector's LLM fallback and pending_slot
    # chain handle leave-type resolution (never a hardcoded default).
    leave_codes: Optional[List[str]] = None
    tid = tenant_id or "default"
    try:
        from agent.dab.code_resolver import CodeResolver
        resolver = CodeResolver(tid)
        leave_type_map = resolver.resolve_code_type("Leave Type")
        if leave_type_map:
            # CodeResolver returns {code: description}; flatten to list of codes
            leave_codes = list(leave_type_map.keys())
            logger.debug(
                "inject_leave_action: resolved %d leave type codes for tenant %s",
                len(leave_codes), tid
            )
        else:
            logger.debug(
                "inject_leave_action: no cached leave types for tenant %s; "
                "injector will use LLM fallback / pending_slot",
                tid
            )
    except Exception as exc:
        logger.warning("inject_leave_action: CodeResolver error for tenant %s: %s", tid, exc)

    result = _inject(
        steps,
        user_query,
        auth_context,
        cached_schema,
        cached_tools=cached_tools,
        tone_context=tone_context,
        leave_codes=leave_codes,
        tenant_id=tid,
        conversation_history=conversation_history,
    )

    return result


def inject_entitlement_update(
    steps: List[Dict],
    user_query: str,
    auth_context: Any,
    cached_schema: Dict,
    cached_tools: Optional[List[Dict]] = None,
    tone_context: Optional[Dict] = None,
    tenant_id: Optional[str] = None,
    entity_name: Optional[str] = None,
    conversation_history: str = "",
) -> Tuple[List[Dict], List[Dict]]:
    """Inject entitlement update steps using the entitlement updater strategy.

    Delegates to leave_entitlement_updater.inject_leave_entitlement_update_step.
    Returns (steps, pending_slots) — no pending slots for entitlement updates currently.
    """
    from agent.actions.strategies.leave_entitlement_updater import inject_leave_entitlement_update_step as _inject

    result = _inject(
        steps,
        user_query,
        auth_context,
        cached_schema,
        cached_tools=cached_tools,
        tone_context=tone_context,
        tenant_id=tenant_id,
    )
    return result  # already (steps, pending_slots)
