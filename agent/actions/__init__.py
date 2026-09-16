"""Unified Action Framework - Agentic step injection for entity operations.

This module provides a unified interface for injecting action steps (create, update)
into the planner's step list. It replaces the per-module injection pattern with a
registry-based approach.

Architecture:
- config/entity_actions.yaml: Entity registry (strategies, fields, permissions)
- base_injector.py: Shared utilities (field parsing, validation, LLM fallback)
- strategies/: Per-strategy implementations
    - simple_update_strategy.py: Self-update (employee_general)
    - leave_strategy.py: Leave requests (action + entitlement)

Usage:
    from agent.actions import inject_all_actions

    steps, pending_slots = inject_all_actions(
        steps, user_query, auth_context, cached_schema,
        cached_tools, tone_context, tenant_id, conversation_history
    )

Adding a new entity:
1. Add entity config to config/entity_actions.yaml
2. If strategy exists (simple_update, leave_action), no new code needed
3. If new strategy needed, create strategies/new_strategy.py
4. Register in inject_all_actions()
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from agent.actions.base_injector import _load_entity_actions_config
from agent.actions.strategies.simple_update_strategy import inject_simple_update
from agent.actions.strategies.leave_strategy import (
    inject_leave_action,
    inject_entitlement_update,
)

logger = logging.getLogger("hr_agent")


# =============================================================================
# Strategy Registry
# Maps strategy name to injection function
# =============================================================================

_STRATEGY_REGISTRY: Dict[str, Any] = {
    "simple_update": inject_simple_update,
    "leave_action": inject_leave_action,
    "entitlement_update": inject_entitlement_update,
}


def _get_entity_strategy_map() -> Dict[str, str]:
    """Build entity -> strategy mapping from config."""
    configs = _load_entity_actions_config()
    entity_map = {}
    for entity_name, config in configs.items():
        strategy = config.get("strategy")
        if strategy:
            entity_map[entity_name] = strategy
    return entity_map


# =============================================================================
# Main Entry Point
# =============================================================================

def inject_all_actions(
    steps: List[Dict],
    user_query: str,
    auth_context: Any,
    cached_schema: Dict,
    cached_tools: Optional[List[Dict]] = None,
    tone_context: Optional[Dict] = None,
    tenant_id: Optional[str] = None,
    conversation_history: str = "",
) -> Tuple[List[Dict], List[Dict]]:
    """Inject action steps for all matching entities.

    This is the main entry point called by tool_planner.py.

    Process:
    1. Load entity configs from YAML
    2. For each entity with an action config:
       - Check if query matches the entity's action pattern
       - If yes, invoke the corresponding strategy
    3. Accumulate pending_slots from all strategies
    4. Return (modified steps, pending_slots list)

    The planner owns the decision of whether to ask for clarification —
    pending_slots are surfaced to the planner so it can decide to produce
    a clarification step or skip the action.

    Args:
        steps: Current planner steps
        user_query: User's query
        auth_context: Auth context with employee info
        cached_schema: DAB cached schema
        cached_tools: DAB tool schemas for validation
        tone_context: Tone/intent context from classifier
        tenant_id: Tenant ID for config loading
        conversation_history: Prior turns for LLM extraction context

    Returns:
        Tuple of (modified steps list, pending_slots list)
        - steps: planner steps with injected action steps
        - pending_slots: [{entity, field, code_type, user_question}, ...]
          when clarification is needed; empty if all fields resolved
    """
    pending_slots: List[Dict] = []

    if not cached_schema:
        return steps, pending_slots

    entity_map = _get_entity_strategy_map()
    if not entity_map:
        logger.debug("inject_all_actions: no entity configs loaded")
        return steps, pending_slots

    q = user_query.lower()

    # Track which strategies have already been invoked for this query so that
    # entities sharing a strategy (e.g. employee_leave_hd + employee_leave both
    # use leave_action) don't produce duplicate pending_slots. The leave
    # injector builds both header and detail steps in a single call, so calling
    # it once per strategy is sufficient.
    invoked_strategies: set = set()

    for entity_name, strategy_name in entity_map.items():
        if strategy_name in invoked_strategies:
            logger.debug(
                "inject_all_actions: skipping %s for %s — strategy %s already invoked",
                entity_name, strategy_name, strategy_name,
            )
            continue
        invoked_strategies.add(strategy_name)

        injector = _STRATEGY_REGISTRY.get(strategy_name)
        if not injector:
            logger.warning("inject_all_actions: no injector for strategy %s", strategy_name)
            continue

        try:
            if strategy_name in ("simple_update", "leave_action", "entitlement_update"):
                steps, entity_pending = injector(
                    steps,
                    user_query,
                    auth_context,
                    cached_schema,
                    cached_tools=cached_tools,
                    tone_context=tone_context,
                    tenant_id=tenant_id,
                    entity_name=entity_name,  # simple_update uses this; leave injectors ignore it
                    conversation_history=conversation_history,
                )
                if entity_pending:
                    pending_slots.extend(entity_pending)
            else:
                steps = injector(
                    steps,
                    user_query,
                    auth_context,
                    cached_schema,
                    cached_tools=cached_tools,
                    tone_context=tone_context,
                    tenant_id=tenant_id,
                )
        except Exception as exc:
            logger.error(
                "inject_all_actions: %s injection failed for %s: %s",
                strategy_name, entity_name, exc
            )

    # Deduplicate pending_slots by (entity, field) — multiple entities sharing
    # a strategy (e.g. employee_leave_hd + employee_leave) can produce identical
    # slot entries. Keep the first occurrence so the user isn't asked twice.
    seen: set = set()
    deduped: List[Dict] = []
    for slot in pending_slots:
        key = (slot.get("entity", ""), slot.get("field", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(slot)
    pending_slots = deduped

    return steps, pending_slots


# =============================================================================
# Individual Injectors (for targeted use)
# =============================================================================

def inject_employee_general_update(
    steps: List[Dict],
    user_query: str,
    auth_context: Any,
    cached_schema: Dict,
    cached_tools: Optional[List[Dict]] = None,
    tone_context: Optional[Dict] = None,
    tenant_id: Optional[str] = None,
    conversation_history: str = "",
) -> Tuple[List[Dict], List[Dict]]:
    """Inject action steps specifically for employee_general updates.

    Convenience function - equivalent to inject_all_actions but only for
    employee_general. Useful when you want to call it separately.

    Returns (steps, pending_slots).
    """
    return inject_simple_update(
        steps,
        user_query,
        auth_context,
        cached_schema,
        cached_tools=cached_tools,
        tone_context=tone_context,
        tenant_id=tenant_id,
        entity_name="employee_general",
        conversation_history=conversation_history,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "inject_all_actions",
    "inject_employee_general_update",
    "inject_leave_action",
    "inject_entitlement_update",
]
