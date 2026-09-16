"""Leave entitlement update strategy for the unified action framework.

If the query is a leave action request and the user wants to adjust
entitlement/balance, inject a read_records + update_record pair for
employee_leave_entitlement.
"""
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agent.actions.base_injector import get_entity_fields_from_schema, is_action_request
from agent.core.config_loader import get_action_keywords_for_tenant

logger = logging.getLogger("hr_agent")


# =============================================================================
# Adjustment Extraction -- parse signed adjustment values from user queries
# =============================================================================

_ADJUSTMENT_PATTERNS = [
    # "carry forward 5 days", "cf 3 days", "carry 5 days forward"
    (r'carry\s*(?:forward\s*)?(?:by\s*)?(\d+)\s*days?', 1),
    (r'cf\s*(\d+)\s*days?', 1),
    # "forfeit 2 days", "lose 3 days"
    (r'(?:forfeit|lose|cut\s*off)\s*(\d+)\s*days?', -1),
    # "add 2 days", "add 3 more days", "increase by 2"
    (r'(?:add|increase|give\s*me)\s*(?:by\s*)?(\d+)\s*days?', 1),
    # "reduce by 3 days", "deduct 2 days"
    (r'(?:reduce|deduct|subtract)\s*(?:by\s*)?(\d+)\s*days?', -1),
    # bare "adjustment of +5", "adjustment +3"
    (r'(?:adjustment\s*(?:of\s*)?)?([+-]?\d+)\s*days?', 1),
    # "N days" with context
    (r'(\d+)\s*days?\s*(?:forward|carry|forfeit|add|remove)', 1),
]


def _extract_adjustment_from_query(user_query: str) -> Tuple[Optional[int], Optional[Dict[str, str]]]:
    """Extract signed adjustment integer from leave entitlement query.

    Returns (adjustment_value, pending_slot_or_none):
      - If a number is found: (signed_int, None)
      - If no number found: (None, pending_slot dict)

    The pending_slot uses field="adjustment" so the summarizer asks
    generically: "By how much would you like to adjust your entitlement?"
    """
    q = user_query.lower()

    for pattern, sign in _ADJUSTMENT_PATTERNS:
        m = re.search(pattern, q)
        if m:
            raw = m.group(1).lstrip("+")
            try:
                value = int(raw) * sign
                logger.info("_extract_adjustment_from_query: found adjustment=%d (pattern=%s)", value, pattern)
                return value, None
            except ValueError:
                continue

    # No number found — produce a pending slot
    pending_slot = {
        "entity": "employee_leave_entitlement",
        "field": "adjustment",
        "code_type": None,
        "user_question": "By how many days would you like to adjust your leave entitlement? (use negative to reduce, e.g., -2, or positive to add)",
    }
    logger.info("_extract_adjustment_from_query: no adjustment found — pending_slot created")
    return None, pending_slot


# =============================================================================
# Main Injector
# =============================================================================

def inject_leave_entitlement_update_step(
    steps: List[Dict],
    user_query: str,
    auth_context: Any,
    cached_schema: Dict,
    cached_tools: Optional[List[Dict]] = None,
    tone_context: Optional[Dict] = None,
    tenant_id: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Post-processor: if the query is a leave action request and the user wants
    to adjust entitlement/balance, inject a read_records + update_record pair
    for employee_leave_entitlement.

    This enables the agent to update leave entitlement records when the user
    explicitly asks to adjust balance, entitlement, or related fields.

    The read step selects fields derived from the cached schema, and the
    update step uses an explicit ``$ref`` to chain the ``id`` from the read
    result so the executor does not need to guess the record key.
    """
    pending_slots: List[Dict] = []

    if not steps or not cached_schema:
        return steps, pending_slots

    # Load tenant-aware config
    action_keywords = get_action_keywords_for_tenant(tenant_id or "default")

    if not is_action_request(user_query, action_keywords):
        return steps, pending_slots

    if tone_context and tone_context.get("intent_category") != "action_request":
        return steps, pending_slots

    if "employee_leave_entitlement" not in cached_schema:
        return steps, pending_slots

    q = user_query.lower()
    # Detect entitlement adjustment intent: action words near leave/entitlement words
    action_words = {"adjust", "update", "modify", "change", "add", "increase", "decrease", "carry"}
    entitlement_words = {"entitlement", "balance", "leave"}

    has_action = any(kw in q for kw in action_words)
    has_entitlement = any(kw in q for kw in entitlement_words)
    wants_entitlement_update = has_action and has_entitlement
    if not wants_entitlement_update:
        return steps, pending_slots

    employee_no = None
    if auth_context and getattr(auth_context, "authenticated", False):
        employee_no = getattr(auth_context, "emp_id", None) or getattr(auth_context, "email", None)
    if not employee_no:
        return steps, pending_slots

    # Detect leave type from query or default to first available
    leave_code = None
    for code in ("ANL", "MCL", "MAT", "UPL", "CL", "HPL", "RPL", "EXM", "WFH"):
        if code.lower() in q:
            leave_code = code
            break
    if not leave_code:
        leave_code = "ANL"

    # Extract adjustment value from query; if missing, surface as pending_slot for planner.
    adjustment_value, adjustment_slot = _extract_adjustment_from_query(user_query)

    # Determine year: prefer current year
    try:
        year = str(datetime.now().year)
    except Exception:
        year = "2026"

    entitlement_fields = get_entity_fields_from_schema(cached_schema, "employee_leave_entitlement")

    if entitlement_fields:
        # Schema-driven path: derive select fields from schema and use $ref for id.
        preferred_order = ["id", "employee_no", "leave_code", "year", "leave_ent", "leave_bf",
                           "forfeit", "adjustment", "remarks", "ValidFrom", "ValidTo"]
        selected = [f for f in preferred_order if f in entitlement_fields]
        if not selected:
            selected = entitlement_fields[:10]
        select_str = ",".join(selected) if selected else "*"

        read_step = {
            "tool": "read_records",
            "args": {
                "entity": "employee_leave_entitlement",
                "select": select_str,
                "filter": f"employee_no eq '{employee_no}' and leave_code eq '{leave_code}' and year eq {year}",
                "first": "1",
            },
            "_step_id": "entitlement_read",
        }

        update_fields = {}
        if "id" in entitlement_fields:
            update_fields["id"] = {"$ref": "entitlement_read.result.0.id"}
        if adjustment_value is not None and "adjustment" in entitlement_fields:
            update_fields["adjustment"] = adjustment_value
        if "remarks" in entitlement_fields:
            update_fields["remarks"] = "Updated via agent"

        if not update_fields or "id" not in update_fields:
            logger.warning("LEAVE_ENTITLEMENT_UPDATE_GUARD: schema missing required update fields, skipping")
            return steps, pending_slots

        # If no adjustment value was extracted, don't inject update step — ask first.
        if adjustment_value is None:
            logger.info(
                "LEAVE_ENTITLEMENT_UPDATE_GUARD: no adjustment value in query — pending_slot created"
            )
            if adjustment_slot:
                pending_slots.append(adjustment_slot)
            return steps, pending_slots

        update_keys = {"id": update_fields["id"]}
        update_field_values = {k: v for k, v in update_fields.items() if k != "id"}
        update_step = {
            "tool": "update_record",
            "args": {
                "entity": "employee_leave_entitlement",
                "keys": update_keys,
                "fields": update_field_values,
            },
        }
    else:
        # Fallback: only inject if adjustment value is available; otherwise ask via pending_slot.
        if adjustment_value is None:
            if adjustment_slot:
                pending_slots.append(adjustment_slot)
            return steps, pending_slots

        # Fallback to the original hardcoded steps when the schema does not
        # expose field details (e.g., tests with empty fields).
        read_step = {
            "tool": "read_records",
            "args": {
                "entity": "employee_leave_entitlement",
                "select": "id,employee_no,leave_code,year,leave_ent,leave_bf,forfeit,adjustment,remarks",
                "filter": f"employee_no eq '{employee_no}' and leave_code eq '{leave_code}' and year eq {year}",
                "first": "1",
            },
            "_step_id": "entitlement_read",
        }

        update_step = {
            "tool": "update_record",
            "args": {
                "entity": "employee_leave_entitlement",
                "keys": {
                    "id": {"$ref": "entitlement_read.result.0.id"},
                },
                "fields": {
                    "adjustment": adjustment_value,
                    "remarks": "Updated via agent",
                },
            },
        }

    if cached_tools:
        try:
            from agent.dab.validation import validate_dab_args
            _, schema_error = validate_dab_args("read_records", read_step["args"], cached_tools)
            if schema_error:
                logger.warning("LEAVE_ENTITLEMENT_UPDATE_GUARD: read step failed schema validation: %s", schema_error)
                return steps, pending_slots
            _, schema_error = validate_dab_args("update_record", update_step["args"], cached_tools)
            if schema_error:
                logger.warning("LEAVE_ENTITLEMENT_UPDATE_GUARD: update step failed schema validation: %s", schema_error)
                return steps, pending_slots
        except Exception as exc:
            logger.warning("LEAVE_ENTITLEMENT_UPDATE_GUARD: schema validation error: %s", exc)

    logger.info(
        "LEAVE_ENTITLEMENT_UPDATE_GUARD: injected read_records + update_record for employee_leave_entitlement (query=%r)",
        user_query,
    )
    steps.append(read_step)
    steps.append(update_step)
    return steps, pending_slots
