"""Simple update strategy for self-update entities.

For entities where an employee updates only their own record via:
1. Read current record (to get PK and existing values)
2. Extract intended field changes from query (regex + LLM fallback)
3. Update with validated fields

Returns (steps, pending_slots) so the planner owns the ask-vs-act decision,
not the injector silencing itself.

Supports:
- Single-field updates: "update my mobile number"
- Multi-field updates: "change my address and emergency contact"
- "Update X to Y" pattern: "update marital status to married"
- LLM fallback: "I'm married now" (regex misses, LLM resolves field+value)
- Pending slot: field detected but no value → returns pending_slot for planner
- Pending-slot follow-up: bare value reply ("married") after the agent asked
  "What would you like to set your marital status to?" — the pending slot is
  loaded from session state and passed as pending_field so the LLM fallback
  binds the reply to the right field.
- Code resolution: "married" → "M" via codesetup lookup
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from agent.actions.base_injector import (
    DEFAULT_FIELD_PATTERNS,
    _llm_extract_field_value,
    get_entity_config,
    get_entity_fields_from_schema,
    is_action_request,
    is_plausible_slot_value,
    parse_date_from_query,
    parse_field_from_query,
    resolve_field_value,
    validate_steps,
)
from agent.core.session_state import (
    clear_pending_action,
    load_pending_action,
    save_pending_action,
)

logger = logging.getLogger("hr_agent")


# =============================================================================
# Implicit life-event statement detection
# =============================================================================
# First-person present-tense statements that imply a profile update without
# using an explicit action verb ("update", "change", etc.). These are routed
# through the LLM extraction path (with implicit_update=True) so the agent can
# resolve the field+value from context, then confirm before committing.
_PERSONAL_STATE_PATTERNS = [
    re.compile(r"^(?:i'm|i am|i just|i've|i have|i got|i moved|i changed|i updated)\b", re.I),
    re.compile(r"^my\s+[\w\s]{1,40}\b(?:is|was|has been|changed|is now)\b", re.I),
]

# Common chitchat continuations after "I'm/I am" that are NOT profile updates.
_CHITCHAT_CONTINUATIONS = frozenset({
    "good", "fine", "okay", "ok", "great", "well", "sorry", "sure",
    "not sure", "back", "here", "done", "ready", "available", "confused",
    "happy", "sad", "tired", "busy", "excited", "curious", "not",
    "not sure", "not ready", "not available", "not free", "not sure",
})


def is_personal_state_statement(query: str) -> bool:
    """Heuristic: is the query a first-person life-event statement?

    Examples that return True:
      "I'm married now"
      "I just moved house"
      "my email is test@example.com"

    Examples that return False:
      "update my marital status"  (explicit action request)
      "what's my leave balance?"  (question)
      "I'm good"                  (chitchat)
      "I'm sorry"                 (chitchat)
    """
    q = query.strip().lower()
    if not q or len(q) > 120:
        return False
    if "?" in q:
        return False
    if not any(p.match(q) for p in _PERSONAL_STATE_PATTERNS):
        return False
    # Strip the leading "i'm/i am/my" prefix and check the first remaining word.
    remainder = re.sub(r"^(?:i'm|i am|i just|i've|i have|i got|i moved|i changed|i updated|my)\s+", "", q)
    first_word = remainder.split()[0] if remainder.split() else ""
    if first_word in _CHITCHAT_CONTINUATIONS:
        return False
    return True


# Field groups for multi-field extraction
FIELD_GROUPS = {
    "contact": ["mobile_no", "personal_email", "mail_address", "contact_no"],
    "emergency": ["emergency_name", "emergency_contact_no", "emergency_relation",
                  "emergency_email", "emergency_name2", "emergency_contact_no2",
                  "emergency_relation2", "emergency_email2", "emergency_address",
                  "emergency_address2"],
    "passport": ["passport_no", "passport_expiry", "visa_no"],
    "work_permit": ["work_permit_no", "work_permit_expiry", "work_permit_type",
                    "work_permit_issued", "work_permit_application",
                    "work_permit_issue_authority"],
    "personal": ["nickname", "personal_title", "name_ext"],
    "address": ["mail_address", "home_mail_address", "google_geo_lat",
                "google_geo_lang", "date_residence"],
    "identification": ["national_id", "old_national_id", "place_of_birth",
                       "state_of_birth", "driving_licence", "place_of_issue",
                       "country_of_issue", "place_of_issues", "country_of_issues",
                       "oku_no", "wp_foreign_national_id"],
}


def extract_update_fields(
    user_query: str,
    allowed_fields: List[str],
    field_patterns: Optional[Dict[str, List[str]]] = None,
    conversation_history: str = "",
    pending_field: Optional[str] = None,
    tenant_id: str = "default",
    implicit_update: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, bool], Optional[Dict[str, str]]]:
    """Extract field -> value pairs from user query.

    Process:
    1. Pending-slot follow-up: if pending_field is set, treat the reply as the
       value for that field (deterministic binding via resolve_field_value).
    2. Try regex patterns for each allowed field
    3. On regex miss, fall back to LLM extraction (agentic)
    4. If LLM also misses, track field as mentioned-without-value

    Also returns a dict tracking which fields were mentioned but had no value,
    and an optional pending_slot dict when a field is detected but value is missing.

    Args:
        user_query: The user's query
        allowed_fields: List of field names that can be updated
        field_patterns: Optional pattern overrides
        conversation_history: Prior turns for LLM context
        pending_field: Field from prior turn needing a value (for follow-up)
        tenant_id: Tenant ID for code resolution
        implicit_update: When True, the query is a life-event statement
            (e.g. "I'm married now") rather than an explicit action request.
            The LLM fallback fires even if no field was keyword-mentioned,
            because the field+value are implied rather than explicit.

    Returns:
        Tuple of (updates dict, mentioned_without_value dict, pending_slot or None)
        - updates: field_name -> extracted_value (codes resolved)
        - mentioned_without_value: field_name -> True (field mentioned, no value found)
        - pending_slot: {entity, field, code_type, user_question} when clarification needed
    """
    q = user_query.lower()
    updates = {}
    mentioned_without_value = {}
    pending_slot = None

    if field_patterns is None:
        field_patterns = DEFAULT_FIELD_PATTERNS

    # Map field names to codesetup type names (for pending_slot)
    CODE_TYPE_MAP = {
        "marital_status": "MARITAL_STATUS",
        "gender": "GENDER",
        "confirmation_status": "CONFIRMATION_STATUS",
        "employee_status": "STATUS",
        "employment_category": "EMPLOYMENT_CATEGORY",
    }

    # -- Pending-slot follow-up: the user replied to a prior clarification
    # question (e.g. "married" after "What would you like to set your
    # marital status to?"). The reply is the VALUE for pending_field —
    # bind it directly instead of requiring the field to be re-mentioned.
    if pending_field and pending_field in allowed_fields:
        value = user_query.strip()
        if value:
            resolved = resolve_field_value(pending_field, value, tenant_id)
            updates[pending_field] = resolved
            logger.info(
                "extract_update_fields: pending-slot follow-up bound %s='%s' (resolved from '%s')",
                pending_field, resolved, value,
            )
            return updates, mentioned_without_value, pending_slot

    # First pass: try each allowed field with regex
    for field in allowed_fields:
        if not _field_mentioned(q, field):
            continue

        value = parse_field_from_query(q, field, field_patterns)
        if value:
            updates[field] = value
        else:
            mentioned_without_value[field] = True

    # Second pass: handle group-based extraction
    for group_name, group_fields in FIELD_GROUPS.items():
        if group_name in q or any(f.replace("_", " ") in q for f in group_fields):
            for field in group_fields:
                if field not in allowed_fields:
                    continue
                if field in updates or field in mentioned_without_value:
                    continue
                value = parse_field_from_query(q, field, field_patterns)
                if value:
                    updates[field] = value
                elif _field_mentioned(q, field):
                    mentioned_without_value[field] = True

    # Third pass: LLM fallback when regex found nothing.
    # This handles ambiguous cases like "I'm married now" (regex misses the
    # field mention) or bare follow-up replies (pending_field set: the user's
    # reply is the value for a prior-turn field).
    if not updates and (mentioned_without_value or implicit_update):
        llm_result = _llm_extract_field_value(
            user_query,
            allowed_fields,
            conversation_history=conversation_history,
            pending_field=pending_field,
            tenant_id=tenant_id,
        )

        if llm_result.get("field") and llm_result.get("value"):
            # LLM found field + value
            field = llm_result["field"]
            value = llm_result["value"]
            if field in allowed_fields:
                resolved = resolve_field_value(field, value, tenant_id)
                updates[field] = resolved
                mentioned_without_value.pop(field, None)
                logger.info(
                    "extract_update_fields: LLM resolved %s='%s' (from '%s')",
                    field, resolved, value
                )
        elif llm_result.get("needs_clarification"):
            # LLM confirms field mentioned, no value, produce pending_slot
            clar_field = (
                llm_result.get("clarification_needed_for")
                or (list(mentioned_without_value.keys())[0] if mentioned_without_value else None)
            )
            if not clar_field:
                # Nothing to clarify (e.g. a bare follow-up that the LLM
                # couldn't bind) — leave pending_slot as None so the caller
                # keeps the existing pending action in session state.
                return updates, mentioned_without_value, pending_slot
            code_type = CODE_TYPE_MAP.get(clar_field)
            user_question = (
                f"What would you like to set your {clar_field.replace('_', ' ')} to?"
            )
            pending_slot = {
                "entity": "employee_general",  # resolved by caller
                "field": clar_field,
                "code_type": code_type,
                "user_question": user_question,
            }
            logger.info(
                "extract_update_fields: LLM confirmed field '%s' needs clarification — pending_slot created",
                clar_field
            )
        elif mentioned_without_value:
            # LLM also found nothing — use first mentioned field as pending
            pending_field_name = list(mentioned_without_value.keys())[0]
            code_type = CODE_TYPE_MAP.get(pending_field_name)
            pending_slot = {
                "entity": "employee_general",
                "field": pending_field_name,
                "code_type": code_type,
                "user_question": f"What would you like to set your {pending_field_name.replace('_', ' ')} to?",
            }
        # else: implicit probe found nothing and no field was mentioned —
        # return empty (the caller falls through to normal planning).

    return updates, mentioned_without_value, pending_slot


def _field_mentioned(query: str, field_name: str) -> bool:
    """Check if a field is mentioned in the query via keywords."""
    # Direct field name
    if field_name in query:
        return True

    # Common variations
    variations = {
        "mobile_no": ["mobile", "phone", "contact number", "handphone", "hp"],
        "personal_email": ["personal email", "my email"],
        "mail_address": ["address", "mailing address", "home address"],
        "contact_no": ["contact", "contact number", "phone number"],
        "emergency_name": ["emergency contact name", "emergency name"],
        "emergency_contact_no": ["emergency contact", "emergency number", "emergency phone"],
        "emergency_relation": ["emergency relation", "relationship to emergency"],
        "emergency_email": ["emergency email"],
        "emergency_address": ["emergency address"],
        "nickname": ["nickname", "nick name"],
        "personal_title": ["title", "salutation"],
        "passport_no": ["passport", "passport number"],
        "passport_expiry": ["passport expiry", "passport expiration"],
        "visa_no": ["visa", "visa number"],
        "work_permit_no": ["work permit", "work permit number"],
        "work_permit_expiry": ["work permit expiry"],
        "national_id": ["ic", "national id", "identification card"],
        "driving_licence": ["driving licence", "license"],
        "marital_status": ["marital status", "marital"],
    }

    for field, keywords in variations.items():
        if field == field_name:
            if any(kw in query for kw in keywords):
                return True

    return False


def inject_simple_update(
    steps: List[Dict],
    user_query: str,
    auth_context: Any,
    cached_schema: Dict,
    cached_tools: Optional[List[Dict]] = None,
    tone_context: Optional[Dict] = None,
    tenant_id: Optional[str] = None,
    entity_name: str = "employee_general",
    conversation_history: str = "",
) -> Tuple[List[Dict], List[Dict]]:
    """Inject read + update steps for simple self-update entities.

    Returns (steps, pending_slots):
      - steps: Modified planner steps with injected read + update
      - pending_slots: List of {entity, field, code_type, user_question} when
        clarification is needed (field detected but no value)

    The planner owns the decision of whether to ask the user or skip — not
    the injector unilaterally silencing itself.

    Args:
        steps: Current planner steps
        user_query: User's query
        auth_context: Auth context with employee info
        cached_schema: DAB cached schema
        cached_tools: DAB tool schemas for validation
        tone_context: Tone/intent context from classifier
        tenant_id: Tenant ID for config loading
        entity_name: Entity to update
        conversation_history: Prior turns for LLM context

    Returns:
        Tuple of (modified steps, pending_slots list)
    """
    pending_slots: List[Dict] = []

    if not steps or not cached_schema:
        return steps, pending_slots

    # Load a pending action from a prior turn (field mentioned without a
    # value). When present, this turn is a follow-up reply (e.g. "married")
    # that must be treated as the value for the pending field — the normal
    # action-keyword gate would reject a bare value reply outright.
    pending_action = load_pending_action(auth_context)
    pending_field = None
    if pending_action and pending_action.get("action") == "update" and pending_action.get("entity") == entity_name:
        pending_field = pending_action.get("field")

    # Stale pending-action guard: if a pending field exists but the current
    # query is NOT a plausible bare-value reply (e.g. "I want to apply 1 day
    # leave" is a new request, not a value for marital_status), the pending
    # action is stale and must be cleared so it doesn't hijack the query.
    # Without this, the entire query string would be bound as the field value
    # (e.g. marital_status="I want to apply 1 day leave"), producing bogus
    # update steps that overwrite the user's actual intent.
    if pending_field and not is_plausible_slot_value(user_query):
        logger.info(
            "SIMPLE_UPDATE: stale pending action for %s.%s — query %r is not a "
            "bare-value reply, clearing and falling through to normal handling",
            entity_name, pending_field, user_query,
        )
        clear_pending_action(auth_context)
        pending_field = None

    # Detect implicit life-event statements ("I'm married now") that imply an
    # update without an explicit action verb. These bypass the action-keyword
    # and intent gates so the LLM extraction can resolve field+value.
    is_implicit = is_personal_state_statement(user_query)

    # Check if this is an action request (or a pending-slot follow-up, or an
    # implicit life-event statement)
    if not pending_field and not is_implicit and not is_action_request(user_query, tenant_id=tenant_id):
        return steps, pending_slots

    # Get entity config
    config = get_entity_config(entity_name)
    if not config:
        logger.debug("SIMPLE_UPDATE: no entity config for %s", entity_name)
        return steps, pending_slots

    if config.get("strategy") != "simple_update":
        return steps, pending_slots

    # Validate the pending field is still updateable for this entity; if not,
    # drop the stale pending and fall through to normal action-request handling.
    allowed = set(config.get("self_updateable_fields", []))
    if pending_field and pending_field not in allowed:
        logger.info("SIMPLE_UPDATE: pending field %s no longer updateable for %s — dropping", pending_field, entity_name)
        pending_field = None
        if not is_implicit and not is_action_request(user_query, tenant_id=tenant_id):
            return steps, pending_slots

    # Check tone_context for update intent
    tc_intent_category = (tone_context or {}).get("intent_category", "")
    tc_intent = (tone_context or {}).get("intent", "")
    if not pending_field and not is_implicit and tc_intent_category not in ("action_request", "policy_action") and tc_intent not in ("update_profile", "update_employee"):
        update_indicators = ["update", "edit", "change", "modify", "correct", "set"]
        if not any(ind in user_query.lower() for ind in update_indicators):
            return steps, pending_slots

    # Check if entity exists in schema
    if entity_name not in cached_schema:
        return steps, pending_slots

    # Check if already has update step for this entity
    has_update = any(
        step.get("tool") == "update_record" and step.get("args", {}).get("entity") == entity_name
        for step in steps
    )
    if has_update:
        return steps, pending_slots

    # Get employee_no from auth context
    employee_no = None
    if auth_context and getattr(auth_context, "authenticated", False):
        employee_no = getattr(auth_context, "emp_id", None) or getattr(auth_context, "email", None)
    if not employee_no:
        logger.info("SIMPLE_UPDATE: no employee_no from auth, skipping")
        return steps, pending_slots

    tid = tenant_id or "default"

    # Get entity fields
    entity_fields = get_entity_fields_from_schema(cached_schema, entity_name)

    # Get allowed/protected fields from config
    protected = set(config.get("protected_fields", []))
    key_fields = config.get("key_fields", ["employee_no"])

    # Map field names to codesetup type names (for code resolution)
    CODE_TYPE_MAP = {
        "marital_status": "MARITAL_STATUS",
        "gender": "GENDER",
        "confirmation_status": "CONFIRMATION_STATUS",
        "employee_status": "STATUS",
        "employment_category": "EMPLOYMENT_CATEGORY",
    }

    # Extract intended updates + track mentioned-but-no-value fields
    raw_updates, mentioned_without_value, pending_slot = extract_update_fields(
        user_query,
        list(allowed),
        DEFAULT_FIELD_PATTERNS,
        conversation_history=conversation_history,
        pending_field=pending_field,
        tenant_id=tid,
        implicit_update=is_implicit,
    )

    # Filter to allowed fields only and resolve codes
    updates: Dict[str, Any] = {}
    for field, value in raw_updates.items():
        if field in protected:
            logger.warning("SIMPLE_UPDATE: skipping protected field %s", field)
            continue
        if field not in entity_fields:
            logger.warning("SIMPLE_UPDATE: field %s not in entity schema", field)
            continue
        resolved = resolve_field_value(field, value, tid)
        updates[field] = resolved

    # If no updates but clarification is needed, return pending_slot for planner
    if not updates and pending_slot:
        # Persist the pending slot so the next turn can bind the user's reply
        # to this field (the LLM fallback won't fire for a bare value reply
        # without the pending_field context).
        save_pending_action(
            auth_context,
            action="update",
            entity=entity_name,
            field=pending_slot.get("field", ""),
            code_type=pending_slot.get("code_type"),
        )
        pending_slots.append(pending_slot)
        return steps, pending_slots

    # If no updates at all, return early (no pending_slot needed)
    if not updates:
        return steps, pending_slots

    # Implicit life-event statements ("I'm married now") must be confirmed
    # before writing. Route through the confirmation gate which stores the
    # steps in session state and returns them for the summarizer to present
    # a confirmation message. The caller (executor) will NOT execute these
    # steps — it will return the confirmation message instead.
    if is_implicit and not pending_field:
        gate_result = profile_update_confirmation_gate(
            entity_name, updates, auth_context, cached_schema,
            cached_tools, conversation_history, tid,
        )
        if gate_result is not None:
            gate_steps, gate_summary = gate_result
            # Mark these steps as confirmation-pending so the executor
            # knows not to execute them yet.
            for s in gate_steps:
                s["_confirmation_pending"] = True
                s["_confirmation_summary"] = gate_summary
            steps.extend(gate_steps)
            # Return a special pending_slot that carries the summary
            pending_slots.append({
                "entity": entity_name,
                "field": list(updates.keys())[0],
                "code_type": None,
                "user_question": gate_summary,
                "confirmation_type": "profile_update",
            })
            return steps, pending_slots

    # Build read + update steps using the shared helper
    new_steps = _build_read_update_steps(
        entity_name, employee_no, updates,
        entity_fields, key_fields, cached_tools,
    )
    if new_steps is None:
        logger.warning("SIMPLE_UPDATE: step validation failed, skipping injection")
        return steps, pending_slots

    logger.info(
        "SIMPLE_UPDATE: injecting read + update for %s (fields=%s)",
        entity_name, list(updates.keys())
    )
    steps.extend(new_steps)

    # The pending action was fulfilled (value bound + steps built) — clear it
    # so a stale slot doesn't hijack a future turn.
    if pending_field:
        clear_pending_action(auth_context)
        logger.info("SIMPLE_UPDATE: pending action for %s.%s fulfilled and cleared", entity_name, pending_field)

    return steps, pending_slots


def handle_pending_update(
    user_query: str,
    entity_name: str,
    field: str,
    auth_context: Any,
    cached_schema: Dict,
    cached_tools: Optional[List[Dict]] = None,
    conversation_history: str = "",
    tenant_id: str = "default",
) -> Optional[Tuple[List[Dict], Optional[str]]]:
    """Bind a follow-up reply to a pending update field and build steps.

    Called by the executor's _handle_pending_update when a prior turn stored
    a pending action (field mentioned without a value) and the user has now
    replied with the value (e.g. "married" after "What would you like to set
    your marital status to?").

    Returns:
        (steps, None)           — read + update steps ready for execution
        (None, reask_message)   — the reply had no usable value; re-ask the
                                  user (the pending action is kept in session
                                  state so the next turn can still bind it)
        None                    — the entity is not a simple_update entity;
                                  caller should fall through to normal planning
    """
    config = get_entity_config(entity_name)
    if not config or config.get("strategy") != "simple_update":
        return None

    allowed = set(config.get("self_updateable_fields", []))
    if field not in allowed:
        logger.info("handle_pending_update: field %s not updateable for %s", field, entity_name)
        return None, f"I can't update {field.replace('_', ' ')} for your profile. Please specify a different field."

    if entity_name not in cached_schema:
        return None

    employee_no = None
    if auth_context and getattr(auth_context, "authenticated", False):
        employee_no = getattr(auth_context, "emp_id", None) or getattr(auth_context, "email", None)
    if not employee_no:
        return None, "I couldn't identify your employee number. Please try again."

    tid = tenant_id or "default"
    entity_fields = get_entity_fields_from_schema(cached_schema, entity_name)
    protected = set(config.get("protected_fields", []))
    key_fields = config.get("key_fields", ["employee_no"])

    # Bind the reply to the pending field. extract_update_fields with
    # pending_field set treats the reply as the value (deterministic fast
    # path via resolve_field_value / codesetup).
    updates, _mentioned, _slot = extract_update_fields(
        user_query,
        [field],
        DEFAULT_FIELD_PATTERNS,
        conversation_history=conversation_history,
        pending_field=field,
        tenant_id=tid,
    )

    if not updates:
        # The reply didn't resolve to a value for the pending field.
        # Re-ask, keeping the pending action in session state.
        question = (
            f"What would you like to set your {field.replace('_', ' ')} to?"
        )
        logger.info("handle_pending_update: reply %r did not bind to %s — re-asking", user_query, field)
        return None, question

    # Filter + resolve (same logic as inject_simple_update)
    resolved_updates: Dict[str, Any] = {}
    for f, value in updates.items():
        if f in protected:
            continue
        if f not in entity_fields:
            continue
        resolved_updates[f] = resolve_field_value(f, value, tid)

    if not resolved_updates:
        return None, f"I couldn't understand the value for {field.replace('_', ' ')}. What would you like to set it to?"

    # Build read + update steps using the shared helper
    new_steps = _build_read_update_steps(
        entity_name, employee_no, resolved_updates,
        entity_fields, key_fields, cached_tools,
    )
    if new_steps is None:
        return None, "I couldn't prepare the update. Please try again."

    logger.info(
        "handle_pending_update: built read+update for %s.%s (value=%s)",
        entity_name, field, resolved_updates.get(field),
    )
    return new_steps, None


# =============================================================================
# Shared step-building helper
# =============================================================================

def _build_read_update_steps(
    entity_name: str,
    employee_no: str,
    updates: Dict[str, Any],
    entity_fields: List[str],
    key_fields: List[str],
    cached_tools: Optional[List[Dict]] = None,
) -> Optional[List[Dict]]:
    """Build read + update steps for a simple_update entity.

    Shared by inject_simple_update and handle_pending_update to avoid
    duplicating the step-construction logic.

    Returns the step list, or None if validation fails.
    """
    select_fields = key_fields + list(updates.keys())
    available_select = [f for f in select_fields if f in entity_fields]
    read_step = {
        "tool": "read_records",
        "args": {
            "entity": entity_name,
            "filter": f"employee_no eq '{employee_no}'",
            "select": ",".join(available_select) if available_select else "*",
            "first": "1",
        },
        "_step_id": f"{entity_name}_read",
    }

    keys = {}
    for kf in key_fields:
        if kf == "employee_no":
            keys[kf] = employee_no
        else:
            keys[kf] = {"$ref": f"{entity_name}_read.result.0.{kf}"}

    update_step = {
        "tool": "update_record",
        "args": {
            "entity": entity_name,
            "keys": keys,
            "fields": updates,
        },
    }

    new_steps = [read_step, update_step]
    if cached_tools and not validate_steps(new_steps, cached_tools):
        logger.warning("SIMPLE_UPDATE: step validation failed for %s", entity_name)
        return None

    return new_steps


# =============================================================================
# Code description resolution
# =============================================================================

def _resolve_code_description(
    field: str,
    value: str,
    tenant_id: str,
) -> str:
    """Resolve a code value to its human-readable description for summaries.

    E.g., marital_status="M" → "Married". Falls back to the raw value
    when no description is found.
    """
    CODE_TYPE_MAP = {
        "marital_status": "MARITAL_STATUS",
        "gender": "GENDER",
        "confirmation_status": "CONFIRMATION_STATUS",
        "employee_status": "STATUS",
        "employment_category": "EMPLOYMENT_CATEGORY",
    }
    code_type = CODE_TYPE_MAP.get(field)
    if not code_type:
        return value
    try:
        from agent.dab.code_resolver import CodeResolver
        resolver = CodeResolver(tenant_id)
        code_map = resolver.resolve_code_type(code_type)
        if value in code_map:
            return code_map[value]
    except Exception:
        pass
    return value


# =============================================================================
# Profile-update confirmation gate
# =============================================================================

def profile_update_confirmation_gate(
    entity_name: str,
    updates: Dict[str, Any],
    auth_context: Any,
    cached_schema: Dict,
    cached_tools: Optional[List[Dict]] = None,
    conversation_history: str = "",
    tenant_id: str = "default",
) -> Optional[Tuple[List[Dict], str]]:
    """Intercept an implicit life-event update and gate it behind confirmation.

    Called when is_personal_state_statement() detected a first-person life-event
    statement (e.g. "I'm married now") and extract_update_fields resolved a
    field+value. Instead of executing the update immediately, this function:

    1. Builds the read+update steps (but does NOT execute them).
    2. Stores a pending_update_confirmation in session state so the next turn
       (when the user says "yes") can re-execute the steps.
    3. Returns (steps, summary) so the caller can present a confirmation
       message to the user.

    Returns:
        (steps, summary) when the update should be confirmed before execution.
        None when the entity is not a simple_update entity or no updates
        were resolved (caller falls through to normal planning).
    """
    from agent.core.session_state import save_pending_update_confirmation

    config = get_entity_config(entity_name)
    if not config or config.get("strategy") != "simple_update":
        return None

    if not updates:
        return None

    allowed = set(config.get("self_updateable_fields", []))
    protected = set(config.get("protected_fields", []))
    key_fields = config.get("key_fields", ["employee_no"])

    # Filter to allowed + non-protected fields
    filtered_updates: Dict[str, Any] = {}
    for field, value in updates.items():
        if field in protected:
            logger.warning("SIMPLE_UPDATE: skipping protected field %s", field)
            continue
        if field not in allowed:
            logger.warning("SIMPLE_UPDATE: field %s not in self_updateable_fields", field)
            continue
        filtered_updates[field] = value

    if not filtered_updates:
        return None

    if entity_name not in cached_schema:
        return None

    employee_no = None
    if auth_context and getattr(auth_context, "authenticated", False):
        employee_no = getattr(auth_context, "emp_id", None) or getattr(auth_context, "email", None)
    if not employee_no:
        return None

    entity_fields = get_entity_fields_from_schema(cached_schema, entity_name)

    # Build human-readable summary for the confirmation message
    summary_parts = []
    for field, value in filtered_updates.items():
        desc = _resolve_code_description(field, str(value), tenant_id)
        summary_parts.append(f"{field.replace('_', ' ')} to {desc}")
    summary = "; ".join(summary_parts)

    # Build steps (not executed yet — stored for confirmation turn)
    steps = _build_read_update_steps(
        entity_name, employee_no, filtered_updates,
        entity_fields, key_fields, cached_tools,
    )
    if steps is None:
        return None

    # Store for the confirmation turn
    save_pending_update_confirmation(
        auth_context,
        entity=entity_name,
        updates=filtered_updates,
        summary=summary,
        steps=steps,
    )

    logger.info(
        "SIMPLE_UPDATE: profile-update confirmation gate — stored pending for %s (summary=%s)",
        entity_name, summary,
    )
    return steps, summary
