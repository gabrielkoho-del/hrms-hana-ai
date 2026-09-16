"""Leave action strategy for the unified action framework.

If the query is a leave action request but no step uses create_record on
employee_leave, inject a placeholder create_record step.
"""
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from agent.actions.base_injector import get_entity_fields_from_schema, is_action_request
from agent.core.config_loader import get_action_keywords_for_tenant

logger = logging.getLogger("hr_agent")


def _get_required_fields(cached_schema: Dict, entity_name: str) -> set:
    """Determine required fields for leave entities from schema descriptions and name heuristics."""
    entity = cached_schema.get(entity_name, {})
    if not isinstance(entity, dict):
        return set()

    fields = entity.get("fields", entity.get("columns", []))
    if not isinstance(fields, list):
        return set()

    required = set()
    for f in fields:
        if not isinstance(f, dict):
            continue
        name = f.get("name", "")
        desc = (f.get("description") or "").lower()

        # Skip auto-generated primary keys: they are NOT NULL but must not be set on create.
        if any(kw in desc for kw in ("auto-increment", "auto_increment", "do not set on create", "primary key")):
            continue

        # Parse description for required markers
        if "required for create" in desc or "not null" in desc:
            required.add(name)
            continue

        # Name-based heuristics for known required fields
        if entity_name == "employee_leave_hd":
            if name in ("employee_no", "status"):
                required.add(name)
        elif entity_name == "employee_leave":
            if name in ("hd_id", "employee_no", "leave_code", "date_from", "date_to",
                       "days", "status", "emergency", "period_type_from", "period_type_to"):
                required.add(name)

    return required


def _build_record_from_schema(
    cached_schema: Dict,
    entity_name: str,
    fill_fn,
    safe_defaults: Optional[Dict[str, Any]] = None,
) -> Optional[Dict]:
    """Build a create_record payload by walking the schema field list.

    Args:
        cached_schema: The cached DAB schema.
        entity_name: Entity to build record for.
        fill_fn: Callable(field_name, entity_name) -> (value, can_fill).
            Returns (value, True) if the field can be filled, (None, False) otherwise.
        safe_defaults: Optional dict of field_name -> default_value for fields
            that should always be included when present in the schema.

    Returns:
        Record dict with fillable fields, or None if required fields are missing.
    """
    fields = get_entity_fields_from_schema(cached_schema, entity_name)
    required = _get_required_fields(cached_schema, entity_name)
    safe_defaults = safe_defaults or {}

    # Skip auto-generated primary key fields
    entity = cached_schema.get(entity_name, {})
    entity_fields = entity.get("fields", entity.get("columns", []))
    pk_fields = set()
    if isinstance(entity_fields, list):
        for f in entity_fields:
            if isinstance(f, dict) and f.get("primary-key"):
                pk_fields.add(f.get("name", ""))

    record = {}
    missing_required = []

    for field in fields:
        if field in pk_fields:
            continue

        if field in safe_defaults:
            record[field] = safe_defaults[field]
            continue

        value, can_fill = fill_fn(field, entity_name)
        if can_fill:
            record[field] = value
        elif field in required:
            missing_required.append(field)

    if missing_required:
        logger.warning(
            "ACTION_INJECTION_GUARD: %s record missing required cols %s, skipping injection",
            entity_name, missing_required,
        )
        return None

    return record


# =============================================================================
# Days Extraction -- parse number of leave days from user queries
# =============================================================================

_DAYS_PATTERNS = [
    # "5 days", "3 days off", "book 5 days", "apply for 3 days"
    r'(?:book|apply\s*for|request|take|use)\s*(\d+)\s*days?',
    # "a week off" → 7, "a full week" → 7
    r'\ba\s*(?:full\s*)?week\b',
    # "a full day" → 1, "full day" → 1
    r'\ba?\s*full\s*day\b',
    # "half day" → 0.5, "half day off" → 0.5
    r'\bhalf\s*day(?:s?\s*off)?\b',
    # "a day" → 1, "a day off" → 1
    r'\ba\s*day(?:\s*off)?\b',
    # "5 days of annual leave"
    r'(\d+)\s*days?\s*(?:of|off)',
]


def _extract_days_from_query(user_query: str) -> Tuple[Optional[int], Optional[Dict[str, str]]]:
    """Extract number of leave days from user query.

    Returns (days_value, pending_slot_or_none):
      - If days found: (int, None)
      - If no days found: (None, pending_slot dict)

    The pending_slot uses field="days" so the summarizer asks:
    "How many days of leave would you like to take?"
    """
    q = user_query.lower()

    for pattern in _DAYS_PATTERNS:
        m = re.search(pattern, q)
        if m:
            if m.lastindex and m.lastindex >= 1 and m.group(1):
                value = int(m.group(1))
                logger.info("_extract_days_from_query: found days=%d", value)
                return value, None
            elif "half" in pattern:
                logger.info("_extract_days_from_query: found days=0.5 (half day)")
                return 0.5, None
            elif "week" in pattern:
                logger.info("_extract_days_from_query: found days=7 (a week)")
                return 7, None
            elif "day" in pattern:
                logger.info("_extract_days_from_query: found days=1 (a day)")
                return 1, None

    pending_slot = {
        "entity": "employee_leave",
        "field": "days",
        "code_type": None,
        "user_question": "How many days of leave would you like to take?",
    }
    logger.info("_extract_days_from_query: no days found — pending_slot created")
    return None, pending_slot


# Month-name to number mapping for deterministic date parsing
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _extract_date_from_query(user_query: str) -> Optional[str]:
    """Deterministically extract a leave start date from the query.

    Fast-path (closed-form values only — never defaults):
      - ISO dates: 2026-09-30
      - Month-name dates: "Sep 30", "September 30", "30 Sep", "30 September"
      - Relative keywords: today, tomorrow

    Returns yyyy-mm-dd or None (caller must then try LLM fallback, then
    surface a pending_slot — never silently default to today).
    """
    q = user_query.lower()

    # ISO date
    m = re.search(r'\b(\d{4})-(\d{2})-(\d{2})\b', q)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # "Sep 30" / "September 30" (with optional year)
    m = re.search(
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\s+(\d{1,2})(?:\s*,?\s*(\d{4}))?\b',
        q,
    )
    if m:
        month = _MONTHS.get(m.group(1)[:3])
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else datetime.now().year
        if month and 1 <= day <= 31:
            try:
                return datetime(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                pass

    # "30 Sep" / "30 September" (with optional year)
    m = re.search(
        r'\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)(?:\s*,?\s*(\d{4}))?\b',
        q,
    )
    if m:
        month = _MONTHS.get(m.group(2)[:3])
        day = int(m.group(1))
        year = int(m.group(3)) if m.group(3) else datetime.now().year
        if month and 1 <= day <= 31:
            try:
                return datetime(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                pass

    # Relative keywords
    if re.search(r'\btomorrow\b', q):
        return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    if re.search(r'\btoday\b', q):
        return datetime.now().strftime("%Y-%m-%d")

    return None


def _llm_extract_leave_slots(
    user_query: str,
    conversation_history: str,
    missing_fields: List[str],
    leave_type_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Agentic fallback: LLM extraction for leave slots the deterministic
    fast-path could not resolve.

    Follows the documented standard layering:
      deterministic fast-path -> LLM fallback on miss -> pending slot on
      confirmed absence -> never a silent default.

    Args:
        user_query: The user's current query
        conversation_history: Prior turns for context
        missing_fields: Slots to extract: subset of {"date_from", "leave_code", "days"}
        leave_type_map: Optional {code: description} from CodeResolver to
            constrain the LLM to valid tenant codes

    Returns:
        Dict {field: value} for successfully extracted fields (may be empty).
        Values are validated (date format, enum membership) before returning;
        invalid values are discarded rather than defaulted.
    """
    from agent.integrations.llm_client import call_llm

    if not missing_fields:
        return {}

    field_specs = []
    if "date_from" in missing_fields:
        field_specs.append('  "date_from": "<leave start date, format yyyy-mm-dd>"')
    if "leave_code" in missing_fields:
        valid_codes = ", ".join(sorted((leave_type_map or {}).keys()))
        if valid_codes:
            field_specs.append(f'  "leave_code": "<one of: {valid_codes}>"')
        else:
            # No tenant leave-type codes resolved from the database; the LLM
            # cannot be trusted to guess a valid code, so leave the field for a
            # later retry once the map is populated.
            field_specs.append('  "leave_code": null')
    if "days" in missing_fields:
        field_specs.append('  "days": "<number of leave days, decimal allowed e.g. 0.5>"')

    system_prompt = (
        "You extract leave application parameters from a user message. "
        "Return ONLY a JSON object with exactly these keys and null for any "
        "value not explicitly stated by the user:\n"
        "{\n" + ",\n".join(field_specs) + "\n}\n"
        "Rules:\n"
        "- Never guess or infer values not stated by the user.\n"
        "- Use yyyy-mm-dd for dates. Resolve month names (\"Sep 30\" -> next "
        "occurrence of September 30 from today).\n"
        "- leave_code must be exactly one of the listed codes.\n"
        "- Return JSON only, no markdown fences."
    )
    user_prompt = (
        f"Recent conversation:\n{conversation_history[-500:] if conversation_history else 'None'}\n\n"
        f"User message: {user_query}"
    )

    try:
        choice = call_llm(
            system_prompt,
            user_prompt,
            temperature=0.0,
            max_tokens=200,
            tier="planner",
        )
        if not choice:
            logger.warning("_llm_extract_leave_slots: empty LLM response")
            return {}
        content = choice.get("message", {}).get("content", "{}").strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        result = json.loads(content.strip())
    except Exception as exc:
        logger.warning("_llm_extract_leave_slots: LLM extraction failed: %s", exc)
        return {}

    # Deterministic validation of LLM output (validator, not extractor)
    validated: Dict[str, Any] = {}
    if "date_from" in missing_fields:
        raw = result.get("date_from")
        if isinstance(raw, str) and re.match(r'^\d{4}-\d{2}-\d{2}$', raw.strip()):
            validated["date_from"] = raw.strip()
    if "leave_code" in missing_fields:
        raw = result.get("leave_code")
        if isinstance(raw, str):
            raw_u = raw.strip().upper()
            # Only accept a code that is present in the tenant's resolved
            # leave-type map. When the map is empty (no codes resolved from
            # the database) no value is accepted — the LLM cannot be trusted to
            # guess a valid tenant code, and a stale/hardcoded fallback like
            # 'AL' would silently route the request to the wrong leave type.
            if leave_type_map and raw_u in leave_type_map:
                validated["leave_code"] = raw_u
    if "days" in missing_fields:
        raw = result.get("days")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
            validated["days"] = raw
        elif isinstance(raw, str):
            try:
                v = float(raw)
                if v > 0:
                    validated["days"] = v
            except ValueError:
                pass

    logger.info(
        "_llm_extract_leave_slots: extracted %s from missing %s",
        list(validated.keys()), missing_fields
    )
    return validated


def _has_leave_type_in_query(user_query: str, tenant_id: Optional[str]) -> bool:
    """Return whether the query names a specific leave type.

    The bare word "leave" is not a type. Tenant codes and descriptions are
    checked as well as the stable natural-language type names so custom tenant
    leave types are recognized consistently.
    """
    q = user_query.lower()
    if any(
        word in q
        for word in (
            "annual", "medical", "sick", "maternity", "unpaid",
            "compassionate", "emergency", "paternity",
        )
    ):
        return True

    if not tenant_id:
        return False

    try:
        from agent.dab.code_resolver import CodeResolver

        leave_type_map = CodeResolver(tenant_id).resolve_code_type("Leave Type")
        return any(
            code.lower() in q or description.lower() in q
            for code, description in leave_type_map.items()
        )
    except Exception:
        return False


def inject_leave_action_step(
    steps: List[Dict],
    user_query: str,
    auth_context: Any,
    cached_schema: Dict,
    cached_tools: Optional[List[Dict]] = None,
    tone_context: Optional[Dict] = None,
    leave_codes: Optional[List[str]] = None,
    tenant_id: Optional[str] = None,
    conversation_history: str = "",
) -> Tuple[List[Dict], List[Dict]]:
    """Post-processor: if the query is a leave action request but no step uses
    create_record on employee_leave, inject a placeholder create_record step.

    This ensures the executor has an action step to present to the user for
    confirmation, rather than stopping at a balance lookup.

    The injected records are built from the cached DAB schema so they stay
    in sync with required columns. The employee_leave step uses an explicit
    ``$ref`` to chain the ``hd_id`` from the header step the executor creates
    first.
    """
    pending_slots: List[Dict] = []
    if not cached_schema:
        return steps, pending_slots

    # Load tenant-aware config
    action_keywords = get_action_keywords_for_tenant(tenant_id or "default")

    # Trust tone_context first: if the classifier already routed this as an
    # action request, do not let the keyword-based is_action_request override
    # that decision. This avoids false negatives for queries like
    # "apply annual leave" where "annual leave" is also an entitlement keyword.
    effective_category = (
        tone_context.get("intent_category", "policy_info")
        if isinstance(tone_context, dict)
        else "policy_info"
    )
    is_leave_action = effective_category == "action_request" and tone_context.get("intent") == "leave_request"
    if not is_leave_action and not is_action_request(user_query, action_keywords):
        return steps, pending_slots

    # If the user's query is missing critical leave details (date or leave type),
    # do NOT inject a create_record step with defaults. Let the LLM ask the user
    # for the missing information first.
    has_date = _extract_date_from_query(user_query) is not None
    has_leave_type = _has_leave_type_in_query(user_query, tenant_id)

    if not has_date or not has_leave_type:
        logger.info("ACTION_INJECTION_GUARD: skipping injection — user query missing date or leave type")
        # Produce pending_slots so the planner can ask the user for missing info
        if not has_date:
            pending_slots.append({
                "entity": "employee_leave",
                "field": "date_from",
                "code_type": None,
                "user_question": "What date would you like to take leave? (yyyy-mm-dd)",
            })
        if not has_leave_type:
            pending_slots.append({
                "entity": "employee_leave",
                "field": "leave_code",
                "code_type": None,
                "user_question": "What type of leave would you like to apply for? (e.g., Annual Leave, Medical Leave, Emergency Leave)",
            })
        return steps, pending_slots

    has_create = any(
        step.get("tool") == "create_record" and step.get("args", {}).get("entity") == "employee_leave"
        for step in steps
    )
    if has_create:
        return steps, pending_slots

    if "employee_leave" not in cached_schema:
        return steps, pending_slots

    employee_no = None
    if auth_context and getattr(auth_context, "authenticated", False):
        employee_no = getattr(auth_context, "emp_id", None) or getattr(auth_context, "email", None)

    q = user_query.lower()

    # Detect leave type from query.
    # Priority 1: description-based match via CodeResolver (most precise —
    # "annual leave" -> "AL"). Priority 2: direct code substring match
    # (for users who type the code itself, e.g. "apply AL on Friday").
    # leave_codes may be List[str] (codes only) or List[Tuple[str, str]] (code, description).
    leave_code = None
    leave_code_list = [
        (c if isinstance(c, str) else c[0])
        for c in (leave_codes or [])
    ]

    # Priority 1: description-based match against CodeResolver's Leave Type entries.
    # Longest description first so "Annual Leave" wins over "Annual".
    if tenant_id:
        try:
            from agent.dab.code_resolver import CodeResolver
            resolver = CodeResolver(tenant_id)
            leave_type_map = resolver.resolve_code_type("Leave Type")
            for code, desc in sorted(
                leave_type_map.items(), key=lambda kv: len(kv[1]), reverse=True
            ):
                # Match the full description ("annual leave") or the bare noun
                # ("annual" with the " leave" suffix stripped).
                desc_full = desc.lower().strip()
                desc_noun = desc_full.replace(" leave", "").strip()
                if (desc_full and desc_full in q) or (desc_noun and len(desc_noun) >= 3 and desc_noun in q):
                    leave_code = code
                    logger.info(
                        "ACTION_INJECTION: leave type matched by description %r -> %r",
                        desc, code
                    )
                    break
        except Exception as exc:
            logger.warning("ACTION_INJECTION: description-based leave type match failed: %s", exc)

    # Priority 2: direct code match (word-boundary to avoid "al" matching
    # inside "medical" or "annual"). Only codes resolved from the tenant's
    # codesetup are matched — no hardcoded legacy list. If no code matches,
    # the LLM fallback and pending_slot chain handle it (never a default).
    if leave_code is None and leave_code_list:
        for code in leave_code_list:
            if re.search(rf'\b{re.escape(code.lower())}\b', q):
                leave_code = code
                break

    hd_fields = get_entity_fields_from_schema(cached_schema, "employee_leave_hd")
    leave_fields = get_entity_fields_from_schema(cached_schema, "employee_leave")

    # -- Deterministic fast-path extraction (closed-form values only) --------
    date_from = _extract_date_from_query(user_query)
    days_value, days_slot = _extract_days_from_query(user_query)

    # -- LLM fallback for slots the fast-path missed (agentic layering) ------
    missing_fields = []
    if date_from is None:
        missing_fields.append("date_from")
    if leave_code is None:
        missing_fields.append("leave_code")
    if days_value is None:
        missing_fields.append("days")

    if missing_fields:
        try:
            leave_type_map_for_llm = None
            if tenant_id:
                from agent.dab.code_resolver import CodeResolver
                leave_type_map_for_llm = CodeResolver(tenant_id).resolve_code_type("Leave Type")
            llm_slots = _llm_extract_leave_slots(
                user_query,
                conversation_history or "",
                missing_fields,
                leave_type_map=leave_type_map_for_llm,
            )
            if "date_from" in llm_slots and date_from is None:
                date_from = llm_slots["date_from"]
                logger.info("ACTION_INJECTION: date_from=%s from LLM fallback", date_from)
            if "leave_code" in llm_slots and leave_code is None:
                leave_code = llm_slots["leave_code"]
                logger.info("ACTION_INJECTION: leave_code=%s from LLM fallback", leave_code)
            if "days" in llm_slots and days_value is None:
                days_value = llm_slots["days"]
                logger.info("ACTION_INJECTION: days=%s from LLM fallback", days_value)
        except Exception as exc:
            logger.warning("ACTION_INJECTION: LLM slot fallback failed: %s", exc)

    # -- Pending slots for confirmed-absent values (never silent defaults) --
    if date_from is None:
        pending_slots.append({
            "entity": "employee_leave",
            "field": "date_from",
            "code_type": None,
            "user_question": "What date would you like to take leave? (yyyy-mm-dd)",
        })
    if leave_code is None:
        pending_slots.append({
            "entity": "employee_leave",
            "field": "leave_code",
            "code_type": "Leave Type",
            "user_question": "What type of leave would you like to apply for? (e.g., Annual Leave, Medical Leave)",
        })
    if days_value is None and days_slot:
        pending_slots.append(days_slot)

    if pending_slots:
        # Missing required slots — do not inject create steps with defaults.
        logger.info(
            "ACTION_INJECTION_GUARD: missing slots %s — surfacing pending_slots",
            [p.get("field") for p in pending_slots],
        )
        return steps, pending_slots

    days = days_value
    date_to = date_from  # single-day leave; multi-day ranges need explicit end date

    if hd_fields and leave_fields:
        # Schema-driven path: build records from the cached DAB schema.
        now_iso = datetime.now().isoformat()

        # create_by is an Int64 column (system user ID). Use the agentic AI
        # system user ID (1) — NOT the employee_no string, which DAB rejects
        # with "Parameter X cannot be resolved as column create_by with type Int64".
        from agent.config import AGENT_SYSTEM_USER_ID

        hd_fill = {
            "employee_no": employee_no or "",
            "create_by": AGENT_SYSTEM_USER_ID,
            "status": "P",
            "create_date": now_iso,
        }
        hd_record = _build_record_from_schema(
            cached_schema,
            "employee_leave_hd",
            fill_fn=lambda field, entity_name: (hd_fill[field], True) if field in hd_fill else (None, False),
            safe_defaults={"status": "P", "create_date": now_iso},
        )
        if hd_record is None:
            logger.warning("ACTION_INJECTION_GUARD: could not build employee_leave_hd record from schema — skipping injection")
            return steps, pending_slots

        hd_step = {
            "tool": "create_record",
            "args": {
                "entity": "employee_leave_hd",
                "data": hd_record,
            },
            "_step_id": "leave_hd",
        }

        leave_fill = {
            "hd_id": {"$ref": "leave_hd.result.id"},
            "employee_no": employee_no or "",
            "leave_code": leave_code,
            "status": "P",
            "emergency": "N",
            "days": days,
            "period_type_from": 1,
            "period_type_to": 1,
            "date_from": date_from,
            "date_to": date_to,
            "submission_date": now_iso,
            "create_by": AGENT_SYSTEM_USER_ID,
            "create_date": now_iso,
        }
        leave_record = _build_record_from_schema(
            cached_schema,
            "employee_leave",
            fill_fn=lambda field, entity_name: (leave_fill[field], True) if field in leave_fill else (None, False),
            safe_defaults={
                "status": "P",
                "emergency": "N",
                "days": days,
                "period_type_from": 1,
                "period_type_to": 1,
            },
        )
        if leave_record is None:
            return steps, pending_slots

        leave_step = {
            "tool": "create_record",
            "args": {
                "entity": "employee_leave",
                "data": leave_record,
            },
        }
    else:
        # Fallback (no schema): all slots already extracted/validated above;
        # pending_slots were surfaced for any missing values, so here all
        # required values are present — no silent defaults.
        now_iso = datetime.now().isoformat()

        # create_by is Int64 — use the agentic AI system user ID, not the
        # employee_no string (see schema-driven path above).
        from agent.config import AGENT_SYSTEM_USER_ID

        hd_step = {
            "tool": "create_record",
            "args": {
                "entity": "employee_leave_hd",
                "data": {
                    "employee_no": employee_no or "",
                    "status": "P",
                    "create_by": AGENT_SYSTEM_USER_ID,
                    "create_date": now_iso,
                },
            },
            "_step_id": "leave_hd",
        }

        leave_step = {
            "tool": "create_record",
            "args": {
                "entity": "employee_leave",
                "data": {
                    "hd_id": {"$ref": "leave_hd.result.id"},
                    "employee_no": employee_no or "",
                    "leave_code": leave_code,
                    "date_from": date_from,
                    "date_to": date_to,
                    "days": days,
                    "status": "P",
                    "emergency": "N",
                    "period_type_from": 1,
                    "period_type_to": 1,
                    "submission_date": now_iso,
                    "create_by": AGENT_SYSTEM_USER_ID,
                    "create_date": now_iso,
                },
            },
        }

    if cached_tools:
        try:
            from agent.dab.validation import validate_dab_args
            _, schema_error = validate_dab_args("create_record", hd_step["args"], cached_tools)
            if schema_error:
                logger.warning("ACTION_INJECTION_GUARD: hd step failed schema validation: %s", schema_error)
                return steps, pending_slots
            _, schema_error = validate_dab_args("create_record", leave_step["args"], cached_tools)
            if schema_error:
                logger.warning("ACTION_INJECTION_GUARD: leave step failed schema validation: %s", schema_error)
                return steps, pending_slots
        except Exception as exc:
            logger.warning("ACTION_INJECTION_GUARD: schema validation error: %s", exc)

    logger.info(
        "ACTION_INJECTION_GUARD: injected create_record for employee_leave_hd + employee_leave (query=%r)",
        user_query,
    )
    steps.append(hd_step)
    steps.append(leave_step)
    return steps, pending_slots
