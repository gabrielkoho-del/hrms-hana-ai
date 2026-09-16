"""Leave-entitlement guard strategy for the unified action framework.

Ensures complete personal leave applications and leave-balance queries use a
leave-entitlement entity instead of falling back to a generic profile lookup
(V_EMP). Incomplete leave applications skip this prefetch and clarify the
missing leave type and start date before any entitlement read or write.

Routing -- the intent classifier's structured output (tone_context) is the
single source of truth; keyword matching is only a supplemental fallback for
informational queries:

1. intent == "leave_request" + intent_category == "action_request"
   and both leave type + start date are present -> inject entitlement.
2. intent == "leave_request" + intent_category == "action_request"
   with either detail missing -> skip entitlement; the action injector asks
   for the missing type/date before any write is prepared.
3. intent_category == "action_request" for any other intent -> skip. Other
   action requests need create/update steps, not a balance lookup.
4. otherwise (informational) -> keyword fallback catches balance questions
   that the classifier routed to a generic informational intent.

Failure visibility: this guard exists to prevent hallucinated leave numbers,
so every path where the guard SHOULD apply but CANNOT inject logs at ERROR
with a distinct LEAVE_ENTITLEMENT_GUARD prefix (no entitlement entity in the
tenant schema, no identity for a self-filter, schema-validation failure).
The executor's pending-slot clarification then presents whatever read data it
has; the gap is diagnosable from logs instead of failing silently.
"""
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.actions.base_injector import get_entity_fields_from_schema
from agent.actions.strategies.leave_action_strategy import (
    _extract_date_from_query,
    _has_leave_type_in_query,
)
from agent.core.config_loader import (
    get_leave_entitlement_entities_for_tenant,
    get_leave_entitlement_entity_preference_for_tenant,
    get_leave_entitlement_keywords_for_tenant,
)
from agent.integrations.hana_sql import _extract_year_from_query

logger = logging.getLogger("hr_agent")

# Display fields for the injected entitlement read, in priority order. The
# select list is intersected with the target entity's actual schema fields,
# so tenants whose views lack the summary columns (e.g.
# v_employee_leave_entitlement has no yearlybalance) still get a valid
# select instead of an "Invalid field" DAB error.
_ENTITLEMENT_DISPLAY_FIELDS = (
    "employee_no", "leave_code", "leave_description", "year",
    "leave_ent", "leave_bf", "forfeit", "adjustment",
    "yearlybalance", "YTDbalance",
)


def _is_leave_entitlement_query(user_query: str, leave_keywords: tuple) -> bool:
    """Supplemental keyword check for informational leave-balance questions.

    NOT used to route action requests -- the intent classifier's structured
    output (tone_context) is the source of truth there; running both checks
    for the same decision let two intent derivations disagree.
    """
    q = user_query.lower()
    return any(kw in q for kw in leave_keywords)


def _should_inject(
    user_query: str,
    tone_context: Optional[Dict],
    leave_keywords: tuple,
    tenant_id: Optional[str] = None,
) -> bool:
    """Decide whether the entitlement guard applies to this query.

    Classifier-first routing (see module docstring); the keyword fallback
    only fires for informational queries.
    """
    intent = (tone_context or {}).get("intent", "")
    category = (tone_context or {}).get("intent_category", "")

    # Complete leave applications can show entitlement before execution.
    # Incomplete applications must clarify type/date first, so an entitlement
    # read here would be wasted and could confuse the clarification response.
    if intent == "leave_request" and category == "action_request":
        has_date = _extract_date_from_query(user_query) is not None
        has_leave_type = _has_leave_type_in_query(user_query, tenant_id)
        return has_date and has_leave_type
    if intent == "leave_request":
        return True
    # Other action requests need create/update steps, not a balance lookup.
    if category == "action_request":
        return False
    # Informational queries: keyword fallback.
    return _is_leave_entitlement_query(user_query, leave_keywords)


def ensure_leave_entitlement_entity(
    steps: List[Dict],
    user_query: str,
    auth_context: Any,
    cached_schema: Dict,
    cached_tools: Optional[List[Dict]] = None,
    tone_context: Optional[Dict] = None,
    tenant_id: Optional[str] = None,
) -> List[Dict]:
    """Inject a leave-entitlement read step when the query needs one and no
    step queries a leave-entitlement entity.

    The entitlement read is INSERTED AT POSITION 0 -- it is the primary
    data for the response (the applicant's balances), ahead of any generic
    profile lookup the planner produced.

    This prevents the planner from answering personal leave questions from
    generic profile lookups (V_EMP) or from hallucinating a number when no
    leave data was fetched.
    """
    if not cached_schema:
        return steps

    # Load tenant-aware config
    leave_keywords = get_leave_entitlement_keywords_for_tenant(tenant_id or "default")
    leave_entities = get_leave_entitlement_entities_for_tenant(tenant_id or "default")
    leave_preference = get_leave_entitlement_entity_preference_for_tenant(tenant_id or "default")

    if not _should_inject(user_query, tone_context, leave_keywords, tenant_id):
        return steps

    # Deferred import: tool_planner imports this module, so a module-level
    # import would be circular. Resolved ONCE here -- never inside the
    # any() loop below.
    from agent.core.tool_planner import _ALL_DAB_TOOLS

    has_leave_entity = any(
        step.get("args", {}).get("entity", "") in leave_entities
        for step in steps
        if step.get("tool") in _ALL_DAB_TOOLS
    )
    if has_leave_entity:
        return steps

    # Pick the best available leave-entitlement entity using the configured
    # preference order, filtered to what exists in the cached schema.
    target_entity = next(
        (name for name in leave_preference if name in cached_schema),
        None,
    )
    if not target_entity:
        # The guard's whole purpose is preventing hallucinated leave
        # numbers; a tenant with no entitlement entity configured is a
        # setup gap that must be visible, not a silent no-op.
        logger.error(
            "LEAVE_ENTITLEMENT_GUARD: query needs entitlement data but tenant %s "
            "schema has none of the configured entitlement entities %s -- "
            "cannot inject; the response will have no leave data",
            tenant_id or "default",
            list(leave_preference),
        )
        return steps

    is_leave_request = (tone_context or {}).get("intent", "") == "leave_request"

    # Build the self-filter. EMPLOYEE_NO is a numeric/alphanumeric employee
    # number -- never an email address. The previous email fallback
    # (EMPLOYEE_NO eq '<email>') returned zero rows and made the summarizer
    # tell users they had no entitlement, so it was removed: without an
    # emp_id there is no valid self-filter and the guard skips injection
    # with an ERROR (surfaced, not silent).
    filt = None
    if auth_context and getattr(auth_context, "authenticated", False):
        is_hr = "read:all_employees" in getattr(auth_context, "permissions", [])
        emp_id = getattr(auth_context, "emp_id", None)
        if emp_id:
            # Leave applications show the APPLICANT's entitlement even for
            # HR users; informational queries keep HR's unfiltered view.
            if is_leave_request or not is_hr:
                safe_emp = str(emp_id).replace("'", "''")
                filt = f"EMPLOYEE_NO eq '{safe_emp}'"
        elif not is_hr:
            logger.error(
                "LEAVE_ENTITLEMENT_GUARD: authenticated user has no emp_id and no "
                "read:all_employees permission -- cannot build a self-filter, "
                "skipping entitlement injection (an email is not an EMPLOYEE_NO; "
                "filtering by it would return zero rows)"
            )
            return steps
        # HR without emp_id (informational query): year-only filter below.

    # Determine year filter: prefer year from query, then current year.
    year_filter = ""
    extracted_year = _extract_year_from_query(user_query)
    if extracted_year:
        year_filter = f" and year eq {extracted_year}"
    else:
        year_filter = f" and year eq {datetime.now().year}"

    # Build the select from the target entity's actual schema fields so the
    # injected step never references columns the tenant's view lacks.
    entity_fields = set(get_entity_fields_from_schema(cached_schema, target_entity))
    select_fields = [f for f in _ENTITLEMENT_DISPLAY_FIELDS if f in entity_fields]
    select = ",".join(select_fields) if select_fields else None

    leave_step: Dict[str, Any] = {
        "tool": "read_records",
        "args": {
            "entity": target_entity,
            "first": "100",
        },
    }
    if select:
        leave_step["args"]["select"] = select
    if filt:
        leave_step["args"]["filter"] = filt + year_filter
    else:
        # year_filter is prefixed with " and "; strip the leading connector
        # so the filter is valid OData. OData $filter has no SQL-style
        # "1=1" tautology, which DAB rejects with a syntax error.
        leave_step["args"]["filter"] = re.sub(r"^\s*and\s+", "", year_filter, flags=re.I)

    # Validate the injected step against the DAB tool schema so it receives
    # the same argument checking as planner-originated steps.
    if cached_tools:
        try:
            from agent.dab.validation import validate_dab_args
            _, schema_error = validate_dab_args("read_records", leave_step["args"], cached_tools)
            if schema_error:
                logger.error(
                    "LEAVE_ENTITLEMENT_GUARD: injected step failed schema validation: %s "
                    "-- cannot inject; the response will have no leave data",
                    schema_error,
                )
                return steps
        except Exception as exc:
            logger.error("LEAVE_ENTITLEMENT_GUARD: schema validation error: %s", exc)

    logger.info(
        "LEAVE_ENTITLEMENT_GUARD: injected read_records for %s at position 0 (query=%r)",
        target_entity,
        user_query,
    )
    # Insert at position 0: the entitlement read is the primary data for the
    # response, ahead of any generic profile lookup the planner produced.
    steps.insert(0, leave_step)
    return steps
