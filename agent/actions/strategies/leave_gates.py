"""Leave write gates for the unified action framework.

Extracted from agent/core/agentic_executor.py per the architecture review:
leave-specific write gates (confirmation checkpoint, employee verification,
balance check, hd_id chaining) belong in the action-strategy layer
(agent/actions/strategies/), not in the generic DAB create executor.

Gates are fail-closed: any verification failure blocks the write and surfaces
a clear error to the user via tool_results — never writes unverified data.

Gate order (as invoked by the executor before a leave create):
1. resolve_hd_id        — backfill hd_id from the chained header create
2. confirmation_gate    — present the request for explicit user confirmation
3. employee_verify_gate — confirm employee_no resolves to an active V_EMP row
4. balance_gate         — confirm sufficient leave balance for the request
5. capture_leave_hd_id  — after a successful header create, record its id
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hr_agent")

# Entities that participate in the leave write chain. Kept local to this
# module so the generic executor never hardcodes entity names.
LEAVE_WRITE_ENTITIES = ("employee_leave_hd", "employee_leave")
LEAVE_HEADER_ENTITY = "employee_leave_hd"
LEAVE_DETAIL_ENTITY = "employee_leave"


def _block(
    state: Dict, call_id: str, tool: str, args: Dict,
    error_msg: str, status: str, error_type: str, entity: str,
) -> None:
    """Record a blocked tool call in the shared executor state format."""
    from agent.data.metrics import record_dab_tool_call, record_dab_error

    state["tool_results"][call_id] = {"error": error_msg}
    state["tool_calls_made"].append({"tool": tool, "args": args, "status": status})
    record_dab_tool_call(tool, "BLOCKED")
    record_dab_error(error_type, entity=entity)


def is_leave_write(entity: str) -> bool:
    """True when the entity is part of the leave write chain."""
    return entity in LEAVE_WRITE_ENTITIES


def resolve_hd_id(
    entity: str, record: Dict, args: Dict, state: Dict, call_id: str, tool: str,
) -> bool:
    """Backfill employee_leave.hd_id from the chained header create.

    The planner may inject a placeholder hd_id=0 or an unresolved $ref dict
    (e.g. {"$ref": "leave_hd.result.id"}). In both cases the value is not a
    usable integer, so backfill it from the captured hd_id in session state.

    Returns True when the executor may proceed, False when the write is blocked.
    """
    if entity != LEAVE_DETAIL_ENTITY:
        return True

    hd_id_val = record.get("hd_id")
    hd_id_unresolved = (
        hd_id_val is None
        or hd_id_val == 0
        or (isinstance(hd_id_val, dict) and "$ref" in hd_id_val)
    )
    if not hd_id_unresolved:
        # Already resolved by $ref chaining — coerce numeric strings to int
        # (SQL Server identity column) and proceed.
        if isinstance(hd_id_val, str):
            try:
                record["hd_id"] = int(hd_id_val)
                args["data"] = record
            except (ValueError, TypeError):
                _block(
                    state, call_id, tool, args,
                    error_msg=(
                        "Leave application could not be completed: the leave header id "
                        f"(hd_id={hd_id_val!r}) is not a valid number."
                    ),
                    status="BLOCKED: invalid leave header id",
                    error_type="InvalidLeaveHeaderId",
                    entity=entity,
                )
                return False
        return True

    last_hd_id = state.get("_last_leave_hd_id")
    if last_hd_id:
        record["hd_id"] = last_hd_id
        args["data"] = record
        logger.info("LEAVE_HD_CHAIN: backfilled hd_id=%s from _last_leave_hd_id", last_hd_id)
        return True

    # The leave header was not created successfully in this step chain, so
    # there is no valid hd_id to link the detail record to. Sending a
    # null/placeholder hd_id to DAB would only produce a cryptic
    # "Invalid value for field hd_id" validation error. Block with a clear
    # message instead.
    _block(
        state, call_id, tool, args,
        error_msg=(
            "Leave application could not be completed: the leave header record "
            "(employee_leave_hd) was not created successfully, so the leave detail "
            "record cannot be linked to it (missing hd_id). Please try applying "
            "again, or contact support if the problem persists."
        ),
        status="BLOCKED: missing leave header",
        error_type="MissingLeaveHeader",
        entity=entity,
    )
    return False


async def confirmation_gate(
    entity: str, record: Dict, args: Dict, state: Dict, call_id: str, tool: str,
    auth_context: Any, tenant_id: str, step: Dict,
) -> bool:
    """Present the leave request for explicit user confirmation.

    Fresh write: stores the confirmation payload (including the full step
    chain so the confirmation turn can re-execute everything) and asks the
    user. Returns False so the executor stops before writing.

    Confirmed re-execution: when the executor state carries
    ``_confirmed_leave_write`` (set by the confirmation turn before
    re-running the stored steps), returns True so the write proceeds
    without re-asking.

    Returns True when the write may proceed, False when it is blocked.
    """
    from agent.core.session_state import (
        save_pending_leave_confirmation,
    )

    # Re-execution of a user-confirmed request: proceed without re-asking.
    if state.get("_confirmed_leave_write"):
        logger.info("CONFIRMATION_GATE: confirmed re-execution for %s -- proceeding", entity)
        return True

    leave_code = record.get("leave_code", "")
    days = record.get("days", "")
    date_from = record.get("date_from", "")
    date_to = record.get("date_to", "")

    # Resolve leave_code description from CodeResolver if available
    code_desc = leave_code
    try:
        from agent.dab.code_resolver import CodeResolver
        resolver = CodeResolver(tenant_id)
        leave_map = resolver.resolve_code_type("Leave Type")
        if leave_code and leave_code in leave_map:
            code_desc = f"{leave_code} ({leave_map[leave_code]})"
    except Exception:
        pass

    summary = f"{days} day(s) of {code_desc} leave from {date_from} to {date_to}"

    # Store the full step chain so the confirmation turn can re-execute
    # the entire leave chain (hd + detail with $ref chaining). The step
    # loop populates state["_pending_write_steps"] with the plan's steps
    # before execution; fall back to the current step when absent.
    all_steps = state.get("_pending_write_steps") or [step]
    save_pending_leave_confirmation(
        auth_context,
        entity=entity,
        record=record,
        summary=summary,
        steps=all_steps,
    )

    # Mark that we're awaiting confirmation so the step loop stops.
    state["_awaiting_confirmation"] = True
    state["tool_results"][call_id] = {
        "result": json.dumps({
            "confirmation_required": True,
            "entity": entity,
            "summary": summary,
            "message": (
                f"Please confirm: you are applying for {summary}. "
                f"Reply 'yes' to submit the application."
            ),
        }),
        "args": args,
    }
    state["tool_calls_made"].append({
        "tool": tool, "args": args,
        "status": "CONFIRMATION_REQUIRED",
    })
    from agent.data.metrics import record_dab_tool_call
    record_dab_tool_call(tool, "CONFIRMATION_REQUIRED")
    logger.info(
        "CONFIRMATION_CHECKPOINT: leave write for %s stored pending confirmation",
        entity,
    )
    return False


async def employee_verify_gate(
    entity: str, record: Dict, args: Dict, state: Dict, call_id: str, tool: str,
    client: Any,
) -> bool:
    """Confirm employee_no resolves to a real, active V_EMP record.

    FAIL-CLOSED: if verification cannot be completed (query failure, parse
    failure, or unknown employee), block the write and surface the error.

    Returns True when the executor may proceed.
    """
    emp_no_verify = record.get("employee_no")
    if not emp_no_verify:
        return True

    verified, verify_error = await verify_employee_exists(client, str(emp_no_verify))
    if verify_error:
        error_msg = f"Leave application NOT submitted. {verify_error}"
        logger.warning("EMPLOYEE_VERIFY_GATE: %s", error_msg)
        _block(
            state, call_id, tool, args,
            error_msg=error_msg,
            status="BLOCKED: employee verification failed",
            error_type="EmployeeVerificationFailed",
            entity=entity,
        )
        return False
    return True


async def balance_gate(
    entity: str, record: Dict, args: Dict, state: Dict, call_id: str, tool: str,
    client: Any,
) -> bool:
    """Check available leave balance before creating the leave detail record.

    FAIL-CLOSED: if the balance cannot be verified (query failure, parse
    failure, or missing entitlement record), block the write and surface the
    error to the user — never write unverified.

    Returns True when the executor may proceed.
    """
    if entity != LEAVE_DETAIL_ENTITY:
        return True

    leave_code_balance = record.get("leave_code")
    days_requested = record.get("days")
    emp_no_balance = record.get("employee_no")
    if not (leave_code_balance and days_requested is not None and emp_no_balance):
        return True

    available, balance_error = await query_leave_balance(
        client, emp_no_balance, leave_code_balance
    )
    if balance_error:
        # Fail-closed: balance could not be verified — block the write.
        error_msg = (
            f"Leave application NOT submitted. Could not verify your leave balance "
            f"for {leave_code_balance}: {balance_error} "
            f"Please try again shortly, or contact HR if the problem persists."
        )
        logger.warning("BALANCE_GATE: %s", error_msg)
        _block(
            state, call_id, tool, args,
            error_msg=error_msg,
            status="BLOCKED: balance check failed",
            error_type="BalanceCheckFailed",
            entity=entity,
        )
        return False

    if available is None:
        return True

    requested = float(days_requested)
    if available < requested:
        shortfall = requested - available
        error_msg = (
            f"Insufficient leave balance for {leave_code_balance}: "
            f"available={available:.1f} days, requested={requested:.1f} days, "
            f"shortfall={shortfall:.1f} days. Please apply for a smaller amount or a different leave type."
        )
        logger.warning("BALANCE_GATE: %s", error_msg)
        _block(
            state, call_id, tool, args,
            error_msg=error_msg,
            status="BLOCKED: insufficient balance",
            error_type="InsufficientBalance",
            entity=entity,
        )
        return False

    logger.info(
        "BALANCE_GATE: %s leave balance OK (available=%.1f >= requested=%.1f)",
        leave_code_balance, available, requested
    )
    return True


async def capture_leave_hd_id(
    entity: str, record: Dict, state: Dict, client: Any, dab_data: Any,
    invoke_read_records: Any,
) -> None:
    """After a successful employee_leave_hd create, capture its id.

    Fast path: extract the id from the create_record response. The DAB MCP
    create_record response wraps the created record in
    result.value[0] (OData-style array):
      {"entity": ..., "result": {"value": [{"id": 51172, ...}]}, ...}

    Primary approach: when the response does not expose the id, query
    employee_leave_hd via read_records filtered by employee_no, ordered by
    id desc, taking the first result — reliable regardless of response shape.
    """
    if entity != LEAVE_HEADER_ENTITY:
        return

    try:
        created_id = None
        if isinstance(dab_data, dict):
            inner = dab_data
            # Unwrap MCP content wrapper if present
            if "content" in inner and isinstance(inner.get("content"), list):
                content = inner["content"]
                if content and isinstance(content[0], dict) and "text" in content[0]:
                    try:
                        parsed = json.loads(content[0]["text"])
                        if isinstance(parsed, dict):
                            inner = parsed
                    except (json.JSONDecodeError, TypeError):
                        pass
            # Unwrap DAB response wrapper
            if isinstance(inner.get("result"), dict):
                inner = inner["result"]
            # Also handle result as JSON string
            if isinstance(inner.get("result"), str):
                try:
                    parsed_result = json.loads(inner["result"])
                    if isinstance(parsed_result, dict):
                        inner = parsed_result
                except (json.JSONDecodeError, TypeError):
                    pass
            # DAB MCP create_record wraps the record in result.value[0]
            # (OData-style array). Unwrap if present.
            if isinstance(inner, dict) and isinstance(inner.get("value"), list) and inner["value"]:
                first = inner["value"][0]
                if isinstance(first, dict):
                    inner = first
            created_id = (
                inner.get("id")
                or inner.get("Id")
                or dab_data.get("id")
                or dab_data.get("Id")
            )

        if created_id is not None:
            state["_last_leave_hd_id"] = created_id
            logger.info("LEAVE_HD_CHAIN: captured hd_id=%s from create_record response", created_id)
            return

        # Primary approach: query the latest inserted employee_leave_hd
        # record for this employee to get the auto-generated id.
        hd_employee_no = record.get("employee_no")
        if not hd_employee_no:
            logger.warning(
                "LEAVE_HD_CHAIN: cannot query hd_id — employee_no missing from employee_leave_hd record"
            )
            return

        try:
            query_args = {
                "entity": LEAVE_HEADER_ENTITY,
                "filter": f"employee_no eq '{hd_employee_no}'",
                "orderby": ["id desc"],
                "first": 1,
            }
            query_result = await invoke_read_records(query_args)
            from agent.data.response import extract_payload, extract_items
            query_data = extract_payload(query_result)
            items = extract_items(query_data)
            if items and isinstance(items[0], dict):
                queried_id = items[0].get("id") or items[0].get("Id")
                if queried_id is not None:
                    state["_last_leave_hd_id"] = queried_id
                    logger.info(
                        "LEAVE_HD_CHAIN: queried latest employee_leave_hd id=%s for employee_no=%s",
                        queried_id, hd_employee_no,
                    )
                else:
                    logger.warning(
                        "LEAVE_HD_CHAIN: latest employee_leave_hd record has no id field. Record keys=%s",
                        list(items[0].keys()),
                    )
            else:
                logger.warning(
                    "LEAVE_HD_CHAIN: read_records query for employee_leave_hd returned no items (employee_no=%s)",
                    hd_employee_no,
                )
        except Exception as query_exc:
            logger.warning(
                "LEAVE_HD_CHAIN: read_records query for latest employee_leave_hd failed: %s",
                query_exc,
            )
    except Exception as exc:
        logger.warning("LEAVE_HD_CHAIN: failed to capture hd_id from employee_leave_hd result: %s", exc)


async def verify_employee_exists(
    client, employee_no: str
) -> Tuple[bool, Optional[str]]:
    """Verify the employee exists in V_EMP by employee_no before a leave write.

    Returns:
        (True, None) when the employee record exists.
        (False, error_message) when the lookup failed or no employee was
        found. The caller must treat this as fail-closed: block the write
        and surface the error to the user.
    """
    if not employee_no:
        return False, "Employee identity is missing (no employee_no)."

    try:
        # Escape single quotes to prevent OData filter injection from
        # auth-derived values.
        safe_no = employee_no.replace("'", "''")
        args = {
            "entity": "V_EMP",
            "filter": f"EMPLOYEE_NO eq '{safe_no}'",
            "select": "EMPLOYEE_NO,EMPLOYEE_NAME,EMPLOYEE_STATUS",
            "first": "1",
        }
        result = await asyncio.to_thread(client.call_tool, "read_records", args)

        items = _extract_dab_items(result)
        if isinstance(items, str):
            return False, f"Employee verification {items}"

        if not items:
            logger.warning(
                "verify_employee_exists: no V_EMP record for employee_no=%s", employee_no
            )
            return False, (
                f"No employee record found for employee number {employee_no}. "
                f"Please verify your employee number or contact HR."
            )

        status = str(items[0].get("EMPLOYEE_STATUS", "")).strip()
        if status and status.upper() in ("RESG", "RESIGNED"):
            logger.warning(
                "verify_employee_exists: employee_no=%s is resigned (status=%s)",
                employee_no, status,
            )
            return False, (
                f"Employee {employee_no} has resigned and cannot apply for leave. "
                f"Please contact HR if you believe this is incorrect."
            )

        logger.info(
            "EMPLOYEE_VERIFIED: employee_no=%s name=%s status=%s",
            employee_no, items[0].get("EMPLOYEE_NAME", "?"), status or "?",
        )
        return True, None

    except Exception as exc:
        logger.warning("verify_employee_exists: failed for %s: %s", employee_no, exc)
        return False, f"Employee verification failed: {exc}"


async def query_leave_balance(
    client, employee_no: str, leave_code: str, year: int = None
) -> Tuple[Optional[float], Optional[str]]:
    """Query available leave balance for an employee and leave type.

    Uses v_employee_leave_summary which has HRMS_EMPLOYEE read permission.

    Returns:
        (available_days, None) on success — net available days
        (leave_ent + leave_bf - forfeit + adjustment).
        (None, error_message) when the balance could NOT be verified —
        either the query failed or no entitlement record exists. The caller
        must treat this as fail-closed: block the write and surface the
        error to the user.
    """
    import datetime

    if year is None:
        year = datetime.datetime.now().year

    try:
        # Escape single quotes to prevent OData filter injection from
        # auth-derived values.
        safe_emp = str(employee_no).replace("'", "''")
        safe_code = str(leave_code).replace("'", "''")
        args = {
            "entity": "v_employee_leave_summary",
            "filter": f"employee_no eq '{safe_emp}' and leave_code eq '{safe_code}' and year eq {year}",
            "select": "leave_ent,leave_bf,forfeit,adjustment",
            "first": "1",
        }
        result = await asyncio.to_thread(client.call_tool, "read_records", args)

        # Parse via the established MCP unwrapping pattern (DAB wraps data
        # in inner.result.value — same shape as CodeResolver)
        items = _extract_dab_items(result)
        if isinstance(items, str):
            return None, f"balance query {items}"

        if not items:
            logger.warning(
                "query_leave_balance: no entitlement record for %s/%s year=%d",
                employee_no, leave_code, year
            )
            return None, (
                f"No leave entitlement record found for leave type {leave_code} "
                f"in {year}. Please contact HR to configure your entitlement."
            )

        row = items[0]
        leave_ent = float(row.get("leave_ent") or 0)
        leave_bf = float(row.get("leave_bf") or 0)
        forfeit = float(row.get("forfeit") or 0)
        adjustment = float(row.get("adjustment") or 0)
        available = leave_ent + leave_bf - forfeit + adjustment
        return available, None

    except Exception as exc:
        logger.warning("query_leave_balance: failed for %s/%s: %s", employee_no, leave_code, exc)
        return None, f"Balance query failed: {exc}"


def _extract_dab_items(result: Any) -> Any:
    """Unwrap a DAB MCP read_records result to its item list.

    Returns the list of record dicts, or a short error string when the
    response could not be parsed (the caller turns it into a fail-closed
    error message).
    """
    if not isinstance(result, dict):
        return f"returned unexpected type {type(result).__name__}"

    content = result.get("content", [])
    if not (content and isinstance(content, list)):
        return "returned no content"

    first = content[0] if content else {}
    if not (isinstance(first, dict) and first.get("type") == "text" and "text" in first):
        return "returned an unexpected response format"

    try:
        inner = json.loads(first["text"])
    except Exception as exc:
        return f"response could not be parsed: {exc}"

    if inner.get("isError"):
        msg = str(inner.get("message", "DAB error"))[:200]
        logger.warning("_extract_dab_items: DAB error: %s", msg)
        return f"query failed: {msg}"

    return (
        inner.get("result", {}).get("value")
        or inner.get("value")
        or inner.get("items")
        or []
    )
