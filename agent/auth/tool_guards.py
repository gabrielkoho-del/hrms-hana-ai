"""agent/auth/tool_guards.py

Permission enforcement for tool arguments and result filtering.

Extracted from agent/main.py to break the import cycle:
agent.core.agentic_executor -> agent.main -> agent.core.agentic_executor.

These guards run before/after DAB tool execution:
  * enforce_tool_args   -- pre-execution arg restriction + self-service injection
  * filter_tool_results -- post-execution row filtering (defense in depth)

They depend only on agent.auth.models.AuthContext and
agent.core.tool_planner.validate_dab_filter_permissions, both of which
are leaf-safe (no import back into agent.main or the executor).
"""
import json
import logging
from datetime import datetime
from typing import Optional

from agent.config import AGENT_SYSTEM_USER_ID
from agent.core.tool_planner import validate_dab_filter_permissions

logger = logging.getLogger("hr_agent")


def enforce_tool_args(tool: str, args: dict, auth_context) -> tuple[dict, Optional[str]]:
    """
    Enforce permission-based restrictions on tool arguments before execution.
    Returns (args, error_message). If error_message is set, the tool call is blocked.
    """
    if not auth_context or not auth_context.authenticated:
        return args, None

    # HR/Admin can use any tool without restriction
    if "read:all_employees" in auth_context.permissions:
        return args, None

    # --- aggregate_records: block for read:self if it reveals org-wide data ---
    if tool == "aggregate_records":
        groupby = args.get("groupby", [])
        # If grouping by department without self-filter, it's org-wide
        if groupby and "read:self" in auth_context.permissions:
            # Allow if it has a self-filter
            filt = args.get("filter", "")
            if not _has_self_filter(filt, auth_context):
                return args, (
                    "Access denied: Aggregations across all employees are not available for your role. "
                    "You can only access your own profile information."
                )

    # --- read_records: validate filter permissions ---
    if tool == "read_records":
        filt = args.get("filter", "")
        is_valid, error = validate_dab_filter_permissions(filt, args.get("entity", ""), auth_context)
        if not is_valid:
            return args, error

    # --- create_record: enforce self-service and required fields ---
    if tool == "create_record":
        record = args.get("data", {})
        if not isinstance(record, dict):
            return args, "Invalid create_record arguments: 'data' must be an object"

        entity = args.get("entity", "").lower()

        # For employee self-service, force employee_no from auth context
        if "read:self" in auth_context.permissions and "read:all_employees" not in auth_context.permissions:
            if entity in ("employee_leave", "employee_leave_hd"):
                emp_id = auth_context.emp_id or ""
                email = auth_context.email or ""
                if emp_id:
                    # Coerce to string: DAB's employee_leave.employee_no is a
                    # varchar; sending a JSON number would fail schema validation.
                    record["employee_no"] = str(emp_id)
                    # create_by is an Int64 column (system user ID). Use the
                    # agentic AI system user ID (1) — NOT the employee_no
                    # string, which DAB rejects for Int64 columns.
                    if entity == "employee_leave_hd" and not record.get("create_by"):
                        record["create_by"] = AGENT_SYSTEM_USER_ID
                elif email:
                    record["employee_no"] = email
                    if entity == "employee_leave_hd" and not record.get("create_by"):
                        record["create_by"] = AGENT_SYSTEM_USER_ID
                else:
                    return args, "Cannot determine employee identity for leave application"

        # Validate required fields for leave applications
        if entity == "employee_leave_hd":
            required = ["employee_no"]
            missing = [f for f in required if not record.get(f)]
            if missing:
                return args, f"Missing required fields for leave header: {', '.join(missing)}"

            if not record.get("status"):
                record["status"] = "Pending"
            if not record.get("create_by"):
                # create_by is Int64 — use the agentic AI system user ID (1),
                # not the employee_no string.
                record["create_by"] = AGENT_SYSTEM_USER_ID

        if entity == "employee_leave":
            # hd_id is injected by the executor from the preceding employee_leave_hd
            # create result, so it is not required here.
            # leave_code, date_from, date_to are placeholders (empty) until user provides details.
            # Only require employee_no and days for the placeholder step.
            required = ["employee_no", "days"]
            missing = [f for f in required if not record.get(f)]
            if missing:
                return args, f"Missing required fields for leave application: {', '.join(missing)}"

            # Default status and submission date if not provided
            if not record.get("status"):
                record["status"] = "Pending"
            if not record.get("submission_date"):
                record["submission_date"] = datetime.now().isoformat()

        args["data"] = record

    return args, None


def _has_self_filter(filter_str: str, auth_context) -> bool:
    """Check if an OData filter contains a self-referential constraint.

    Checks both EMAIL (case-insensitive) and emp_id/EMPLOYEE_NO value.
    Multi-tenant aware: supports both employee table (emp_id) and V_EMP view (EMPLOYEE_NO).
    """
    if not filter_str or not auth_context:
        return False
    filt_lower = filter_str.lower()
    if auth_context.email and auth_context.email.lower() in filt_lower:
        return True
    if auth_context.emp_id:
        emp_id_str = str(auth_context.emp_id)
        if emp_id_str in filter_str:
            return True
    return False


def filter_tool_results(tool: str, result_text: str, auth_context) -> str:
    """
    Post-execution result filtering based on user permissions.
    Defense in depth: even if filter validation missed something, filter results.
    """
    if not auth_context or not auth_context.authenticated:
        return result_text

    # HR/Admin sees everything
    if "read:all_employees" in auth_context.permissions:
        return result_text

    # Only filter read_records results for now
    if tool != "read_records":
        return result_text

    try:
        data = json.loads(result_text) if isinstance(result_text, str) else result_text
    except Exception:
        return result_text

    if not isinstance(data, dict):
        return result_text

    # DAB read_records returns {"entity": "...", "result": [...], "message": "..."}
    # Fall back to REST-style {"value": [...]} or {"items": [...]}
    items = data.get("items", data.get("value", data.get("result", [])))
    if not isinstance(items, list):
        return result_text

    if not items:
        return result_text

    # For read:self users, strict filtering to own data only
    if "read:self" in auth_context.permissions:
        filtered_items = []
        for item in items:
            if not isinstance(item, dict):
                filtered_items.append(item)
                continue
            is_self = False
            if auth_context.email:
                item_email = str(item.get("EMAIL", item.get("email", ""))).lower()
                if item_email == auth_context.email.lower():
                    is_self = True
            if auth_context.emp_id:
                # V_EMP view uses EMPLOYEE_NO; employee table uses emp_id
                item_emp_id = str(item.get("EMPLOYEE_NO", item.get("emp_id", "")))
                if item_emp_id == str(auth_context.emp_id):
                    is_self = True
            if is_self:
                filtered_items.append(item)

        data["items"] = filtered_items
        data["value"] = filtered_items
        # Preserve pagination info if present
        return json.dumps(data, default=str)

    return result_text
