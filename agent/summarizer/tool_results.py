import json
import logging

from agent.dab.dab_response import extract_items_with_meta

logger = logging.getLogger("hr_agent")


def format_result_summary(tool_name, data, tool_args=None, llm_fn=None):
    tool_args = tool_args or {}
    if "error" in data:
        return f"{tool_name}: Error - {data.get('error', 'Unknown error')}"

    items, count, has_more, end_cursor = extract_items_with_meta(data)
    if count == 0:
        return f"{tool_name}: No results found."

    if items and isinstance(items[0], dict):
        if len(items) == 1 and len(items[0]) == 1:
            key = list(items[0].keys())[0]
            value = items[0][key]
            return f"{tool_name}: {key} = {value}"

        first_keys = list(items[0].keys())
        agg_keys = {'count', 'sum', 'avg', 'min', 'max', 'total'}
        if any(k.lower() in agg_keys for k in first_keys):
            return summarize_aggregation_llm(tool_name, data, tool_args, "", llm_fn)

        if "first_name" in items[0] or "name" in items[0]:
            return format_employee_list(tool_name, items)

        return f"{tool_name}: {count} records"

    raw = json.dumps(data, default=str)
    return f"{tool_name}: {raw[:400]}"


def format_employee_brief(label, emp):
    name = f"{emp.get('first_name', '')} {emp.get('last_name', '')}".strip() or emp.get('name', '')
    parts = [f"{label}: {name}"]
    for key in ['job_title', 'department', 'salary', 'age', 'status']:
        if emp.get(key):
            if key == 'salary':
                parts.append(f"salary ${emp[key]}")
            elif key == 'age':
                parts.append(f"age {emp[key]}")
            elif key == 'status' and emp[key] != 'Active':
                parts.append(f"status: {emp[key]}")
            else:
                parts.append(f"{key.replace('_', ' ')}: {emp[key]}")
    return ", ".join(parts)


def summarize_employee_list(tool_name, emps):
    count = len(emps)
    if count == 0:
        return f"{tool_name}: No employees found."
    depts = {}
    salaries = []
    names = []
    for e in emps:
        dept = e.get('department', 'Unknown')
        depts[dept] = depts.get(dept, 0) + 1
        if e.get('salary'):
            try:
                salaries.append(float(e['salary']))
            except (ValueError, TypeError):
                pass
        names.append(f"{e.get('first_name', '')} {e.get('last_name', '')}".strip() or e.get('name', ''))
    lines = [f"{tool_name}: {count} employees"]
    if len(depts) > 1:
        top_depts = sorted(depts.items(), key=lambda x: -x[1])[:3]
        lines.append(f"Top depts: {', '.join([f'{d}({c})' for d,c in top_depts])}")
    if salaries:
        lines.append(f"Salary range: ${min(salaries):.0f}-${max(salaries):.0f}")
    lines.append(f"Examples: {', '.join(names[:3])}{'...' if count > 3 else ''}")
    return " | ".join(lines)


# NOTE: summarize_aggregation_llm lives in aggregation.py and imports extract_dab_items itself.
# We still need it here for the backward-compatible format_result_summary -> summarize_aggregation_llm call.
from agent.summarizer.aggregation import summarize_aggregation_llm
