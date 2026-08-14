import json
import logging

from agent.data.response import extract_items_with_meta

logger = logging.getLogger("hr_agent")


def summarize_aggregation_llm(tool_name: str, data: dict, tool_args: dict,
                             user_query: str, llm_fn) -> str:
    items, count, has_more, end_cursor = extract_items_with_meta(data)

    entity = tool_args.get("entity", "")
    filter_str = tool_args.get("filter", "")

    raw_lines = [f"{tool_name}: {count} aggregation result(s) from '{entity}'"]
    if filter_str:
        raw_lines.append(f"Filter: {filter_str}")
    if has_more:
        raw_lines.append(f"NOTE: More groups available (pagination)")

    if items and isinstance(items[0], dict):
        columns = list(items[0].keys())
        raw_lines.append(f"Columns: {', '.join(columns)}")
        raw_lines.append("Rows:")
        for item in items[:50]:
            pairs = [f"{k}={v}" for k, v in item.items()]
            raw_lines.append("  " + ", ".join(pairs))
        if count > 50:
            raw_lines.append(f"  ... and {count - 50} more rows")
    else:
        raw_lines.append("Raw: " + json.dumps(items[:50], default=str))

    raw_context = "\n".join(raw_lines)

    system = """You are a data summarizer. Given aggregation query results, produce a concise human-readable summary.

RULES:
- Identify what is being grouped by and what is being measured
- Highlight the top categories and any notable patterns
- Use bullet points for observations
- Keep it under 150 words
- Bold key numbers with **markdown**
- If there's a total/sum, mention it
- If there are percentages, show them"""

    prompt = f"""User question: "{user_query}"

Aggregation data:
{raw_context}

Provide a concise summary:"""

    choice = llm_fn(system, prompt, temperature=0.2, max_tokens=600)
    if choice and choice.get("message", {}).get("content"):
        summary = choice["message"]["content"].strip()
        return f"{tool_name} summary:\n{summary}"
    else:
        fallback_lines = [f"{tool_name}: {count} result(s)"]
        for item in items[:10]:
            if isinstance(item, dict):
                pairs = [f"{k}={v}" for k, v in item.items()]
                fallback_lines.append("  " + ", ".join(pairs))
            else:
                fallback_lines.append(f"  {item}")
        return "\n".join(fallback_lines)
