"""agent/utils/dab_response.py

Consolidated DAB/MCP response extraction utilities.
Handles multiple response wrapper formats from Data API Builder MCP tools.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hr_agent")

# Known DAB response keys that contain data
_DAB_DATA_KEYS = ("result", "value", "items", "data", "records", "rows")

# Keywords that indicate DAB error text
_DAB_ERROR_KEYWORDS = ("not defined", "not found", "Invalid field", "EntityNotFound", "error", "failed")


def extract_items(result: Any) -> List[Dict]:
    """
    Extract a list of dict records from any DAB/MCP response format.

    Handles:
      - Direct list: [{"age": 25}, ...]
      - Dict with known key: {"result": [{"age": 25}, ...]}
      - Nested dicts (up to 2 levels deep)
      - String containing JSON
      - MCP content wrapper: {"content": [{"type": "text", "text": "..."}]}
    """
    if not result:
        return []
    if isinstance(result, list):
        return [r for r in result if isinstance(r, dict)]
    if isinstance(result, dict):
        # Try known DAB keys first
        for key in _DAB_DATA_KEYS:
            if key in result:
                val = result[key]
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                    return val
                nested = extract_items(val)
                if nested:
                    return nested
        # Fallback: any list-of-dicts value in the dict
        for val in result.values():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                return val
            if isinstance(val, dict):
                for nested_key in _DAB_DATA_KEYS:
                    if nested_key in val and isinstance(val[nested_key], list):
                        items = [r for r in val[nested_key] if isinstance(r, dict)]
                        if items:
                            return items
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            return extract_items(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def extract_payload(result: Any) -> Any:
    """
    Unwrap MCP CallToolResult content wrapper to get the inner DAB payload.

    Handles:
      - Direct DAB payload (no wrapper)
      - MCP wrapper: {"content": [{"type": "text", "text": "JSON"}], "isError": bool}
      - JSON-RPC error: {"error": {...}}
      - DAB error in text content without isError flag

    Returns:
      - Parsed dict on success
      - Error dict: {"isError": True, "message": "..."} on failure
      - Original input if unrecognizable
    """
    if not isinstance(result, dict):
        return result

    # JSON-RPC top-level error (MCP bridge failure)
    if "error" in result and not isinstance(result.get("error"), bool):
        err = result["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        logger.error("MCP JSON-RPC error: %s", msg[:200])
        return {"isError": True, "message": msg}

    # Direct isError with message (from invoke_dab_tool_with_retry)
    if result.get("isError") and "message" in result:
        return {"isError": True, "message": result["message"]}

    # MCP content wrapper with isError inside
    if result.get("isError"):
        content = result.get("content", [])
        if content and isinstance(content, list) and len(content) > 0:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return {"isError": True, "message": first["text"]}
        return {"isError": True, "message": "Unknown DAB error"}

    # MCP content wrapper: unwrap the text content
    content = result.get("content", [])
    if content and isinstance(content, list) and len(content) > 0:
        first = content[0]
        if isinstance(first, dict) and first.get("type") == "text" and "text" in first:
            text = first["text"]
            if isinstance(text, str):
                # Detect error text without isError flag
                if any(kw in text for kw in _DAB_ERROR_KEYWORDS):
                    logger.error("DAB returned error in text content: %s", text[:200])
                    return {"isError": True, "message": text}
                try:
                    parsed = json.loads(text)
                    return parsed if isinstance(parsed, dict) else {"result": parsed}
                except json.JSONDecodeError:
                    # Non-JSON text that isn't an error — return as raw message
                    if text and not text.startswith("{"):
                        return {"isError": True, "message": text}

    return result


def is_dab_error(result: Any) -> bool:
    """Return True if the result represents a DAB error."""
    if isinstance(result, dict):
        return bool(result.get("isError"))
    if isinstance(result, str):
        return any(kw in result.lower() for kw in _DAB_ERROR_KEYWORDS)
    return False


def extract_items_with_meta(result: Any) -> Tuple[List[Dict], int, bool, Optional[str]]:
    """
    Extract items plus pagination metadata from a DAB response.

    Returns: (items_list, count, has_next_page, end_cursor)
    """
    if not isinstance(result, dict):
        items = extract_items(result)
        return items, len(items), False, None

    items = result.get("items", result.get("value", result.get("result", [])))
    if not isinstance(items, list):
        items = [items] if items is not None else []

    end_cursor = result.get("endCursor")
    has_next = result.get("hasNextPage", False)
    return items, len(items), has_next, end_cursor


def extract_items_from_tool_results(tool_results: Dict[str, Any]) -> List[Dict]:
    """
    Scan all tool results and return the first non-empty list of items found.
    Used by chart generator and binning logic.
    """
    for tool_name, output in tool_results.items():
        if tool_name.startswith("__"):
            continue
        raw = output.get("result") if isinstance(output, dict) else output
        extracted = extract_items(raw)
        if extracted:
            # Skip DAB error wrappers (type+text columns only)
            if len(extracted) == 1 and set(extracted[0].keys()) == {"type", "text"}:
                logger.warning("Skipping error wrapper in %s: %s", tool_name, extracted[0].get("text", "")[:200])
                continue
            if all(set(row.keys()) == {"type", "text"} for row in extracted):
                logger.warning("Skipping text-only result in %s (likely error)", tool_name)
                continue
            return extracted
    return []


def format_dab_items_context(tool_name: str, items: List[Dict], tool_args: Dict,
                              has_more: bool, end_cursor: Optional[str]) -> List[str]:
    """
    Format DAB items into human-readable context lines for the LLM summarizer.

    Handles zero-row, single-value, and multi-row cases with appropriate
    warnings for missing personal data vs. restrictive filters.
    """
    lines = []
    entity = tool_args.get("entity", "unknown")
    filter_str = tool_args.get("filter", "")

    if not items:
        lines.append(f"{tool_name}: Query returned 0 rows")
        if filter_str:
            lines.append(f"  Filter: {filter_str}")
        lines.append(f"  Entity: {entity}")
        if any(k in entity.lower() for k in ("employee", "leave", "salary", "benefit")) and not tool_args.get("function"):
            lines.append(f"  WARNING: No personal records found in the database. Do NOT use policy documents to infer or substitute for this missing personal data.")
        else:
            lines.append(f"  NOTE: No matching records found. The filter may be too restrictive.")
        return lines

    if len(items) == 1 and len(items[0]) == 1:
        key = list(items[0].keys())[0]
        value = items[0][key]
        lines.append(f"{tool_name}: Single value result")
        lines.append(f"  Entity: {entity}")
        lines.append(f"  Filter: {filter_str}")
        lines.append(f"  Column: {key}")
        lines.append(f"  Value: {value}")
        return lines

    lines.append(f"{tool_name}: {len(items)} record(s) from entity '{entity}'")
    if filter_str:
        lines.append(f"  Filter: {filter_str}")
    if has_more:
        lines.append(f"  NOTE: More records available (pagination). Showing first {len(items)}.")

    lines.append("Raw data:")
    for item in items:
        if isinstance(item, dict):
            fields = [f"{k}={v}" for k, v in item.items() if v is not None]
            lines.append("  " + ", ".join(fields))
        else:
            lines.append(f"  {item}")

    return lines