"""agent/dab/validation.py

JSON Schema validation for DAB tool arguments before MCP execution.
Validates tool args against the cached tool schemas to catch type errors early.
"""
import json
import logging
from typing import Any, Dict, Optional

from jsonschema import validate, ValidationError, Draft7Validator

logger = logging.getLogger("hr_agent")

# MCP parameters that must be numeric. Coerce from string before schema validation
# because DAB 2.x ReadRecordsTool calls JsonElement.GetInt32() directly and
# crashes on string values like "first": "2".
_NUMERIC_MCP_KEYS = frozenset(("first", "skip", "offset"))


def _coerce_numeric_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce known numeric MCP parameters from string to int/float in place."""
    for key in _NUMERIC_MCP_KEYS:
        if key in args and isinstance(args[key], str):
            try:
                args[key] = int(args[key])
            except (ValueError, TypeError):
                pass
    return args


def _build_schema_validator(tool_schema: Dict) -> Optional[Draft7Validator]:
    """Build a JSON Schema validator from a Groq-style tool schema."""
    if not tool_schema or not isinstance(tool_schema, dict):
        return None

    # Groq tool schema: {"type": "object", "properties": {...}, "required": [...]}
    parameters = tool_schema.get("parameters", tool_schema)
    if not isinstance(parameters, dict):
        return None

    try:
        return Draft7Validator(parameters)
    except Exception as e:
        logger.warning("Failed to build schema validator: %s", e)
        return None


def validate_tool_args(
    tool_name: str,
    args: Dict[str, Any],
    cached_tools: Any,
) -> tuple[Dict, Optional[str]]:
    """
    Validate tool arguments against the cached tool schema.

    First coerces known numeric MCP parameters (first, skip, offset) from
    string to int, because DAB 2.x ReadRecordsTool calls GetInt32() directly
    and crashes on string values.

    Returns:
        (args, None) on success
        (args, error_message) on validation failure
    """
    args = _coerce_numeric_args(args)

    if not cached_tools or not isinstance(cached_tools, list):
        return args, None

    # Find the matching tool schema
    tool_schema = None
    for tool in cached_tools:
        if isinstance(tool, dict) and tool.get("function", {}).get("name") == tool_name:
            tool_schema = tool.get("function", {}).get("parameters")
            break

    if not tool_schema:
        return args, None

    validator = _build_schema_validator(tool_schema)
    if not validator:
        return args, None

    try:
        validator.validate(args)
        return args, None
    except ValidationError as e:
        error_msg = f"Schema validation failed for {tool_name}: {e.message} (path: {list(e.path)})"
        logger.warning(error_msg)
        return args, error_msg
    except Exception as e:
        logger.warning("Unexpected validation error for %s: %s", tool_name, e)
        return args, None


def validate_dab_args(tool_name: str, args: Dict, cached_tools: Any) -> tuple[Dict, Optional[str]]:
    """Alias for validate_tool_args."""
    return validate_tool_args(tool_name, args, cached_tools)
