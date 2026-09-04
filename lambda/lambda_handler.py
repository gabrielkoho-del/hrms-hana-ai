"""
lambda/lambda_handler.py
AWS Lambda entrypoint for executing LLM-generated forecasting code.

Security model:
  - Minimal safe builtins: math/collections primitives only
  - No __import__, no open, no exec, no eval -- exec() cannot import new modules
  - Data contract: receives a JSON-serialized dict in ``data`` and exposes it
    as ``input_data`` to the user code. This matches the contract used by the
    local SubprocessSandbox (see agent/core/sandbox.py) and the LLM
    prompt (see prompts/forecast_codegen.md).

Event schema (JSON):
  {
    "code": "<Python source string>",
    "data": "<JSON-serialized dict>"
  }

Returns a JSON object with ``result_data`` (the value bound to
``_forecast_result`` in the user code) and ``logs`` (captured stdout).
"""
import io
import json
import sys

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

# Minimal safe builtins: math/collections primitives only.
# No __import__, no open, no exec, no eval -- exec() cannot import new modules.
# This is a strict subset of agent/core/sandbox.py:_SAFE_BUILTINS.
_SAFE_BUILTINS = {
    # constants
    "None": None,
    "True": True,
    "False": False,
    "Ellipsis": Ellipsis,
    # math / logic / collections
    "abs": abs, "all": all, "any": any, "bool": bool,
    "dict": dict, "divmod": divmod, "enumerate": enumerate,
    "filter": filter, "float": float, "format": format, "frozenset": frozenset,
    "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "map": map,
    "max": max, "min": min, "next": next, "pow": pow,
    "print": print, "range": range, "repr": repr, "reversed": reversed,
    "round": round, "set": set, "slice": slice, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "type": type,
    "zip": zip,
    # exceptions (commonly raised/checked by generated code)
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "ZeroDivisionError": ZeroDivisionError,
    "AttributeError": AttributeError,
    "StopIteration": StopIteration,
}

# Modules explicitly exposed to the sandboxed code.
# The LLM prompt also allows json, math, datetime, statistics, collections --
# those are lazy-imported inside the generated script via __import__ is NOT
# available, so we must pre-inject them here.
import collections  # noqa: E402
import datetime  # noqa: E402
import json as _json  # noqa: E402,F401
import math  # noqa: E402
import statistics  # noqa: E402,F401

_RESTRICTED_GLOBALS = {
    "__builtins__": _SAFE_BUILTINS,
    "pd": pd,
    "StatsForecast": StatsForecast,
    "AutoARIMA": AutoARIMA,
    "math": math,
    "datetime": datetime,
    "collections": collections,
    "json": _json,
}


def _parse_input_data(raw: Any) -> Dict[str, Any]:
    """Parse the ``data`` event field into a dict.

    Accepts:
      * A JSON object (dict) -- passed through.
      * A JSON string containing an object -- parsed via json.loads.
      * Empty/None -- returns an empty dict.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"'data' is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("'data' must be a JSON object")
        return parsed
    raise ValueError(f"'data' must be a dict or JSON string, got {type(raw).__name__}")


def lambda_handler(event, context):
    """AWS Lambda entrypoint.

    Expected event:
    {
      "code": "<Python source string>",
      "data": <JSON object or JSON string>
    }

    Returns JSON with result_data (the value bound to ``_forecast_result``)
    and captured print logs.
    """
    if not isinstance(event, dict):
        return {"error": "Event must be a JSON object"}

    code = event.get("code", "")
    if not code:
        return {"error": "Missing 'code' in event"}

    try:
        input_data = _parse_input_data(event.get("data"))
    except ValueError as exc:
        return {"error": str(exc)}

    # Capture print() / stdout from sandboxed code.
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured

    try:
        exec_globals = dict(_RESTRICTED_GLOBALS)
        local_vars: Dict[str, Any] = {}
        exec_globals["input_data"] = input_data
        try:
            exec(code, exec_globals, local_vars)
        except Exception as exc:
            sys.stdout = old_stdout
            return {
                "error": str(exc),
                "logs": captured.getvalue(),
            }
    finally:
        sys.stdout = old_stdout

    # Match the agent sandbox contract: _forecast_result is set at module scope
    # (so it lives in exec_globals, not local_vars).
    result_data = exec_globals.get("_forecast_result", {})

    return {
        "result_data": result_data,
        "logs": captured.getvalue(),
    }
