"""
agent/hana_client.py
HTTP-only MCP client for SAP HANA MCP Server.

Transport:
   HANAMCP_HTTP_URL (required in FastAPI)

Result normalization:
   HANA returns {"columns": [...], "rows": [...]} for queries.
   normalize_hana_result() converts this to DAB-style list[dict]
   so downstream agents (summarizer, chart, export) remain unchanged.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("hr_agent")


def hana_columns_rows_to_dicts(
    columns: List[str],
    rows: List[Any],
) -> List[Dict[str, Any]]:
    if not columns or not rows:
        return []

    def _is_header_row(row: Any) -> bool:
        """Detect when a row is a duplicate header (values match column names)."""
        if isinstance(row, dict):
            if len(row) != len(columns):
                return False
            return all(
                str(row.get(col, "")).strip().upper() == str(col).strip().upper()
                for col in columns
            )
        if isinstance(row, (list, tuple)):
            if len(row) != len(columns):
                return False
            return all(
                str(row[i]).strip().upper() == str(columns[i]).strip().upper()
                for i in range(len(columns))
            )
        return False

    def _row_to_dict(row: Any) -> Dict[str, Any]:
        if isinstance(row, dict):
            return dict(row)
        if isinstance(row, (list, tuple)):
            return dict(zip(columns, row))
        return {str(columns[0]): row}

    out = []
    header_stripped = False
    for idx, row in enumerate(rows):
        if not header_stripped and _is_header_row(row):
            logger.warning(
                "HANA header-row detected at index %d (format=%s, cols=%d). Skipping.",
                idx, type(row).__name__, len(columns),
            )
            header_stripped = True
            continue
        out.append(_row_to_dict(row))
    return out


_DESCRIBE_FIELD_PATTERN = re.compile(
    r"(?im)^(?:column_name|field_name|name)\s*[:=]\s*(.+?)$"
)
_DESCRIBE_TYPE_PATTERN = re.compile(
    r"(?im)^(?:data_type|type|data_type_name)\s*[:=]\s*(.+?)$"
)


def _parse_hana_describe_text(text: str) -> Optional[tuple]:
    """Parse describe-table text into (columns, rows).

    Handles common HANA describe/explain text layouts including:
      COLUMN_NAME : ...
      DATA_TYPE   : ...
    and tabular / repeating blocks separated by blank lines.
    """
    if not text or not text.strip():
        return None

    columns = ["column_name", "data_type_name", "description"]
    rows: List[Dict[str, Any]] = []
    current: Dict[str, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                rows.append(current)
                current = {}
            continue

        col_match = _DESCRIBE_FIELD_PATTERN.match(line)
        type_match = _DESCRIBE_TYPE_PATTERN.match(line)

        if col_match:
            if current:
                rows.append(current)
                current = {}
            current["column_name"] = col_match.group(1).strip()
        elif type_match:
            current["data_type_name"] = type_match.group(1).strip()
        elif not current and "," in line:
            # Heuristic: comma-separated header-like line
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if parts:
                current["column_name"] = parts[0]
                if len(parts) > 1:
                    current["data_type_name"] = parts[1]
        elif not current and line:
            current["column_name"] = line

    if current:
        rows.append(current)

    if not rows:
        return None

    # Normalize empty strings and deduplicate column names while preserving order
    seen = set()
    normalized_cols = []
    for col in columns:
        if col not in seen:
            normalized_cols.append(col)
            seen.add(col)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                normalized_cols.append(key)
                seen.add(key)

    return normalized_cols, rows


def normalize_hana_result(result: Any, tool_name: str = "") -> Any:
    if not isinstance(result, dict):
        if isinstance(result, list):
            return {"result": result}
        return {"result": [{"raw": result}]}

    logger.info(
        "HANA_RAW_RESULT: tool=%s type=%s keys=%s",
        tool_name or "unknown",
        type(result).__name__,
        list(result.keys()),
    )

    if result.get("isError"):
        if not result.get("message") or result.get("message") == "Unknown HANA error":
            result["message"] = _format_hana_error(result)
        return result

    cols = result.get("columns")
    rows = result.get("rows")
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        cols = cols if cols is not None else structured.get("columns")
        rows = rows if rows is not None else structured.get("rows")

    if cols is not None and rows is not None:
        if rows and isinstance(rows[0], dict):
            result["result"] = rows
        else:
            result["result"] = hana_columns_rows_to_dicts(cols, rows)
        result["message"] = f"HANA query returned {len(result['result'])} rows"
        logger.debug("HANA normalized %d rows", len(result["result"]))
        logger.info(
            "HANA_NORM_RESULT: tool=%s has_result=True result_len=%d",
            tool_name or "unknown",
            len(result.get("result", [])),
        )
        return result

    # Best-effort fallback for describe-style tools that return text content.
    if tool_name in {"hana_describe_table", "hana_explain_table", "hana_get_sample_data"}:
        content = result.get("content")
        if isinstance(content, list) and content:
            texts = []
            for c in content:
                if isinstance(c, dict):
                    text = c.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
            if texts:
                combined = "\n".join(texts)
                try:
                    parsed = _parse_hana_describe_text(combined)
                    if parsed:
                        cols, rows = parsed
                        if rows and isinstance(rows[0], dict):
                            result["result"] = rows
                        else:
                            result["result"] = hana_columns_rows_to_dicts(cols, rows or [])
                        result["message"] = f"HANA describe returned {len(result['result'])} fields"
                        logger.info(
                            "HANA_NORM_RESULT: tool=%s parsed_describe fields=%d",
                            tool_name,
                            len(result.get("result", [])),
                        )
                        return result
                except Exception as e:
                    logger.debug("HANA describe parse failed for %s: %s", tool_name, e)

    logger.warning(
        "HANA normalize: no columns/rows in result. keys=%r content_types=%r",
        list(result.keys()),
        [
            c.get("type")
            for c in result.get("content", [])
            if isinstance(c, dict)
        ],
    )
    return result


def _format_hana_error(result: Dict[str, Any]) -> str:
    content_text = ""
    content = result.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            text = first.get("text", "")
            if isinstance(text, str) and text.strip():
                content_text = text.strip()

    message = result.get("message", "")
    if not isinstance(message, str):
        message = ""
    if message.strip() and message != "Unknown HANA error":
        return message.strip()

    if content_text:
        lines = [line.strip() for line in content_text.splitlines() if line.strip()]
        sql_code = ""
        sql_state = ""
        for line in lines:
            if line.startswith("sqlCode="):
                sql_code = line
            elif line.startswith("sqlState="):
                sql_state = line
        main_lines = [line for line in lines if not line.startswith("sqlCode=") and not line.startswith("sqlState=")]
        main_msg = " ".join(main_lines) if main_lines else content_text
        parts = [main_msg]
        if sql_code:
            parts.append(sql_code)
        if sql_state:
            parts.append(sql_state)
        return "; ".join(parts)

    parts = []
    for key in ("code", "error", "reason", "detail", "sqlState"):
        val = result.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(f"{key}={val.strip()}")
    raw = result.get("raw")
    if raw:
        parts.append(f"raw={raw}")
    return "; ".join(parts) if parts else "Unknown HANA error"


def extract_hana_items(result: Any) -> List[Dict]:
    if not result:
        return []
    if isinstance(result, dict) and result.get("isError"):
        return []
    normalized = normalize_hana_result(result, tool_name="")
    return [r for r in normalized.get("result", []) if isinstance(r, dict)]


def validate_hana_tool_args(
    tool_name: str,
    args: Dict[str, Any],
    hana_schema_registry: Dict[str, List[str]],
) -> Optional[str]:
    """
    Validate HANA tool arguments against the discovered schema registry.

    Returns:
        Error message string if validation fails, None otherwise.
    """
    schema_name = args.get("schema_name")
    table_name = args.get("table_name")

    if not hana_schema_registry:
        if tool_name in ("hana_get_sample_data", "hana_describe_table", "hana_explain_table", "hana_list_tables"):
            return "HANA schema registry is empty. Ensure SchemaRegistryService loaded tenant caches at startup."
        return None

    normalized_schema = schema_name.upper() if schema_name else None
    authorized_keys = {k.upper() for k in hana_schema_registry}

    if normalized_schema and normalized_schema not in authorized_keys:
        authorized = ", ".join(sorted(hana_schema_registry.keys()))
        return f"schema_name '{schema_name}' is not authorized. Authorized schemas: {authorized}"

    if (
        table_name
        and normalized_schema
        and tool_name in ("hana_get_sample_data", "hana_describe_table", "hana_explain_table")
    ):
        normalized_table = table_name.upper()
        authorized_tables = {t.upper() for t in hana_schema_registry.get(normalized_schema, [])}
        if normalized_table not in authorized_tables:
            close_matches = [
                t for t in hana_schema_registry.get(normalized_schema, [])
                if t.upper().startswith(normalized_table[:3]) and t.upper() != normalized_table
            ]
            preview = ", ".join(sorted(hana_schema_registry.get(normalized_schema, []))[:10])
            if close_matches:
                return (
                    f"table_name '{table_name}' not found in schema '{schema_name}'. "
                    f"Did you mean: {', '.join(close_matches)}? "
                    f"Available tables include: {preview}"
                )
            return f"table_name '{table_name}' not found in schema '{schema_name}'. Available tables: {preview}"

    return None


def _looks_like_numeric_type_error(error_text: str) -> bool:
    """Heuristic check for HANA errors caused by aggregating a non-numeric column."""
    if not isinstance(error_text, str):
        return False
    lowered = error_text.lower()
    return any(flag in lowered for flag in [
        "invalid number",
        "not a numeric type",
        "only numeric type is available",
        "type mismatch",
        "cannot convert",
        "invalid numeric",
        "numeric overflow",
        "invalid argument for function",
        "inconsistent datatype",
    ])


def _wrap_aggregate_fields_with_cast(query: str) -> str:
    """Best-effort SQL sanitizer: wrap aggregate target fields with CAST(... AS DECIMAL).

    This is intentionally conservative: it only rewrites known aggregate patterns,
    and only when the query does not already contain an explicit CAST.
    """
    if not isinstance(query, str):
        return query
    if "cast(" in query.lower():
        return query

    import re
    pattern = re.compile(r"\b(SUM|AVG|MIN|MAX)\(([^)]+)\)", re.IGNORECASE)
    replacements: List[str] = []

    def _replacer(match: re.Match) -> str:
        func = match.group(1)
        field = match.group(2).strip()
        if any(ch.isalpha() for ch in field):
            replacements.append(f"{func}(CAST({field} AS DECIMAL))")
            return f"{func}(CAST({field} AS DECIMAL))"
        return match.group(0)

    new_query = pattern.sub(_replacer, query)
    if replacements:
        logger.info("HANA_SQL_CAST: rewrote aggregate targets: %s", replacements)
    return new_query





class _HttpMpcClient:
    def __init__(self, base_url: str, timeout: int = 30):
        for suffix in ("/mcp/sse", "/mcp/message", "/mcp"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self._http = requests.Session()

    def initialize(self) -> None:
        if self.session_id:
            return
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "hr-ai-agent-hana", "version": "1.0.0"},
            },
        }
        result = self._post(payload)
        if isinstance(result, dict):
            sid = (
                result.get("sessionId")
                or (result.get("result", {}).get("sessionId") if isinstance(result.get("result"), dict) else None)
            )
            if sid:
                self.session_id = sid
                logger.info("HANA HTTP session initialized: %s", sid[:8] + "...")

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if not self.session_id:
            self.initialize()
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        headers = {"Mcp-Session-Id": self.session_id} if self.session_id else {}
        return self._post(payload, headers=headers)

    def list_tools(self) -> Any:
        if not self.session_id:
            self.initialize()
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/list",
            "params": {},
        }
        headers = {"Mcp-Session-Id": self.session_id} if self.session_id else {}
        return self._post(payload, headers=headers)

    def close(self) -> None:
        self.session_id = None
        self._http.close()

    def _post(self, payload: Dict, headers: Optional[Dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/mcp"
        req_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if headers:
            req_headers.update(headers)

        last_error = None
        for attempt in range(3):
            try:
                resp = self._http.post(
                    url,
                    data=json.dumps(payload),
                    headers=req_headers,
                    timeout=self.timeout,
                    stream=True,
                )
                resp.raise_for_status()
                return _parse_mcp_response(resp)
            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.warning("HANA HTTP connection failed (attempt %d): %s", attempt + 1, e)
                time.sleep(0.5 * (2 ** attempt))
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning("HANA HTTP timeout (attempt %d): %s", attempt + 1, e)
                time.sleep(1.0)
            except requests.exceptions.HTTPError as e:
                logger.error("HANA HTTP error: %s", e)
                raise
            except Exception as e:
                last_error = e
                logger.error("HANA HTTP request failed (attempt %d): %s", attempt + 1, e)
                raise

        raise RuntimeError(f"HANA HTTP request failed after 3 retries: {last_error}")


def _parse_mcp_response(response: requests.Response) -> Dict[str, Any]:
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type and "event-stream" not in content_type:
        try:
            data = response.json()
            if isinstance(data, dict) and "jsonrpc" in data and "result" in data:
                return data["result"]
            return data
        except (json.JSONDecodeError, ValueError):
            pass

    events = []
    current_event = {}
    for line in response.iter_lines(decode_unicode=True):
        line = line.strip() if line else ""
        if not line:
            if current_event:
                events.append(current_event)
                current_event = {}
            continue
        if line.startswith("event:"):
            current_event["event"] = line[6:].strip()
        elif line.startswith("data:"):
            data_str = line[5:].strip()
            try:
                current_event["data"] = json.loads(data_str)
            except (json.JSONDecodeError, TypeError):
                current_event["data_raw"] = data_str

    if current_event:
        events.append(current_event)

    for evt in events:
        payload = evt.get("data")
        if isinstance(payload, dict):
            if "jsonrpc" in payload and "result" in payload:
                return payload["result"]
            return payload
        raw = evt.get("data_raw")
        if raw:
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict) and "jsonrpc" in payload and "result" in payload:
                    return payload["result"]
                return payload
            except (json.JSONDecodeError, TypeError):
                return {"raw_text": raw}

    try:
        data = response.json()
        if isinstance(data, dict) and "jsonrpc" in data and "result" in data:
            return data["result"]
        return data
    except (json.JSONDecodeError, ValueError):
        raise RuntimeError(f"Unparseable HANA MCP response: {response.status_code}")


class HanaMcpClient:
    """
    HTTP-only facade for SAP HANA MCP.

    Requires HANAMCP_HTTP_URL to be set.
    Defaults to http://127.0.0.1:3100/mcp for local testing environments.
    """

    def __init__(self, http_url: Optional[str] = None):
        self.http_url = http_url or os.getenv(
            "HANAMCP_HTTP_URL", "http://127.0.0.1:3100/mcp"
        )
        self._http_client = _HttpMpcClient(self.http_url)

    def initialize(self) -> None:
        self._http_client.initialize()

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        return self._http_client.call_tool(tool_name, arguments)

    def list_tools(self) -> Any:
        return self._http_client.list_tools()

    def close(self) -> None:
        self._http_client.close()


class HanaTenantClientManager:
    """
    Registry of per-tenant HANA MCP clients.

    Usage:
        hana_manager = HanaTenantClientManager()
        client = hana_manager.get_client("TENANT_A")
        result = client.call_tool("hana_execute_query", {"query": "SELECT 1"})
    """

    def __init__(self):
        self._clients: Dict[str, HanaMcpClient] = {}

    def get_client(self, tenant_id: str) -> HanaMcpClient:
        if tenant_id not in self._clients:
            self._clients[tenant_id] = HanaMcpClient()
            self._clients[tenant_id].initialize()
            logger.info("HANA HTTP client initialized for tenant: %s", tenant_id)
        return self._clients[tenant_id]

    def close_all(self) -> None:
        for client in self._clients.values():
            try:
                client.close()
            except Exception as e:
                logger.warning("Error closing HANA client: %s", e)
        self._clients.clear()


# Global singleton (must be after class definitions)
hana_manager = HanaTenantClientManager()


# ==============================
# DYNAMIC HANA TOOL DISCOVERY
# ==============================
# Static fallback schemas used when MCP tools/list is unreachable or returns nothing.
_STATIC_HANA_TOOL_SCHEMAS: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "hana_execute_query",
            "description": (
                "Execute SQL against SAP HANA. "
                "Use for queries against SAP HANA schemas and tables. "
                "Supports SELECT/WITH; results include columns and rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute against HANA"},
                    "maxRows": {"type": "number", "description": "Max rows to return"},
                    "includeTotal": {"type": "boolean", "description": "If true, also return total row count"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hana_query_next_page",
            "description": "Fetch the next page of results from a previous HANA query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Original query string"},
                    "maxRows": {"type": "number", "description": "Max rows to return"},
                    "includeTotal": {"type": "boolean", "description": "If true, also return total row count"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hana_describe_table",
            "description": "Describe the structure of a HANA table (columns, types).",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Table name"},
                    "schema_name": {"type": "string", "description": "Schema name (optional)"},
                    "catalog_database": {"type": "string", "description": "MDC catalog database (optional)"},
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hana_explain_table",
            "description": "Describe a HANA table with business semantics overlay (descriptions, code values).",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Table name"},
                    "schema_name": {"type": "string", "description": "Schema name (optional)"},
                    "catalog_database": {"type": "string", "description": "MDC catalog database (optional)"},
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hana_list_tables",
            "description": "List tables in a HANA schema with optional prefix filter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "schema_name": {"type": "string", "description": "Schema name (optional)"},
                    "prefix": {"type": "string", "description": "Table name prefix filter (optional)"},
                    "limit": {"type": "number", "description": "Max tables to return (optional)"},
                    "offset": {"type": "number", "description": "Pagination offset (optional)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hana_list_schemas",
            "description": "List all available HANA schemas.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hana_get_sample_data",
            "description": "Fetch sample rows from a HANA table (SELECT TOP N).",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Table name"},
                    "schema_name": {"type": "string", "description": "Schema name (optional)"},
                    "limit": {"type": "number", "description": "Number of rows (default 10, max 1000)"},
                },
                "required": ["table_name"],
            },
        },
    },
]

_CACHED_HANA_TOOL_SCHEMAS: Dict[str, List[Dict]] = {}
_HANA_TOOLS_DISCOVERED = False
_HANA_TOOL_SCHEMAS_LOCK = asyncio.Lock()


def _convert_mcp_tool_to_openai(mcp_tool: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an MCP tool definition to OpenAI function-calling format."""
    if not isinstance(mcp_tool, dict):
        return {}
    name = (mcp_tool.get("name") or "").strip()
    if not name:
        return {}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (mcp_tool.get("description") or "").strip(),
            "parameters": mcp_tool.get("inputSchema") or {"type": "object", "properties": {}},
        },
    }


def get_cached_hana_tool_schemas(tenant_id: Optional[str] = None) -> List[Dict]:
    """Return cached HANA tool schemas in OpenAI-compatible format.

    If dynamic discovery has not yet run for the given tenant, falls back
    to static schemas.
    """
    key = (tenant_id or "default").upper()
    schemas = _CACHED_HANA_TOOL_SCHEMAS.get(key)
    if schemas:
        return list(schemas)
    return list(_STATIC_HANA_TOOL_SCHEMAS)


def get_hana_tool_names(tenant_id: Optional[str] = None) -> List[str]:
    """Return the current HANA tool names for the given tenant.

    Derived dynamically from the cached schemas, falling back to the
    static schema names if discovery has not yet run.
    """
    return [
        s["function"]["name"]
        for s in get_cached_hana_tool_schemas(tenant_id=tenant_id)
        if isinstance(s, dict) and s.get("function", {}).get("name")
    ]


async def refresh_hana_tool_schemas(tenant_id: Optional[str] = None, http_url: Optional[str] = None) -> List[Dict]:
    """Discover HANA tools via MCP tools/list and cache them per tenant.

    If the HANA server is unreachable or returns no tools, falls back
    to the static schema list.
    """
    global _HANA_TOOLS_DISCOVERED
    key = (tenant_id or "default").upper()

    async with _HANA_TOOL_SCHEMAS_LOCK:
        if _HANA_TOOLS_DISCOVERED and key in _CACHED_HANA_TOOL_SCHEMAS:
            return list(_CACHED_HANA_TOOL_SCHEMAS[key])

        close_client = False
        client = None
        try:
            if http_url:
                client = HanaMcpClient(http_url)
                close_client = True
            elif tenant_id:
                client = hana_manager.get_client(tenant_id)
            else:
                client = HanaMcpClient()
                close_client = True

            # Run sync requests in thread pool to avoid blocking event loop
            raw_result = await asyncio.to_thread(client.list_tools)
            mcp_tools: List[Dict] = []
            if isinstance(raw_result, dict):
                mcp_tools = raw_result.get("tools", [])
                if not mcp_tools and isinstance(raw_result.get("result"), dict):
                    mcp_tools = raw_result["result"].get("tools", [])
            if isinstance(mcp_tools, list):
                mcp_tools = [t for t in mcp_tools if isinstance(t, dict)]

            if mcp_tools:
                converted = [_convert_mcp_tool_to_openai(t) for t in mcp_tools]
                converted = [t for t in converted if t]
                if converted:
                    _CACHED_HANA_TOOL_SCHEMAS[key] = converted
                    _HANA_TOOLS_DISCOVERED = True
                    logger.info("Discovered %d HANA tools via MCP tools/list for tenant=%s", len(converted), key)
                    return list(converted)

            raise ValueError("MCP tools/list returned empty tools list")
        except requests.exceptions.ConnectionError as e:
            logger.error("HANA MCP server unreachable for tenant=%s: %s", key, e)
            if not close_client and tenant_id and client is not None:
                hana_manager._clients.pop(key, None)
            raise
        except Exception as e:
            sanitized = str(e).replace(str(client._http_client.base_url) if client and hasattr(client, '_http_client') else "", "<redacted>")
            logger.warning("Dynamic HANA tool discovery failed for tenant=%s: %s — using static fallback", key, sanitized)
            _CACHED_HANA_TOOL_SCHEMAS[key] = list(_STATIC_HANA_TOOL_SCHEMAS)
            _HANA_TOOLS_DISCOVERED = True
            return list(_STATIC_HANA_TOOL_SCHEMAS)
        finally:
            if close_client and client is not None:
                client.close()
