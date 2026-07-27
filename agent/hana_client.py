"""
agent/hana_client.py
HTTP-first MCP client for SAP HANA MCP Server (finance data).

Transport priority:
  1. HANAMCP_HTTP_URL (preferred in FastAPI)
  2. STDIO (legacy/testing)

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
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import requests

logger = logging.getLogger("hr_agent")

# ─── Public tool catalog (matches hana-mcp-server tool names) ────────────────
HANA_TOOL_NAMES = [
    "hana_execute_query",
    "hana_query_next_page",
    "hana_list_schemas",
    "hana_list_tables",
    "hana_describe_table",
    "hana_explain_table",
    "hana_get_sample_data",
]


# ─── Result normalization ────────────────────────────────────────────────────

def hana_columns_rows_to_dicts(
    columns: List[str],
    rows: List[List[Any]],
) -> List[Dict[str, Any]]:
    """Convert HANA {'columns': [...], 'rows': [...]} to DAB-style list[dict]."""
    if not columns or not rows:
        return []
    return [dict(zip(columns, row)) for row in rows]


def normalize_hana_result(result: Any) -> Any:
    """Normalize HANA MCP response into DAB-style format.

    Canonical HANA query result:
        {"columns": [...], "rows": [...], "returnedRows": N, ...}

    Becomes (in-place on the dict for list[dict] wrapping):
        {"result": [{"COL1": v1, ...}, ...], "message": "...", ...}

    Pass-through for non-dict inputs so downstream extract_items() can wrap them.
    """
    if not isinstance(result, dict):
        if isinstance(result, list):
            return {"result": result}
        return {"result": [{"raw": result}]}

    if result.get("isError"):
        return result

    cols = result.get("columns")
    rows = result.get("rows")
    if cols is not None and rows is not None:
        result["result"] = hana_columns_rows_to_dicts(cols, rows)
        result["message"] = f"HANA query returned {len(result['result'])} rows"
        logger.debug("HANA normalized %d rows", len(result["result"]))
        return result

    return result


def extract_hana_items(result: Any) -> List[Dict]:
    """Extract list[dict] from any HANA response wrapper."""
    if not result:
        return []
    if isinstance(result, dict) and result.get("isError"):
        return []
    normalized = normalize_hana_result(result)
    return [r for r in normalized.get("result", []) if isinstance(r, dict)]


# ─── HTTP transport (preferred in FastAPI) ────────────────────────────────────

class _HttpMpcClient:
    """Minimal MCP over HTTP client matching DABClient patterns."""

    def __init__(self, base_url: str, timeout: int = 30):
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
    """Parse MCP server response: JSON, SSE, or raw body."""
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


# ─── Public client facade ────────────────────────────────────────────────────

class HanaMcpClient:
    """
    Async-friendly facade for SAP HANA MCP.

    Prefers HTTP when HANAMCP_HTTP_URL is set; otherwise falls back to STDIO.
    """

    def __init__(
        self,
        http_url: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict] = None,
    ):
        self.http_url = http_url or os.getenv("HANAMCP_HTTP_URL", "")
        self.command = command or "node"
        self.args = args or self._default_args()
        self.env = env or {**os.environ}
        self._http_client: Optional[_HttpMpcClient] = (
            _HttpMpcClient(self.http_url) if self.http_url else None
        )
        self._stdio_process = None

    def _default_args(self) -> List[str]:
        base = os.path.dirname(os.path.abspath(__file__))
        server_path = os.path.join(base, "..", "hana-mcp-server", "hana-mcp-server.js")
        return [os.path.normpath(server_path)]

    def initialize(self) -> None:
        if self._http_client:
            self._http_client.initialize()
        else:
            self._init_stdio()

    def _init_stdio(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )
        self._stdio_context = stdio_client(server_params)

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if self._http_client:
            raw = self._http_client.call_tool(tool_name, arguments)
        else:
            raw = asyncio.get_event_loop().run_until_complete(
                self._call_tool_stdio(tool_name, arguments)
            )
        return normalize_hana_result(raw)

    async def _call_tool_stdio(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return result

    def close(self) -> None:
        if self._http_client:
            self._http_client.close()


# ─── Tenant manager (registry pattern for executor) ──────────────────────────

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
            logger.info("HANA client initialized for tenant: %s", tenant_id)
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
