"""
agent/dab_client.py
MCP client that talks to the MCP-DAB Bridge server.
"""
import json
import logging
import os
import re
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import requests

# Per-tenant DAB routing
from agent.auth.tenant_resolver import resolver

logger = logging.getLogger("hr_agent")

STRING_ID_FIELDS = frozenset(("employee_no",))

@dataclass
class DABSession:
    session_id: str
    base_url: str
    initialized: bool = False

class DABClient:
    """
    MCP client for the DAB Bridge server.
    """
    def __init__(self, base_url: str, timeout: int = 30, max_retries: int = 3):
        self.base_url = self._sanitize_base_url(base_url.rstrip("/"))
        self.timeout = timeout
        self.max_retries = max_retries
        self.session: Optional[DABSession] = None
        self._http = requests.Session()
        self._sse_response: Optional[requests.Response] = None

    @staticmethod
    def _sanitize_base_url(url: str) -> str:
        """Strip any trailing MCP transport path so we never double-path."""
        for suffix in ("/mcp/sse", "/mcp/message", "/mcp"):
            if url.endswith(suffix):
                return url[: -len(suffix)]
        return url

    @staticmethod
    def _coerce_mcp_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce known numeric MCP parameters from string to int/float.

        DAB 2.x ReadRecordsTool calls JsonElement.GetInt32() directly,
        so passing \"first\": \"2\" crashes. Force numeric types here.
        """
        coerced = dict(arguments)
        for key in ("first", "skip", "offset"):
            if key in coerced and isinstance(coerced[key], str):
                try:
                    coerced[key] = int(coerced[key])
                except (ValueError, TypeError):
                    pass
        return coerced

    def _make_request(
        self,
        method: str,
        jsonrpc_request: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        # DAB 2.0.8 uses a single /mcp endpoint. Session state is tracked
        # via the Mcp-Session-Id header returned on initialize.
        url = f"{self.base_url}/mcp"

        req_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if headers:
            req_headers.update(headers)

        # Attach session header for non-initialize requests
        if jsonrpc_request and jsonrpc_request.get('method') != 'initialize':
            if self.session and self.session.session_id:
                req_headers["Mcp-Session-Id"] = self.session.session_id

        payload = json.dumps(jsonrpc_request) if jsonrpc_request else None
        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.debug("MCP request to %s (attempt %d)", url, attempt + 1)
                response = self._http.post(
                    url,
                    data=payload,
                    headers=req_headers,
                    timeout=self.timeout,
                    stream=True
                )
                response.raise_for_status()

                # Capture session ID from initialize response header
                if jsonrpc_request and jsonrpc_request.get('method') == 'initialize':
                    session_id = response.headers.get('Mcp-Session-Id') or response.headers.get('mcp-session-id')
                    if session_id:
                        self.session = DABSession(session_id=session_id, base_url=self.base_url, initialized=True)
                        logger.info("MCP session initialized: %s", session_id[:8] + "...")

                return self._parse_sse_response(response)

            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {e}"
                logger.warning("MCP connection failed (attempt %d): %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))
            except requests.exceptions.Timeout as e:
                last_error = f"Timeout: {e}"
                logger.warning("MCP timeout (attempt %d): %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    time.sleep(1.0)
            except requests.exceptions.HTTPError as e:
                if response.status_code == 404 and self.session:
                    logger.warning("Session expired or invalid, re-initializing...")
                    self.session = None
                    self.initialize()
                    if attempt < self.max_retries - 1:
                        continue
                raise
            except Exception as e:
                last_error = f"Request error: {e}"
                logger.error("MCP request failed (attempt %d): %s", attempt + 1, e)
                raise

        raise RuntimeError(f"MCP request failed after {self.max_retries} retries: {last_error}")

    def _fetch_sse_session_id(self) -> Optional[str]:
        """Open a temporary SSE connection to /mcp/sse and extract session ID."""
        sse_url = f"{self.base_url}/mcp/sse"
        try:
            with requests.get(sse_url, stream=True, timeout=self.timeout) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if "sessionId=" in data:
                            sid = data.split("sessionId=")[1].split(" ")[0].split("\n")[0]
                            logger.info("MCP session fetched via SSE: %s", sid[:8] + "...")
                            return sid
        except Exception as e:
            logger.debug("Could not fetch SSE session ID: %s", e)
        return None

    def _parse_sse_response(self, response: requests.Response) -> Dict[str, Any]:
        """Parse MCP server response according to JSON-RPC 2.0 over SSE spec."""
        
        # Path 1: Plain JSON (non-SSE)
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type and "event-stream" not in content_type:
            try:
                data = response.json()
                if isinstance(data, dict) and "jsonrpc" in data and "result" in data:
                    return data["result"]
                return data
            except (json.JSONDecodeError, ValueError):
                pass

        # Path 2: SSE with event/data markers
        events = []
        current_event = {}
        has_sse_markers = False

        for line in response.iter_lines(decode_unicode=True):
            if line is None:
                continue
            line = line.strip()
            if not line:
                if current_event:
                    events.append(current_event)
                    current_event = {}
                continue

            if line.startswith("event:") or line.startswith("data:"):
                has_sse_markers = True
            if line.startswith("event:"):
                current_event["event"] = line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].strip()
                try:
                    current_event["data"] = json.loads(data_str)
                except json.JSONDecodeError:
                    current_event["data_raw"] = data_str

        if current_event:
            events.append(current_event)

        # Process events: unwrap JSON-RPC envelope from data field
        for evt in events:
            # Try parsed dict first
            payload = evt.get("data")
            if isinstance(payload, dict):
                if "jsonrpc" in payload and "result" in payload:
                    return payload["result"]
                return payload
            
            # Try raw string (parse failed above, maybe escaped JSON)
            raw = evt.get("data_raw")
            if raw:
                try:
                    payload = json.loads(raw)
                    if isinstance(payload, dict) and "jsonrpc" in payload and "result" in payload:
                        return payload["result"]
                    return payload
                except json.JSONDecodeError:
                    return {"raw_text": raw}

        # Path 3: No SSE markers, raw JSON body
        try:
            data = response.json()
            if isinstance(data, dict) and "jsonrpc" in data and "result" in data:
                return data["result"]
            return data
        except (json.JSONDecodeError, ValueError):
            logger.error("Failed to parse response as SSE or JSON. Status: %s", response.status_code)
            raise RuntimeError(f"Unparseable MCP response: {response.status_code}")

    def initialize(self) -> None:
        if self.session and self.session.initialized:
            return

        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "hr-ai-agent", "version": "1.0.0"}
            }
        }
        result = self._make_request("POST", request)
        logger.info("MCP bridge initialized: %s", result.get("result", {}).get("serverInfo", {}).get("name", "unknown"))

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Invoke a DAB tool over MCP."""
        if not self.session or not self.session.initialized:
            self.initialize()

        arguments = self._coerce_mcp_arguments(arguments)
        return self._call_tool_mcp(tool_name, arguments)

    def _call_tool_mcp(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Original MCP tools/call path (unchanged behavior)."""
        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        headers = {}
        if self.session and self.session.session_id:
            headers["Mcp-Session-Id"] = self.session.session_id
            logger.info("Sending MCP request with session: %s", self.session.session_id[:8])
        else:
            logger.warning("No MCP session ID available!")

        result = self._make_request("POST", request, headers=headers)

        # PATCH D2: isError is inside content[0].text, not the wrapper.
        # Let agentic_executor._extract_dab_payload handle unwrapping.
        # Only check JSON-RPC level errors (MCP bridge errors).
        if "error" in result:
            error = result["error"]
            logger.error("MCP tool error: %s", error)
            raise RuntimeError(f"MCP tool {tool_name} failed: {error}")

        # Return raw DAB wrapper (may contain content with isError inside)
        return result

    def _extract_items_from_result(self, result: Any) -> List[Dict]:
        """Unwrap MCP content wrapper from a read_records response to get data items.

        Follows the same MCP content unwrapping pattern as describe_entities.
        """
        from agent.dab.dab_response import extract_items

        unwrapped = self._unwrap_mcp_content(result)
        return extract_items(unwrapped)

    def _unwrap_mcp_content(self, result: Any) -> Any:
        """Unwrap MCP content wrapper to get the inner DAB payload.

        Handles:
          - Direct DAB payload (no wrapper)
          - MCP wrapper: {"content": [{"type": "text", "text": "JSON"}], "isError": bool}
        """
        if not isinstance(result, dict):
            return result

        # JSON-RPC top-level error
        if "error" in result and not isinstance(result.get("error"), bool):
            err = result["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            logger.error("MCP JSON-RPC error: %s", msg[:200])
            return {"isError": True, "message": msg}

        # Direct isError with message
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
                    # Try to parse as JSON
                    try:
                        parsed = json.loads(text)
                        logger.debug("Unwrapped MCP content wrapper, parsed JSON")
                        return parsed if isinstance(parsed, dict) else {"result": parsed}
                    except json.JSONDecodeError:
                        # Not JSON text, return as raw message
                        logger.warning("MCP content text is not JSON: %s", text[:200])
                        return {"isError": True, "message": text}

        return result

    @staticmethod
    def _infer_type_from_value(val: Any, field_name: str = "") -> str:
        """Map a Python value to its DAB/OData EDM type name.

        Known string-ID fields that contain numeric values are forced to
        ``Edm.String`` so downstream OData filters quote them correctly.
        """
        if field_name and field_name.lower() in STRING_ID_FIELDS:
            return "Edm.String"
        if val is None:
            return "string"
        if isinstance(val, bool):
            return "boolean"
        if isinstance(val, int):
            return "Edm.Int64"
        if isinstance(val, float):
            return "Edm.Double"
        return "string"

    def infer_fields_for_entity(self, entity_name: str) -> List[Dict[str, Any]]:
        """Infer field list by reading 1 record and extracting column names + types.

        Used when describe_entities returns entities without field details
        (e.g., views without explicit 'fields' in DAB config).
        Returns list of {"name": ..., "type": ...} dicts with actual EDM types.
        """
        try:
            result = self.call_tool("read_records", {"entity": entity_name, "first": 1})
            items = self._extract_items_from_result(result)
            if items and isinstance(items[0], dict):
                sample = items[0]
                fields = [
                    {"name": k, "type": self._infer_type_from_value(v, field_name=k)}
                    for k, v in sample.items()
                ]
                logger.info(
                    "Inferred %d fields from sample query for %s: %s",
                    len(fields), entity_name,
                    ", ".join(f["name"] for f in fields[:5])
                    + (f" ... +{len(fields)-5} more" if len(fields) > 5 else "")
                )
                return fields
            else:
                logger.warning(
                    "infer_fields_for_entity(%s): sample query returned no items", entity_name
                )
        except Exception as e:
            logger.warning("Failed to infer fields for %s: %s", entity_name, e)
        return []

    def describe_entities(self) -> List[Dict[str, Any]]:
        """Call DAB's describe_entities to discover all available entities."""
        result = self.call_tool("describe_entities", {})
        
        # Debug: log what _parse_sse_response actually returned
        logger.info("TYPE=%s KEYS=%s HAS_CONTENT=%s", 
                type(result).__name__,
                list(result.keys()) if isinstance(result, dict) else "N/A",
                "content" in result if isinstance(result, dict) else False)  
        
        # After _parse_sse_response, result should be {"content": [...]}
        content = result.get("content", [])
        if content and isinstance(content, list):
            first = content[0]
            if isinstance(first, dict) and first.get("type") == "text" and "text" in first:
                try:
                    parsed = json.loads(first["text"])
                    if isinstance(parsed, dict) and "entities" in parsed:
                        return parsed["entities"] or []
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    logger.warning("Failed to parse describe_entities text content")
                return []
            
            if isinstance(first, dict) and "name" in first:
                return content

        logger.warning("describe_entities: unexpected shape. type=%s", type(result).__name__)
        return []

# ─────────────────────────────────────────────────────────
# TENANT MANAGER
# ─────────────────────────────────────────────────────────

class DABTenantClientManager:
    def __init__(self):
        self._clients: Dict[str, DABClient] = {}
        self._entity_cache: Dict[str, Any] = {}
        self._cache_ttl_seconds = 300
        self._cache_timestamp: Dict[str, float] = {}
        # Fallback URL for emergency or when tenant not in YAML
        self._fallback_url = os.getenv("MCP_SERVER_URL", "http://localhost:5000")

    def get_client(self, tenant_id: str) -> DABClient:
        """Get or create DAB client for tenant. Initialization is lazy;
        first call may block if sync context. For async, use get_client_async."""
        if tenant_id not in self._clients:
            dab_url = resolver.get_tenant_dab_url(tenant_id)
            if not dab_url:
                dab_url = self._fallback_url
            client = DABClient(dab_url)
            client.initialize()
            self._clients[tenant_id] = client
        return self._clients[tenant_id]

    async def get_client_async(self, tenant_id: str) -> DABClient:
        """PATCH D10: Async-safe client initialization."""
        import asyncio
        if tenant_id not in self._clients:
            dab_url = resolver.get_tenant_dab_url(tenant_id)
            if not dab_url:
                dab_url = self._fallback_url
            client = DABClient(dab_url)
            await asyncio.to_thread(client.initialize)
            self._clients[tenant_id] = client
        return self._clients[tenant_id]

    def _enrich_entities_with_fields(self, client: DABClient, entities: List[Dict]) -> List[Dict]:
        """Infer fields for entities that have no field details.

        DAB's describe_entities only returns field info when 'fields' are
        explicitly defined in the DAB config. For views/tables without
        explicit field definitions, we infer fields by reading 1 sample
        record and extracting column names from the response keys.
        This guarantees the schema index and tool planner have correct,
        case-exact field names to use in OData filters.
        """
        enriched = []
        for entity in entities:
            if not isinstance(entity, dict):
                enriched.append(entity)
                continue

            existing_fields = entity.get("fields", entity.get("columns", []))
            if existing_fields:
                enriched.append(entity)
                continue

            entity_name = entity.get("name", "")
            if not entity_name:
                enriched.append(entity)
                continue

            inferred = client.infer_fields_for_entity(entity_name)
            if inferred:
                entity = {**entity, "fields": inferred}
                logger.info("Enriched %s with %d inferred fields", entity_name, len(inferred))

            enriched.append(entity)
        return enriched

    def get_entities(self, tenant_id: str, force_refresh: bool = False) -> Any:
        now = time.time()
        cached = self._entity_cache.get(tenant_id)
        timestamp = self._cache_timestamp.get(tenant_id, 0)

        if cached and not force_refresh and (now - timestamp) < self._cache_ttl_seconds:
            return cached

        client = self.get_client(tenant_id)
        try:
            entities = client.describe_entities()
            # Enrich entities that have no field details
            entities = self._enrich_entities_with_fields(client, entities)
            self._entity_cache[tenant_id] = entities
            self._cache_timestamp[tenant_id] = now
            return entities
        except Exception as e:
            if cached:
                return cached
            raise

    def invalidate_cache(self, tenant_id: str) -> None:
        self._entity_cache.pop(tenant_id, None)
        self._cache_timestamp.pop(tenant_id, None)

    def close_all(self) -> None:
        # PATCH D11: Graceful per-client close to prevent one failure from aborting shutdown
        for client in list(self._clients.values()):
            try:
                client.close()
            except Exception as e:
                logger.warning("Error closing DAB client: %s", e)
        self._clients.clear()
        self._entity_cache.clear()
        self._cache_timestamp.clear()

# ─────────────────────────────────────────────────────────
# ASYNC INVOKER (unchanged)
# ─────────────────────────────────────────────────────────

async def invoke_dab_tool_with_retry(client: DABClient, tool: str, args: dict, retries: int = 2) -> Any:
    """Invoke DAB tool with retry. Uses asyncio.to_thread (Python 3.9+).

    Returns isError format aligned with agentic_executor._extract_dab_payload.
    """
    import asyncio
    last_error = None

    for attempt in range(retries):
        try:
            return await asyncio.to_thread(client.call_tool, tool, args)
        except Exception as e:
            last_error = str(e)
            logger.warning("DAB tool %s failed (attempt %d/%d): %s", tool, attempt + 1, retries, e)
            if attempt < retries - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))

    # Align error format with executor expectation (isError, not error)
    logger.error("DAB tool %s failed after %d retries: %s", tool, retries, last_error)
    return {"isError": True, "tool": tool, "message": last_error}

# Format functions (unchanged)
def format_dab_entities_for_prompt(entities: List[Dict[str, Any]]) -> str:
    if not entities:
        return "No entities available."
    lines = ["Available HR Entities (tables/views):"]
    for entity in entities:
        name = entity.get("name", "unknown")
        kind = entity.get("kind", "table")
        desc = entity.get("description", "")
        fields = entity.get("fields", entity.get("columns", []))
        lines.append(f"\nEntity: {name} ({kind})")
        if desc:
            lines.append(f"  Description: {desc}")
        for field in fields:
            if isinstance(field, dict):
                fname = field.get("name", "")
                ftype = field.get("type", "")
                nullable = "NULL" if field.get("nullable", True) else "NOT NULL"
                lines.append(f"  - {fname} ({ftype}, {nullable})")
            else:
                lines.append(f"  - {field}")
    return "\n".join(lines)

def format_dab_entities_as_tools(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_records",
                "description": "Read records from an entity with optional filtering, selection, ordering, and pagination.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string"},
                        "select": {"type": "string", "description": "Comma-separated fields to return. Omit for all."},
                        "filter": {"type": "string", "description": "OData filter expression"},
                        "orderby": {"type": "array", "items": {"type": "string"}, "description": "Sort fields with direction, e.g. [\'salary desc\', \'name\']."},
                        "first": {"type": "integer", "description": "Max records to return"}
                    },
                    "required": ["entity"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "aggregate_records",
                "description": "Aggregate records with optional grouping.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string"},
                        "function": {"type": "string", "enum": ["count", "sum", "avg", "min", "max"]},
                        "field": {"type": "string"},
                        "groupby": {"type": "array", "items": {"type": "string"}},
                        "orderby": {"type": "string", "enum": ["asc", "desc"], "description": "Sort direction for grouped results by aggregated value. Requires groupby."},
                        "distinct": {"type": "boolean", "description": "Remove duplicate values before aggregating. Not valid with field *."},
                        "having": {"type": "string", "description": "OData filter on aggregated results. Requires groupby."},
                        "filter": {"type": "string"},
                        "first": {"type": "integer"}
                    },
                    "required": ["entity", "function"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "describe_entities",
                "description": "Discover all available entities and their fields.",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]
    return tools

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL SINGLETON (must be AFTER class definitions)
# ════════════════════════════════════════════════════════════════════════════
dab_manager = DABTenantClientManager()