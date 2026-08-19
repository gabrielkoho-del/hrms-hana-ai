import os
import sys
from pathlib import Path

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(env_path)

import json
import re
import asyncio
import traceback
import uuid
import time
import glob
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import numpy as np

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import logging
from contextlib import asynccontextmanager

# ==============================
# MODULE IMPORTS
# ==============================
from agent.output.excel_exporter import EXPORT_DIR, export_to_excel, cleanup_old_exports
from agent.integrations.rag_retriever import rag_retriever, retrieve_policy_context

from agent.auth import verify_token, AuthContext
from agent.config import (
    CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, RAG_TOP_K, RAG_SIMILARITY_THRESHOLD,
    LARGE_RESULT_THRESHOLD, AGENT_BASE_URL, DEFAULT_MAX_ROWS, MAX_SQL_LENGTH,
    UNLIMITED_ROWS, RATE_LIMIT_PER_MINUTE, RATE_LIMIT_WINDOW_SECONDS,
    CONTEXT_WINDOW_SIZE, MAX_TOKENS, SCHEMA_REGISTRY_TTL_SECONDS,
    SCHEMA_REGISTRY_REFRESH_INTERVAL_SECONDS
)
from agent.output.chart_generator import generate_chart, extract_chartable_data, CHART_OUTPUT_DIR
from agent.core.tool_planner import build_tool_plan
from agent.core.intent_classifier import get_intent_exemplar_index

from agent.integrations.schema_index import (
    warm_schema_index,
    SchemaFieldIndex,
    SCHEMA_INDEX_CACHE_DIR,
    EMBEDDING_DIM,
)

# DAB client (replaces custom MCP SSE transport)
from agent.integrations.dab_client import (
    DABClient, DABTenantClientManager, format_dab_entities_as_tools,
    format_dab_entities_for_prompt, invoke_dab_tool_with_retry
)

# HANA client integration
from agent.integrations.hana_client import hana_manager

# Schema registry service (replaces ad-hoc file I/O in agentic_executor)
from agent.integrations.schema_registry import schema_registry_service

# Reflexive agent executor
from agent.core.agentic_executor import run_reflexive_agent

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("hr_agent")

# ==============================
# CACHED TOOLS & SCHEMA
# ==============================
CACHED_TOOLS: List[Dict] = []
CACHED_TOOLS_PROMPT: str = ""
CACHED_SCHEMA: Dict = {}
CACHED_HANA_SCHEMAS: Dict = {}

_executor = ThreadPoolExecutor(max_workers=2)

# ==============================
# AGENT MODE
# ==============================
AGENT_MODE = os.getenv("AGENT_MODE", "reflexive").lower()
logger.info("Agent mode: %s", AGENT_MODE)

# ==============================
# DAB CLIENT MANAGER
# ==============================
dab_manager = DABTenantClientManager()

# ==============================
# TOOL DISCOVERY
# ==============================
async def discover_tools():
    """Discover tools and schema from DAB server and HANA MCP server."""
    global CACHED_TOOLS, CACHED_TOOLS_PROMPT, CACHED_SCHEMA
    tool_lines = []
    tenant_id = os.getenv("DAB_DISCOVERY_TENANT", "RDEMOROCKFORT")
    try:
        logger.info("Discovering tools from tenant: %s", tenant_id)
        entities = dab_manager.get_entities(tenant_id, force_refresh=True)

        # Convert to OpenAI-compatible tool definitions
        CACHED_TOOLS = format_dab_entities_as_tools(entities)

        # Build prompt description
        schema_block = format_dab_entities_for_prompt(entities)
        tool_lines = []
        for tool in CACHED_TOOLS:
            name = tool["function"]["name"]
            desc = tool["function"]["description"]
            schema_str = json.dumps(tool["function"]["parameters"], indent=2)
            tool_lines.append(f"- {name}: {desc}")
            tool_lines.append(f"  Schema: {schema_str}")

        logger.info("Discovered %d total tools", len(CACHED_TOOLS))

        # Cache schema
        CACHED_SCHEMA = {}
        if isinstance(entities, list):
            for entity in entities:
                name = entity.get("name")
                if name:
                    CACHED_SCHEMA[name] = entity
                    fields = entity.get("fields", entity.get("columns", []))
                    logger.info("Cached schema for %s: %d fields", name, len(fields))
        elif isinstance(entities, dict):
            CACHED_SCHEMA = entities

        # Eagerly build/warm the schema embedding index at discovery time (startup),
        # off the request path, so the first user query doesn't pay the embed cost
        # or risk a 429. Uses the same tenant that discovery ran for.
        schema_index_ready = False
        if CACHED_SCHEMA:
            tmp_instance = SchemaFieldIndex(CACHED_SCHEMA, tenant_id=tenant_id)
            cache_path = tmp_instance._cache_path()
            if cache_path.exists():
                try:
                    data = np.load(cache_path, allow_pickle=False)
                    cached_dim = int(data["embedding_dim"])
                    if cached_dim == EMBEDDING_DIM:
                        schema_index_ready = True
                        logger.info(
                            "Schema index cache exists at %s (dim=%d, age=%.0fs) â€” skipping warmup build.",
                            cache_path.name,
                            cached_dim,
                            time.time() - float(data["built_at"]),
                        )
                    else:
                        logger.warning(
                            "Schema cache dim mismatch (cached=%d, config=%d) â€” rebuilding.",
                            cached_dim,
                            EMBEDDING_DIM,
                        )
                except Exception as e:
                    logger.warning("Schema cache pre-check failed, will warm: %s", e)

            if not schema_index_ready:
                index = await warm_schema_index(CACHED_SCHEMA, tenant_id=tenant_id)
                if index is not None:
                    schema_index_ready = True
                else:
                    logger.warning(
                        "Startup schema index warmup FAILED. "
                        "First semantic schema search will trigger live embedding API calls and may hit rate limits."
                    )

        if not schema_index_ready and CACHED_SCHEMA:
            logger.error(
                "SCHEMA INDEX NOT READY AT STARTUP. "
                "Either warmup failed (check error above) or schema was empty. "
                "First request requiring semantic field retrieval will pay embedding cost."
            )

    except Exception as e:
        logger.error("DAB tool discovery failed: %s", e)

    # HANA schema discovery â€” run once at startup (independent of DAB)
    try:
        from agent.integrations.hana_client import hana_manager
        hana_schemas = await _discover_hana_schemas(tenant_id)
        CACHED_HANA_SCHEMAS.clear()
        CACHED_HANA_SCHEMAS.update(hana_schemas)
        logger.info("Discovered %d HANA schemas at startup", len(hana_schemas))
    except Exception as e:
        logger.warning("HANA schema discovery failed: %s", e)

    # HANA tool discovery (dynamic via MCP tools/list, with static fallback)
    try:
        from agent.integrations.hana_client import refresh_hana_tool_schemas
        hana_tools = await refresh_hana_tool_schemas(tenant_id=tenant_id)

        if not isinstance(hana_tools, list):
            raise TypeError(
                f"refresh_hana_tool_schemas() must return a list, got {type(hana_tools).__name__}"
            )

        invalid_tools = []
        for tool in hana_tools:
            if not isinstance(tool, dict):
                invalid_tools.append({"tool": tool, "reason": "not a dict"})
                continue
            function = tool.get("function")
            if not isinstance(function, dict):
                invalid_tools.append({"tool": tool, "reason": "missing function"})
                continue
            if not function.get("name"):
                invalid_tools.append({"tool": tool, "reason": "missing function.name"})
                continue

        if invalid_tools:
            raise ValueError(
                f"refresh_hana_tool_schemas() returned invalid tools: {invalid_tools}"
            )

        CACHED_TOOLS.extend(hana_tools)
        for tool in hana_tools:
            name = tool["function"]["name"]
            desc = tool["function"]["description"]
            schema_str = json.dumps(tool["function"]["parameters"], indent=2)
            tool_lines.append(f"- {name}: {desc}")
            tool_lines.append(f"  Schema: {schema_str}")
        logger.info("Appended %d HANA tool schemas", len(hana_tools))
    except Exception as e:
        logger.error("HANA tool schema setup failed: %s", e)
        raise

    CACHED_TOOLS_PROMPT = "\n".join(tool_lines)
    logger.info("Discovered %d total tools", len(CACHED_TOOLS))


async def _discover_hana_schemas(tenant_id: str) -> Dict[str, List[str]]:
    """Discover HANA schemas and their tables once at startup.

    Uses the SchemaRegistryService for cache loading and falls back
    to live HANA discovery if no cache exists or if the cached registry
    is stale or mismatched.
    """
    from agent.integrations.hana_client import hana_manager

    # Use the registry service's TTL-aware get_registry()
    cached = schema_registry_service.get_registry(tenant_id)
    if cached:
        logger.info("HANA schema loaded from registry service for tenant=%s (%d schemas)", tenant_id, len(cached))
        return cached

    # Live discovery via HANA MCP
    client = hana_manager.get_client(tenant_id)
    schemas: Dict[str, List[str]] = {}

    try:
        list_schemas_result = client.call_tool("hana_list_schemas", {})
        normalized_schemas = _normalize_name_list(list_schemas_result)

        for schema_name in normalized_schemas:
            try:
                list_tables_result = client.call_tool("hana_list_tables", {"schema_name": schema_name})
                tables = _normalize_name_list(list_tables_result)
                if tables:
                    schemas[schema_name] = tables
                    logger.info("HANA schema: %s (%d tables)", schema_name, len(tables))
            except Exception as e:
                logger.warning("HANA list_tables failed for schema=%s: %s", schema_name, e)

        # P1: Compare with any existing cached registry and invalidate on mismatch
        existing = schema_registry_service._registries.get(tenant_id.upper())
        if existing and existing != schemas:
            logger.info(
                "HANA schema registry mismatch for tenant=%s (cached=%d schemas, live=%d schemas) â€” invalidating cache",
                tenant_id, len(existing), len(schemas),
            )
            schema_registry_service.invalidate_tenant(tenant_id)

        # Persist to registry service and disk
        schema_registry_service._registries[tenant_id.upper()] = schemas
        schema_registry_service._last_loaded_at[tenant_id.upper()] = schema_registry_service._now()
        cache_file = schema_registry_service._tenant_cache_path(tenant_id)
        schema_registry_service._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"schemas": schemas, "updated_at": time.time()}, f)
        logger.info("HANA schema cache saved to disk (%d schemas)", len(schemas))

        # Enrich with business semantics via hana_explain_table
        try:
            schema_registry_service.enrich_schema_with_semantics(
                tenant_id, schemas, hana_manager.get_client(tenant_id), max_tables=50
            )
        except Exception as e:
            logger.warning("HANA schema semantics enrichment failed: %s", e)
    except Exception as e:
        logger.error("HANA schema discovery failed: %s", e)

    return schemas


def _normalize_name_list(result: Any) -> List[str]:
    """Parse HANA list response from the current Streamable HTTP server.

    Modern `hana_list_schemas` / `hana_list_tables` return
    `structuredContent.items` populated by formatNameListToolResult().
    """
    if not isinstance(result, dict):
        return []
    structured = result.get("structuredContent")
    if not isinstance(structured, dict):
        return []
    names = structured.get("items") or []
    return [str(n).upper() for n in names if n]




# ==============================
# PERMISSION ENFORCEMENT
# ==============================
def enforce_tool_args(tool: str, args: dict, auth_context: AuthContext) -> tuple[dict, Optional[str]]:
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
        from agent.tool_planner import validate_dab_filter_permissions
        filt = args.get("filter", "")
        is_valid, error = validate_dab_filter_permissions(filt, args.get("entity", ""), auth_context)
        if not is_valid:
            return args, error

    return args, None


def _has_self_filter(filter_str: str, auth_context: AuthContext) -> bool:
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


def filter_tool_results(tool: str, result_text: str, auth_context: AuthContext) -> str:
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


async def _validate_and_refresh_startup_registries(hana_manager: Any) -> None:
    """Validate loaded registries at startup and refresh stale/mismatched ones.

    For each loaded tenant:
      - If no cache file exists, skip (already loaded from HANA at startup).
      - If cache is older than TTL, re-discover from HANA and overwrite cache.
      - If cache file differs from live HANA discovery, invalidate and reload.
    """
    tenant_ids = list(schema_registry_service.get_all_tenants())
    if not tenant_ids:
        return

    for tenant_id in tenant_ids:
        tenant_key = tenant_id.upper()
        cache_file = schema_registry_service._tenant_cache_path(tenant_id)
        cached_schemas = schema_registry_service._registries.get(tenant_key, {})
        is_stale = schema_registry_service._is_stale(tenant_key)
        cache_exists = cache_file.exists()

        if not cache_exists:
            logger.info("SchemaRegistry: startup validation skipped for tenant=%s (no cache file)", tenant_id)
            continue

        if not is_stale and cached_schemas:
            try:
                client = hana_manager.get_client(tenant_id)
                raw_schemas = client.call_tool("hana_list_schemas", {})
                try:
                    from agent.main import _normalize_name_list
                    normalized = _normalize_name_list(raw_schemas)
                except Exception:
                    normalized = []

                live_schemas: Dict[str, List[str]] = {}
                for schema_name in normalized or []:
                    try:
                        raw_tables = client.call_tool("hana_list_tables", {"schema_name": schema_name})
                        try:
                            from agent.main import _normalize_name_list
                            tables = _normalize_name_list(raw_tables)
                        except Exception:
                            tables = []
                        if tables:
                            live_schemas[schema_name] = tables
                    except Exception as e:
                        logger.debug("SchemaRegistry: startup validation list_tables failed for %s: %s", schema_name, e)

                if live_schemas != cached_schemas:
                    logger.info(
                        "SchemaRegistry: startup mismatch detected for tenant=%s (cached=%d schemas, live=%d schemas) â€” refreshing",
                        tenant_id, len(cached_schemas), len(live_schemas),
                    )
                    schema_registry_service.invalidate_tenant(tenant_id)
                    schema_registry_service._registries[tenant_key] = live_schemas
                    schema_registry_service._last_loaded_at[tenant_key] = schema_registry_service._now()
                    schema_registry_service._cache_dir.mkdir(parents=True, exist_ok=True)
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump({"schemas": live_schemas, "updated_at": schema_registry_service._now()}, f)
                else:
                    logger.info("SchemaRegistry: startup validation passed for tenant=%s (cache matches live)", tenant_id)
            except Exception as e:
                logger.warning("SchemaRegistry: startup validation failed for tenant=%s: %s", tenant_id, e)
        elif is_stale:
            logger.info(
                "SchemaRegistry: startup stale cache detected for tenant=%s (age=%ds > TTL=%ds) â€” will reload on next access",
                tenant_id,
                int(schema_registry_service._now() - schema_registry_service._last_loaded_at.get(tenant_key, 0)),
                schema_registry_service.ttl_seconds,
            )


# ==============================
# FASTAPI APP
# ==============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    await discover_tools()
    if os.getenv("GEMINI_API_KEY"):
        rag_retriever.initialize()
    cleanup_old_exports()
    os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)

    # Load all HANA schema registries at startup (per-tenant, in-memory)
    registry_summary = schema_registry_service.load_all_tenants()
    logger.info(
        "SchemaRegistry: startup loaded %d tenants: %s",
        len(registry_summary),
        ", ".join(f"{t}={c}" for t, c in registry_summary.items()),
    )
    # Keep CACHED_HANA_SCHEMAS populated for backward compatibility
    CACHED_HANA_SCHEMAS.clear()
    for tenant_id, schemas in schema_registry_service._registries.items():
        CACHED_HANA_SCHEMAS.update(schemas)

    # Validate and refresh stale/mismatched registries at startup
    await _validate_and_refresh_startup_registries(hana_manager)

    # Refresh CACHED_HANA_SCHEMAS after validation in case any registries were updated
    CACHED_HANA_SCHEMAS.clear()
    for tenant_id, schemas in schema_registry_service._registries.items():
        CACHED_HANA_SCHEMAS.update(schemas)

    # Warm up the same HANA tenant used for DAB discovery (no separate LOCALDEV)
    discovery_tenant = os.getenv("DAB_DISCOVERY_TENANT", "RDEMOROCKFORT")
    try:
        hana_manager.get_client(discovery_tenant)
        logger.info("HANA MCP client warmed up for discovery tenant: %s", discovery_tenant)
    except Exception as e:
        logger.info("HANA MCP warmup skipped or unavailable for tenant %s: %s", discovery_tenant, e)

    # Pre-warm intent exemplar index off the request path.
    # This only embeds on the very first startup when no disk cache exists.
    if os.getenv("GEMINI_API_KEY"):
        try:
            await get_intent_exemplar_index()
            logger.info("IntentExemplarIndex: pre-warmed at startup")
        except Exception as e:
            logger.warning("IntentExemplarIndex: startup pre-warm failed: %s", e)

    # Start background schema registry refresh (P2)
    try:
        await schema_registry_service.start_background_refresh(
            tenant_ids=list(schema_registry_service.get_all_tenants()),
            hana_manager=hana_manager,
            refresh_interval_seconds=SCHEMA_REGISTRY_REFRESH_INTERVAL_SECONDS,
        )
    except Exception as e:
        logger.warning("SchemaRegistry: background refresh startup failed: %s", e)

    yield
    # Cancel background refresh on shutdown
    if schema_registry_service._refresh_task is not None:
        schema_registry_service._refresh_task.cancel()
        try:
            await schema_registry_service._refresh_task
        except asyncio.CancelledError:
            pass
    dab_manager.close_all()
    hana_manager.close_all()

app = FastAPI(title="HR AI Agent API", lifespan=lifespan)

app.mount("/exports", StaticFiles(directory=EXPORT_DIR), name="exports")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# RATE LIMITING
# ==============================
request_counts: Dict[str, List[float]] = {}

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    client_ip = request.client.host
    now = time.time()
    request_counts[client_ip] = [t for t in request_counts.get(client_ip, []) if now - t < RATE_LIMIT_WINDOW_SECONDS]
    if len(request_counts[client_ip]) > RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: max {RATE_LIMIT_PER_MINUTE} requests/minute")
    request_counts[client_ip].append(now)
    return await call_next(request)


# ==============================
# OPENAI-COMPATIBLE MODELS
# ==============================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Literal["hr-agent", "hr-agent-fast", "hr-agent-creative", "ai-agent"] = "hr-agent"
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 0.7
    model_config = {"extra": "allow"}

class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


# ==============================
# CORE AGENT LOGIC (REFLEXIVE ONLY)
# ==============================
async def run_agent(messages: List[Dict], auth_context: AuthContext) -> str:
    user_query = messages[-1]["content"] if messages else ""

    conversation_history = ""
    if len(messages) > 2:
        # Include BOTH user and assistant messages for full context
        # Industry standard: last N turns (user + assistant) for context fusion
        recent_msgs = messages[-6:-1]  # Last 5 turns before current
        turns = []
        for m in recent_msgs:
            role_label = "User" if m["role"] == "user" else "Assistant"
            turns.append(f"{role_label}: {m['content'][:200]}")
        conversation_history = "\n".join(turns)

    logger.info("Using reflexive agent mode")
    return await run_reflexive_agent(
        user_query=user_query,
        conversation_history=conversation_history,
        auth_context=auth_context,
        cached_tools=CACHED_TOOLS,
        cached_schema=CACHED_SCHEMA,
    )


# ==============================
# RESPONSE FORMATTING
# ==============================
def format_streaming_response(request: ChatCompletionRequest, answer: str):
    async def generate():
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        model = "hr-agent" if request.model == "ai-agent" else request.model

        first_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"

        chunk_size = 10
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i:i+chunk_size]
            data_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(data_chunk)}\n\n"
            await asyncio.sleep(0.01)

        stop_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(stop_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def format_json_response(request: ChatCompletionRequest, answer: str):
    model = "hr-agent" if request.model == "ai-agent" else request.model
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=model,
        choices=[Choice(
            message=ChatMessage(role="assistant", content=answer),
            finish_reason="stop"
        )]
    )


# ==============================
# FASTAPI ENDPOINTS
# ==============================
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    auth_context: AuthContext = Depends(verify_token)
):
    logger.info("Request from user: %s (tenant=%s, roles=%s)",
                auth_context.email, auth_context.tenant_id, auth_context.internal_roles)

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        answer = await run_agent(messages, auth_context)
    except Exception as e:
        logger.exception("Agent execution failed")
        raise HTTPException(status_code=500, detail=str(e))

    if request.stream:
        return format_streaming_response(request, answer)
    else:
        return format_json_response(request, answer)


@app.get("/v1/models")
async def list_models(auth_context: AuthContext = Depends(verify_token)):
    return {
        "object": "list",
        "data": [
            {"id": "hr-agent", "object": "model", "created": 1700000000, "owned_by": "custom"},
            {"id": "hr-agent-fast", "object": "model", "created": 1700000000, "owned_by": "custom"},
            {"id": "hr-agent-creative", "object": "model", "created": 1700000000, "owned_by": "custom"}
        ]
    }


@app.get("/charts/{filename}")
async def serve_chart(filename: str):
    """Serve generated chart PNGs. No auth required â€” filenames are random timestamps (unguessable)."""
    filepath = os.path.join(CHART_OUTPUT_DIR, filename)
    logger.info("CHART_SERVE: filename=%s exists=%s", filename, os.path.exists(filepath))
    if os.path.exists(filepath):
        return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="Chart not found")


@app.get("/health/charts")
async def chart_health(auth_context: AuthContext = Depends(verify_token)):
    """List available chart files."""
    chart_files = glob.glob(os.path.join(CHART_OUTPUT_DIR, "chart_*.png"))
    return {
        "chart_dir": CHART_OUTPUT_DIR,
        "chart_count": len(chart_files),
        "charts": [os.path.basename(f) for f in sorted(chart_files)[-10:]]
    }


@app.get("/health")
async def health(auth_context: AuthContext = Depends(verify_token)):
    rag_status = "available" if rag_retriever.is_available() else "unavailable"
    export_count = len(glob.glob(os.path.join(EXPORT_DIR, "*.xlsx")))
    chart_count = len(glob.glob(os.path.join(CHART_OUTPUT_DIR, "chart_*.png")))
    registry_summary = schema_registry_service.get_summary()
    return {
        "status": "ok",
        "rag": rag_status,
        "dab_tools": len(CACHED_TOOLS),
        "entities": list(CACHED_SCHEMA.keys()),
        "exports_ready": export_count,
        "export_dir": EXPORT_DIR,
        "charts_ready": chart_count,
        "chart_dir": CHART_OUTPUT_DIR,
        "authenticated_user": auth_context.email,
        "emp_id": auth_context.emp_id,
        "tenant": auth_context.tenant_id,
        "roles": auth_context.internal_roles,
        "permissions": sorted(auth_context.permissions),
        "agent_mode": AGENT_MODE,
        "schema_registry": registry_summary,
    }


if __name__ == "__main__":
    import uvicorn
    discovery_tenant = os.getenv("DAB_DISCOVERY_TENANT", "RDEMOROCKFORT")
    print("Starting HR AI Agent API on http://localhost:8000")
    print(f"DAB discovery tenant: {discovery_tenant}")
    print("DAB routing: per-tenant via tenant_mappings.yaml")
    print(f"RAG collection: {CHROMA_COLLECTION_NAME} at {CHROMA_DB_PATH}")
    print(f"Excel exports: {EXPORT_DIR}")
    print(f"Chart output: {CHART_OUTPUT_DIR}")
    print(f"Auth mode: {os.getenv('AUTH_MODE', 'test')}")
    print(f"Agent mode: {AGENT_MODE}")
    uvicorn.run(app, host="0.0.0.0", port=8000)

