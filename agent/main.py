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
    SCHEMA_REGISTRY_REFRESH_INTERVAL_SECONDS,
    JWT_SECRET, JWT_ALGORITHM, JWT_AUDIENCE, AUTH_MODE,
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

# Dynamic dashboard routes
from agent.integrations.dynamic_dashboard import dashboard_router

# Reflexive agent executor
from agent.core.agentic_executor import run_reflexive_agent
from agent.core.progress import set_progress_emitter, reset_progress_emitter

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
    """Discover tools and schema from SQL MCP server and HANA MCP server."""
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
                    actual_dim = int(data["embeddings"].shape[1]) if hasattr(data["embeddings"], "shape") else cached_dim
                    if cached_dim == actual_dim:
                        schema_index_ready = True
                        logger.info(
                            "Schema index cache exists at %s (dim=%d, age=%.0fs) Ã¢â‚¬â€ skipping warmup build.",
                            cache_path.name,
                            cached_dim,
                            time.time() - float(data["built_at"]),
                        )
                    else:
                        logger.warning(
                            "Schema cache dim mismatch (stored=%d, actual=%d) Ã¢â‚¬â€ rebuilding.",
                            cached_dim,
                            actual_dim,
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

    # HANA schema discovery Ã¢â‚¬â€ run once at startup (independent of DAB)
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
                "HANA schema registry mismatch for tenant=%s (cached=%d schemas, live=%d schemas) Ã¢â‚¬â€ invalidating cache",
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
# (relocated to agent/auth/tool_guards.py to break the import cycle
#  agent.core.agentic_executor -> agent.main -> agent.core.agentic_executor)
# ==============================
from agent.auth.tool_guards import enforce_tool_args, filter_tool_results  # noqa: F401 (re-exported)


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
                        "SchemaRegistry: startup mismatch detected for tenant=%s (cached=%d schemas, live=%d schemas) Ã¢â‚¬â€ refreshing",
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
                "SchemaRegistry: startup stale cache detected for tenant=%s (age=%ds > TTL=%ds) Ã¢â‚¬â€ will reload on next access",
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

# Mount the custom HR chat UI (built React app in ui/dist).
# Served same-origin so the relative /v1 fetch works without CORS.
# NOTE: /ui/config route is registered BEFORE this mount so it isn't
# shadowed by StaticFiles.
@app.get("/ui/config")
async def ui_config():
    """Return JWT tokens for the custom HR chat UI.

    In dev (AUTH_MODE=test), tokens are generated on-the-fly from JWT_SECRET
    with appropriate roles. In production, HR_EMPLOYEE_TOKEN and HR_ADMIN_TOKEN
    env vars should be set to pre-issued JWTs.
    """
    import jwt as pyjwt
    import datetime

    employee_token = os.getenv("HR_EMPLOYEE_TOKEN", "")
    admin_token = os.getenv("HR_ADMIN_TOKEN", "")

    if not employee_token or not admin_token:
        # Dev mode: generate tokens on-the-fly from JWT_SECRET
        now = datetime.datetime.now(datetime.timezone.utc)
        common = {
            "iss": "local-dev-issuer",
            "aud": JWT_AUDIENCE,
            "iat": int(now.timestamp()),
            "exp": int((now + datetime.timedelta(hours=8)).timestamp()),
        }
        if not employee_token:
            employee_token = pyjwt.encode(
                {**common, "tenant_id": "RDEMOROCKFORT", "groups": ["rockfortEmployee"],
                 "roles": ["HRMS_EMPLOYEE"], "email": "dev.employee@rymnet.com",
                 "sub": "dev-employee", "emp_id": "A0002"},
                JWT_SECRET, algorithm=JWT_ALGORITHM,
            )
        if not admin_token:
            admin_token = pyjwt.encode(
                {**common, "tenant_id": "RDEMOROCKFORT", "groups": ["rockfortHR"],
                 "roles": ["HRMS_HR"], "email": "dev.admin@rymnet.com",
                 "sub": "dev-admin", "emp_id": "A0001"},
                JWT_SECRET, algorithm=JWT_ALGORITHM,
            )

    return {"employee_token": employee_token, "admin_token": admin_token}

_ui_dist = Path(__file__).parent.parent / "ui" / "dist"
if _ui_dist.exists():
    app.mount("/ui", StaticFiles(directory=str(_ui_dist), html=True), name="ui")
    logger.info("Mounted custom UI at /ui from %s", _ui_dist)
else:
    logger.warning("ui/dist not found -- custom UI not mounted. Run 'npm run build' in ui/")

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
async def run_agent(messages: List[Dict], auth_context: AuthContext, is_custom_ui: bool = False) -> str:
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

    logger.info("Using reflexive agent mode (custom_ui=%s)", is_custom_ui)
    return await run_reflexive_agent(
        user_query=user_query,
        conversation_history=conversation_history,
        auth_context=auth_context,
        cached_tools=CACHED_TOOLS,
        cached_schema=CACHED_SCHEMA,
        is_custom_ui=is_custom_ui,
    )


# ==============================
# RESPONSE FORMATTING
# ==============================
def _model_name(request: ChatCompletionRequest) -> str:
    return "hr-agent" if request.model == "ai-agent" else request.model


def _role_chunk(completion_id: str, created: int, model: str) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _progress_chunk(completion_id: str, created: int, model: str, progress: Dict) -> str:
    """OpenAI-shaped chunk carrying a side-channel progress event.

    The delta is empty so LibreChat (and any strict OpenAI consumer) sees a
    no-op chunk; the hr_progress field is read by the custom HR chat UI.
    """
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "hr_progress": progress,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _chart_chunk(completion_id: str, created: int, model: str, payload: Dict) -> str:
    """OpenAI-shaped chunk carrying a structured Chart.js payload.

    Emitted only for clients that send X-HR-Client: custom-ui. The delta is
    empty so LibreChat (which ignores unknown fields) is unaffected.
    """
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "hr_chart": payload,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _content_chunk(completion_id: str, created: int, model: str, text: str) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _stop_chunk(completion_id: str, created: int, model: str) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def _answer_chunks(completion_id: str, created: int, model: str, answer: str):
    """Yield the role chunk, content chunks (10 chars each), and stop chunk.

    Refactored out of format_streaming_response so the streaming endpoint can
    interleave progress chunks before the answer.
    """
    yield _role_chunk(completion_id, created, model)

    chunk_size = 10
    for i in range(0, len(answer), chunk_size):
        yield _content_chunk(completion_id, created, model, answer[i:i + chunk_size])
        await asyncio.sleep(0.01)

    yield _stop_chunk(completion_id, created, model)
    yield "data: [DONE]\n\n"


async def _content_chunks(completion_id: str, created: int, model: str, answer: str):
    """Yield content chunks (10 chars each) and stop chunk, WITHOUT the role chunk.

    Used by stream_agent_response which emits the role chunk first, then
    progress events, then content.
    """
    chunk_size = 10
    for i in range(0, len(answer), chunk_size):
        yield _content_chunk(completion_id, created, model, answer[i:i + chunk_size])
        await asyncio.sleep(0.01)

    yield _stop_chunk(completion_id, created, model)
    yield "data: [DONE]\n\n"


def format_streaming_response(request: ChatCompletionRequest, answer: str):
    """Legacy streaming response -- kept for backward compatibility.

    New streaming path uses stream_agent_response() which interleaves progress
    events. This function is retained for any direct callers/tests.
    """
    async def generate():
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        model = _model_name(request)
        async for chunk in _answer_chunks(completion_id, created, model, answer):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


async def stream_agent_response(request: ChatCompletionRequest, messages: List[Dict], auth_context: AuthContext, is_custom_ui: bool = False):
    """Stream the agent pipeline with interleaved progress and chart events.

    Runs run_agent() as a background task while draining a queue. Progress
    events (via the ContextVar emitter in agent.core.progress) are yielded as
    OpenAI-shaped chunks with an hr_progress side-channel field. Structured
    chart payloads (hr_chart) are yielded as separate chunks when the client
    sends X-HR-Client: custom-ui. When the task completes, the answer is
    chunked and streamed.

    The is_custom_ui flag (derived from the X-HR-Client header by the caller)
    determines whether markdown chart artifacts are suppressed in favor of the
    hr_chart side-channel.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    model = _model_name(request)

    queue: asyncio.Queue = asyncio.Queue()

    def _on_progress(stage: str, label: str, detail: str = None):
        queue.put_nowait({"type": "progress", "stage": stage, "label": label, "detail": detail})

    def _on_chart(payload: Dict):
        queue.put_nowait({"type": "chart", "payload": payload})

    async def _run():
        token = set_progress_emitter(_on_progress, _on_chart if is_custom_ui else None)
        try:
            return await run_agent(messages, auth_context, is_custom_ui=is_custom_ui)
        finally:
            reset_progress_emitter(token)

    task = asyncio.create_task(_run())

    async def generate():
        try:
            # Emit the role chunk first (establishes assistant identity).
            yield _role_chunk(completion_id, created, model)

            # Drain progress/chart events while the agent runs.
            while not task.done():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # No progress for 15s -- emit a keepalive comment so
                    # proxies don't close the connection.
                    yield ": keepalive\n\n"
                    continue
                if event["type"] == "chart":
                    yield _chart_chunk(completion_id, created, model, event["payload"])
                else:
                    yield _progress_chunk(completion_id, created, model, event)

            # Drain any events queued after the task finished.
            while not queue.empty():
                event = queue.get_nowait()
                if event["type"] == "chart":
                    yield _chart_chunk(completion_id, created, model, event["payload"])
                else:
                    yield _progress_chunk(completion_id, created, model, event)

            # Stream the answer content (role chunk already sent above).
            answer = task.result()
            async for chunk in _content_chunks(completion_id, created, model, answer):
                yield chunk
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Agent execution failed during stream")
            # Can't change HTTP status after headers sent; emit a friendly
            # error as content so the client sees something.
            error_msg = "I encountered an error while processing your request. Please try again."
            async for chunk in _answer_chunks(completion_id, created, model, error_msg):
                yield chunk
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
app.include_router(dashboard_router)

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    auth_context: AuthContext = Depends(verify_token),
    http_request: Request = None,
):
    logger.info("Request from user: %s (tenant=%s, roles=%s)",
                auth_context.email, auth_context.tenant_id, auth_context.internal_roles)

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Client hint: custom HR chat UI sends X-HR-Client: custom-ui.
    # This gates the hr_chart side-channel and suppresses markdown chart
    # injection for that client (LibreChat and all other clients are unaffected).
    client_hint = http_request.headers.get("X-HR-Client", "") if http_request else ""
    is_custom_ui = client_hint == "custom-ui"

    if request.stream:
        return await stream_agent_response(request, messages, auth_context, is_custom_ui)
    else:
        try:
            answer = await run_agent(messages, auth_context, is_custom_ui=is_custom_ui)
        except Exception as e:
            logger.exception("Agent execution failed")
            raise HTTPException(status_code=500, detail=str(e))
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
    """Serve generated chart PNGs. No auth required Ã¢â‚¬â€ filenames are random timestamps (unguessable)."""
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
    print(f"Starting HR AI Agent API on http://localhost:{os.getenv('AGENT_PORT', '8001')}")
    print(f"DAB discovery tenant: {discovery_tenant}")
    print("DAB routing: per-tenant via tenant_mappings.yaml")
    print(f"RAG collection: {CHROMA_COLLECTION_NAME} at {CHROMA_DB_PATH}")
    print(f"Excel exports: {EXPORT_DIR}")
    print(f"Chart output: {CHART_OUTPUT_DIR}")
    print(f"Auth mode: {os.getenv('AUTH_MODE', 'test')}")
    print(f"Agent mode: {AGENT_MODE}")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("AGENT_PORT", "8001")))

