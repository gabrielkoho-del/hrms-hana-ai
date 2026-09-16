# HR AI Agent + DAB Data Layer -- Architecture Document

## 1. System Overview

This is a **multi-tenant, permission-aware, reflexive HR AI Agent** built on FastAPI with a **DAB (Data API Builder)** + **SAP HANA** data layer. The chat UI is LibreChat.

### Core Capabilities

| Capability | Implementation |
|------------|---------------|
| **Structured HR data** | DAB / SQL Server exposed via MCP JSON-RPC over SSE |
| **Structured Finance data** | SAP HANA via MCP HTTP (Port 3100) |
| **Unstructured HR policy knowledge** | RAG (ChromaDB + Gemini embeddings) |
| **Role-based access control** | YAML-based multi-tenant RBAC with 3 role tiers |
| **OpenAI-compatible API** | `/v1/chat/completions` with streaming + JSON modes |
| **Chat UI** | LibreChat |
| **Semantic schema retrieval** | In-memory embedding index for field discovery |
| **Production observability** | OpenTelemetry metrics with graceful null fallback |
| **Zero-touch code resolution** | Per-tenant codesetup reverse index for LLM context |
| **Session state management** | Follow-up action caching (DST-lite) |
| **Config-driven intent classification** | YAML-configured intents, categories, keywords, exemplars with two-stage LLM routing + embedding-based retrieval |
| **Dynamic finance table discovery** | SAP HANA schema scanning at startup with background refresh |
| **Forecasting engine** | Sandboxed subprocess execution with external market data enrichment (OpenDOSM, World Bank, Yahoo Finance) and LLM-generated nixtla models |

### Design Philosophy

> **1st priority: Production practice** -- layered defense, declarative config, telemetry, strict validation, deterministic fallbacks.
> 
> **2nd priority: DRY consolidation** -- shared utilities (`agent/dab/`), config-driven registries, reducing duplicated embedding/client initialization logic.
> 
> **3rd priority: Agentic / non-hardcoding** -- LLM-driven reasoning, dynamic planning, self-healing.

---

## 2. High-Level Architecture

```
  LibreChat          FastAPI (Port 8001)
  (Chat UI)          /v1/chat/completions
                     /health
                     /charts/*
                     /exports/*
                     /sandbox/*
                           |
                           v
                 +---------+---------+
                 |   Reflexive Agent  |
                 |   (4-Stage Loop)   |
                 +---------+---------+
                           |
              +------------+------------+
              |            |            |
+----+-----+ +---+-----+ +---+------+
         |  DAB /   | |  SAP    | | Dynamic |
         | SQL MCP  | |  HANA   | | Dashboard|
         | (5000)   | |  MCP    | | (/dash.) |
         | SQL/OData| | (3100)  | | HTML     |
         +----------+ +---------+ +-----------+
               |            |            |
         SQL Server      HANA DB      DAB bridge
         (HR data)       (Finance)     (Charts)

Forecasting path:
  forecasting_query -> ExternalDataFetcher -> CodeGenerator -> SubprocessSandbox -> Summarizer
    - Internal HR data via DAB (headcount, hires, terminations)
    - External market data via HTTP (OpenDOSM, World Bank, Yahoo Finance)
    - Merge into single JSON -> locked-down subprocess (512MB, 30s, no network)
    - LLM-generated nixtla script reads pre-loaded input.json, writes output.json

Dashboard building path:
  /dashboard.html?query=... or /dashboard.html?config=...
    - Frontend parses query/config and calls /api/ai-query or /api/query
    - Backend executes DAB aggregate_records / read_records directly
    - Returns Chart.js-ready JSON; no Superset MCP or virtual dataset required


---

## 3. Component Inventory

### 3.1 LLM Client (`agent/llm_client.py`)

The **LLM client** uses Google Gemini 3.5 Flash via OpenAI-compatible endpoint with tier-aware model selection, token budget tracking, and automatic fallback chain (model-not-found -> `gemini-3.5-flash`, 429 -> retry, JSON mode error -> retry without format).

**Budget (Gemini Free Tier):** TPM 250K, RPM 15 (tracked), RPD 500 (warn at 400, hard stop at 500).

---

### 3.2 DAB Client (`agent/dab_client.py`) / HANA Client (`agent/hana_client.py`)

The system has two structured-data sources, both exposed as MCP-style tools to the agent.

#### DAB Client

MCP-DAB Bridge client connecting via JSON-RPC over SSE. Key features: 3-path response parsing, JSON-RPC envelope unwrapping, session-aware (`Mcp-Session-Id`), exponential backoff retry. `DABTenantClientManager` provides per-tenant client routing via `tenant_mappings.yaml` with lazy init, cached entity discovery (5-min TTL), and graceful shutdown.

Tools: `read_records` (OData filtering), `aggregate_records` (grouped aggregations), `describe_entities` (schema discovery).

#### HANA Client

SAP HANA MCP client connecting via HTTP transport (JSON-RPC over HTTP). Key features: HTTP-only transport, result normalization from `{"columns", "rows"}` to agent-standard `{"result": [dicts]}`, per-tenant client registry with lazy init. Finance tables discovered dynamically from HANA schema at startup via `hana_list_tables`, cached in-memory with configurable TTL (default 1 hour), with YAML fallback if HANA is unavailable. Wide HANA results (>12 columns) are trimmed to a 12-column Markdown preview before reaching the LLM (per `AGENTS.md`); DAB results are excluded from this trim.

Tools: `hana_execute_query`, `hana_describe_table`, `hana_list_tables`, `hana_get_sample_data`.

#### Dynamic Dashboard (`agent/integrations/dynamic_dashboard.py`)

FastAPI router serving the single-query dashboard HTML and JSON API endpoints.
Endpoints: `/dashboard.html`, `/api/health`, `/api/datasets`, `/api/query`,
`/api/ai-query`, `/api/kpis`, `/api/datasets/{name}/data`.
Data is fetched directly from the DAB bridge; no Superset MCP or virtual dataset is required.

---

### 3.3 Schema Registry Service (`agent/integrations/schema_registry.py`)

Per-tenant HANA schema registry that replaces ad-hoc file I/O in agentic_executor with a single startup-loaded, in-memory registry using local JSON cache files instead of Redis.

Key features: in-memory per-tenant registry, TTL-based reload, background refresh, JSON cache fallback, optional schema enrichment via `hana_explain_table`.

---

### 3.4 Reflexive Agent Pipeline (`agent/agentic_executor.py`)

Core execution engine with four stages:

```
L1 Guard (regex) -> Stage 0: Intent + Tone -> Stage 1: Plan -> Stage 2: Execute -> Stage 3: Summarize
```

**L1 Guard**: Deterministic regex matching for greetings/smalltalk. Returns immediately without LLM calls.

**Stage 0: Intent & Tone Classification** (`agent/core/intent_classifier.py`): Two-stage LLM approach. Stage 1 (coarse router) classifies into a broad category (10 options). Stage 2 (fine classifier) resolves the specific intent and tone fields within that category. Embedding-based exemplar retrieval augments the fine classifier prompt, with Jaccard token-overlap fallback when embeddings are unavailable. Returns structured `IntentResult` with intent, category, data scope, chart eligibility, urgency, emotional state, topic sensitivity, empathy needs, confidence, action orientation, export intent, ambiguity flag, and finance query flag. Tone context propagates through planner, executor, and summarizer. Config-driven by `config/intents.yaml` with eval gating in `config/intents_eval.yaml`.

**Stage 1: Tool Planning** (`agent/tool_planner.py`): Dual-path architecture -- primary path uses native OpenAI-compatible tool calling; fallback uses JSON text parsing with regex. Key features: permission-aware tool filtering, token-budget-aware schema injection, registry-driven HANA tool discovery, DAB filter validation, tone-aware guidance, dynamic binning inference, schema token pruning, semantic schema retrieval for large schemas, explicit chart type detection, export intent inference, TTL cache, hybrid metric extraction for financial queries.

**Stage 2: Execution**: Tool registry routes DAB and HANA calls through a unified interface. Steps: normalize args -> enforce permissions -> validate schema -> invoke tool -> extract payload -> filter results -> dynamic binning -> code resolution -> chart generation -> Excel export. Agent state maintained across steps in unified `state["tool_results"]` dict.

**Stage 3: Summarization** (`agent/summarizer/`): Modular 3-stage response structure -- Inform (factual answer first), Assist (1-3 next steps), Offer feedback (open invitation). Supports ambiguous response override, export-aware guidance, code context injection, and chart artifact injection.

**Special paths**: Forecasting bypasses normal loop with end-to-end pipeline (ExternalDataFetcher -> CodeGenerator -> SubprocessSandbox -> OutputParser -> Summarizer). Dashboard requests are served by the dynamic dashboard router (`/dashboard.html`) with direct DAB queries.

---

### 3.5 Intent Classification (`agent/core/intent_classifier.py`, `agent/core/intent_config.py`, `agent/core/eval_intent_classifier.py`)

Two-stage LLM-based intent and tone classification with config-driven semantics and embedding-based exemplar retrieval.

#### Architecture

```
User Query
    |
    v
+-----------------+     +----------------------+
| Stage 1: Coarse |----*| Stage 2: Fine        |
| Router (LLM)    |     | Classifier (LLM)     |
| 10 categories   |     | ~60 intents + tone   |
+-----------------+     +----------------------+
                                 |
                    +------------+------------+
                    |                         |
             +------v------+           +------v------+
             | Embedding   |           | Jaccard     |
             | Exemplar    |           | Token       |
             | Retrieval   |           | Fallback    |
             +------+------+           +------+------+
                    |                         |
                    +------------+------------+
                                 |
                    +------------v------------+
                    | IntentResult            |
                    | (intent, category,      |
                    |  data_scope, tone, ...) |
                    +-------------------------+
```

#### Design

- **Config-driven**: All intents, categories, keywords, and exemplars live in `config/intents.yaml`. No hardcoded enums in source code.
- **Two-stage routing**: Stage 1 narrows to a broad category (10 options). Stage 2 resolves the specific intent (~60 options) plus tone fields. This reduces token usage and improves accuracy compared to a single monolithic classifier.
- **Embedding exemplar retrieval**: `IntentExemplarIndex` embeds config exemplar texts using the same Gemini model as schema field index. At query time, the fine classifier prompt is augmented with the top-k most similar exemplars. Falls back to Jaccard token overlap if the embedding API is unavailable.
- **Eval gating**: `config/intents_eval.yaml` contains a labeled eval dataset and CI thresholds (per-intent precision/recall >= 0.85, overall F1 >= 0.90). `agent/core/eval_intent_classifier.py` provides the eval framework.
- **Single source of truth for keywords**: Finance keywords, aggregate indicators, export keywords, and affirmative keywords are loaded from `config/intents.yaml`. `config/finance_config.yaml` is retained only for HANA table discovery prefixes and fallback table lists.

#### Key Types

```python
@dataclass
class IntentResult:
    intent: str                      # e.g., "leave_request", "finance_budget"
    intent_category: str             # "personal_data" | "aggregate_data" | ...
    data_scope: str                  # "individual" | "aggregate" | "none"
    chart_eligible: bool             # True ONLY if aggregate + not personal/action/emergency
    urgency_level: str               # "routine" | "time_sensitive" | "urgent" | "distressed"
    emotional_state: str             # "neutral" | "anxious" | "frustrated" | ...
    topic_sensitivity: str           # "low" | "medium" | "high"
    needs_empathy: bool
    confidence: float                # 0.0-1.0
    action_oriented: bool
    wants_export: bool
    is_ambiguous: bool
    finance_query: bool
    forecasting_query: bool
```

---

### 3.6 RAG System (`agent/rag_retriever.py`, `ingestion/`)

#### Architecture

```
PyMuPDF (fitz) -> LlamaIndex Document -> SentenceSplitter (1200 chars, 150 overlap)
                    v
            Gemini Embedding-2 (google-genai SDK)
                    v
            ChromaDB Persistent Vector Store
                    v
            Query: embed -> similarity search -> top-5 chunks
```

Rate limiting is conservative and thread-safe (lock-based window tracking, free-tier headroom).

#### Ingestion Pipeline (`ingestion/ingestion.py` + `index.py`)

- Contextual chunking: prepends `Document: {title} | Section: {heading} | Page: {page}` to each chunk before embedding
- Section detection via regex: numbered headings (`1.0.`, `6.1.1.`) or ALL CAPS lines
- Footer stripping: removes "Sample Document - Malaysia HR Forum" and page numbers
- Batch insert: 10 chunks at a time
- Re-ingestion: deletes old collection, creates fresh one

---

### 3.7 Semantic Schema Index (`agent/schema_index.py`)

**Lightweight in-memory embedding index** for semantic field retrieval. No external vector DB needed -- pure Python + NumPy for <500 fields.

#### Design

- **Batch-first embedding** with per-item fallback on batch failure
- **Hybrid search** (cosine similarity + keyword overlap) for field discovery
- **Disk-persisted cache** keyed by schema hash, with lazy init and startup warm-up

#### Usage

Usage: `search_relevant_fields(user_query, cached_schema, tenant_id, top_k=8)` returns top-k semantically relevant fields with hybrid scores.

---

### 3.8 Code Resolver (`agent/code_resolver.py`)

Zero-touch code resolution for multi-tenant DAB production. Fetches `codesetup` from each tenant's DAB and builds a reverse index for LLM context injection.

Design: per-tenant singleton with isolated cache, cursor-based pagination, TTL caching, and heuristic filtering to skip non-semantic fields.

---

### 3.9 Shared DAB Utilities (`agent/dab/`)

DRY-consolidated cross-cutting concerns for DAB/MCP operations.

#### `dab/dab_response.py` -- DAB/MCP Response Extraction

Centralized response unwrapping.

DRY-consolidated cross-cutting concerns for DAB/MCP operations in `agent/dab/`:
- `dab_response.py`: centralized response extraction (`extract_items`, `extract_payload`, `is_dab_error`)
- `metrics.py`: OpenTelemetry metrics with `_NullMeter` fallback (`dab_tool_calls`, `dab_errors`, `charts_generated`, `entity_name_normalized`)
- `validation.py`: JSON Schema validation via `jsonschema.Draft7Validator`
- `odata_normalizer.py`: schema-driven OData argument normalization

---

### 3.10 Auth System (`agent/auth/`)

Dual-algorithm JWT validation (`auth/models.py`): HS256 for local dev, RS256 for production with issuer validation. Claims extracted: `tenant_id`/`tid`/`org_id`, `groups`/`roles`, `email`/`upn`, `emp_id`, `oid`/`sub`.

Tenant resolution (`auth/tenant_resolver.py`): YAML-based multi-tenant RBAC (`config/tenant_mappings.yaml`) with external groups -> internal roles -> permissions chain.

Two-layer permission enforcement: pre-execution blocks (`aggregate_records` without self-filter, `read_records` without email/emp_id) and post-execution row filtering by email/emp_id.

Auth modes (`auth/dependencies.py`): `test` (falls back to `DEV_TEST_TOKEN`), `production` (requires Bearer header), `disabled` (anonymous, no permissions).

---

### 3.11 Visualization & Export

**Excel Exporter** (`agent/excel_exporter.py`): pandas + openpyxl, chart-rich exports with 3–5 worksheets (Metadata optional), auto-generated for large results, served at `/exports/{filename}`.

**Chart Generator** (`agent/chart_generator.py`): matplotlib (Agg) + pandas + Mermaid. Environment-driven modes, privacy enforcement, color-blind palette, chart artifact injection into LLM context.

**Forecasting Engine** (`agent/forecasting/`): Secure sandboxed time-series forecasting with external market enrichment. LLM-generated nixtla code runs in a locked-down subprocess (AST pre-flight + subprocess isolation + restricted `__builtins__`, 30s timeout, 512MB memory). RestrictedPython is deliberately NOT used — it rewrites bytecode and breaks statsforecast/numba dunder methods; the subprocess boundary is the real security boundary.

---

## 4. Data Flow (Request Lifecycle)

```
User -> FastAPI Middleware (CORS, rate limit) -> verify_token() -> run_agent()
     -> L1 Guard / Intent -> Plan -> Execute (DAB/HANA) -> Summarize -> Response
```

Special paths:
- **Forecasting**: Fetches external data -> merges with HR data -> sandboxed subprocess -> output validation -> summarizer with market context
  - **Dashboard**: Served by the dynamic dashboard router (`/dashboard.html`) with direct DAB queries; no Superset MCP or preview-first workflow required.

LLM calls per request:
- Greeting: 0 (regex guard)
- Standard query: 4 (coarse intent + fine intent + plan + summarize)
- Financial multi-metric: 4-5 (cache-dependent)
- Forecasting: 2-3
- Dashboard preview: 2-3; confirm: 1

---

## 5. Key Design Patterns

### 5.1 Defense in Depth (Security)

| Layer | Mechanism | File |
|-------|-----------|------|
| Transport | JWT Bearer tokens | `auth/dependencies.py` |
| Identity | Tenant isolation + group resolution | `auth/tenant_resolver.py` |
| Pre-execution | OData argument normalization | `dab/odata_normalizer.py` |
| Pre-execution | JSON Schema validation | `dab/validation.py` |
| Pre-execution | Tool-level permission blocks | `main.py:enforce_tool_args()` |
| DAB filter validation | OData self-filter validation | `tool_planner.py:validate_dab_filter_permissions()` |
| Post-execution | Result row filtering by email/emp_id | `main.py:filter_tool_results()` |
| HANA transport | HTTP transport | `hana_client.py:_HttpMpcClient` |
| HANA normalization | Adapter encapsulation | `hana_client.py:normalize_hana_result()` |
| DAB bridge | Read-only entity definitions | `dab-config.json` |

### 5.2 DRY Consolidation (`agent/dab/`)

Previously duplicated logic now centralized:

| Duplication | Consolidated To | Consumers |
|-------------|-----------------|-----------|
| DAB response extraction (3+ variants) | `dab/dab_response.py` | `agentic_executor.py`, `summarizer/response_summarizer.py`, `chart_generator.py` |
| OTel metrics (scattered ad-hoc) | `dab/metrics.py` | `agentic_executor.py`, `dab_client.py`, `chart_generator.py` |
| JSON Schema validation (inline) | `dab/validation.py` | `agentic_executor.py`, `main.py` |
| OData normalization (inline) | `dab/odata_normalizer.py` | `agentic_executor.py` |

### 5.3 Session State Management (DST-lite)

Lightweight in-memory session state for follow-up actions:

```python
_SESSION_STATE: Dict[str, Dict] = {}

# Saved per turn:
{
    "last_query": user_query,
    "last_intent": intent_result.intent,
    "last_chart_data": chartable_data,
    "last_chart_config": chart_config,
    "last_export_url": export_url,
}
```

Enables:
- **Export follow-up**: "yes" after "Would you like me to export this?" -> generates Excel from cached data
- **Chart type preservation**: Follow-up exports use same chart type as original
- **Ambiguous resolution**: "yes" to multi-option offers forces clarification instead of guessing

### 5.4 Tone-Aware UX

The system treats **emotional state as a first-class concern**:

- **Distressed** (death, accident, hospital): Lead with empathy before policy
- **Urgent** (deadline today): Direct and efficient, actionable first
- **Frustrated**: Acknowledge briefly, then concrete steps
- **Confused**: Numbered steps, extra clarity
- **Celebratory**: Match positive energy

Tone context flows: `classify_intent` -> `build_tool_plan` (hints) -> `summarize_results` (tone rules in prompt).

### 5.5 Source of Truth Discipline

Critical rule repeated across the codebase:

> **"The database is the SOURCE OF TRUTH for personal entitlements. Policy documents are for procedural details only."**

This prevents the AI from hallucinating leave balances based on policy text. If a user's record shows 0 days, the AI says: *"I checked your records and do not see an annual leave entitlement on file."*

### 5.6 Graceful Degradation

| Component | Fallback |
|-----------|----------|
| Gemini primary model | Auto-switch to `gemini-3.5-flash` |
| Gemini 429 rate limit | Retry with `retry-after` sleep |
| Gemini JSON mode error | Retry without `response_format` |
| DAB server down | `discover_tools()` catches exception, sets empty cache |
| HANA server down | Returns empty result with warning |
| RAG unavailable | `retrieve_policy_context()` returns `""`, summarizer proceeds without it |
| Intent classification fails | Returns neutral default (`general_hr`, routine, neutral) |
| Tool plan parsing fails | Returns empty steps -> agent asks for clarification |
| Chart generation fails | Logs error, returns `""` (no chart appended) |
| Excel export fails | Returns `""` (no link prepended) |
| OpenTelemetry unavailable | `_NullMeter` no-op fallback |
| Gemini batch embedding fails | Per-item fallback with zero-vector padding |
| Schema index cache stale | Rebuild from scratch with disk persistence |
| Code resolver fails | Proceeds without code context |
| Semantic schema retrieval fails | Returns full schema (no pruning) |

### 5.7 Caching Strategy

| Cache | Scope | Warmed When | TTL |
|-------|-------|-------------|-----|
| `CACHED_TOOLS` | Global | App startup (`discover_tools`) | -- |
| `CACHED_TOOLS_PROMPT` | Global | App startup | -- |
| `CACHED_SCHEMA` | Global | App startup + per-table | -- |
| `distinct_values` | Per-column in schema | App startup (top 20 per categorical column) | -- |
| `classify_intent` | Per-query | LRU cache (128 entries) | -- |
| `IntentExemplarIndex` | Global | Lazy build on first intent classification | In-memory (Jaccard fallback if no API key) |
| `schema formatting` | Per-tenant | On first use | 5 min |
| `DAB entities` | Per-tenant | On first use | 5 min |
| `HANA clients` | Per-tenant | Startup (`hana_manager.warmup`) | -- |
| `SchemaFieldIndex` | Per-tenant | Startup (`warm_schema_index`) | 1 hr (disk) |
| `CodeResolver` | Per-tenant | On first use | 5 min |
| `Session state` | Per-user | Every turn | In-memory only |
| **Finance tables** | **Global** | **Startup (`initialize_finance_tables`)** | **1 hr (configurable)** |
| **HANA Schema Registry** | **Per-tenant** | **Startup + background refresh** | **Configurable (default 1 hr)** |
| **Metric extraction** | **Per-query** | **On hybrid extraction** | **5 min (TTL, 256 entries)** |
| **External market data (DOSM)** | **Global** | **On forecast request** | **1 day** |
| **External market data (World Bank)** | **Global** | **On forecast request** | **7 days** |
| **External market data (yfinance)** | **Global** | **On forecast request** | **4 hours** |
| **Forecast code templates** | **Global** | **On first generation** | **In-memory (128 entries)** |

---

## 6. Configuration Reference

Key environment variables (full list in `config/.env.example`):

| Env Var | Default | Purpose |
|---------|---------|---------|
| `GEMINI_API_KEY` | -- | Primary LLM API key |
| `HANAMCP_HTTP_URL` | `http://localhost:3100` | SAP HANA MCP endpoint |
| `JWT_SECRET` | `local-dev-secret-change-me` | JWT signing |
| `AUTH_MODE` | `test` | `test` / `production` / `disabled` |
| `CHART_MODE` | `auto` | `matplotlib` / `mermaid` / `auto` / `base64` |
| `SANDBOX_MEMORY_MB` | `512` | Sandbox memory limit |
| `SANDBOX_TIMEOUT_SECONDS` | `30` | Sandbox timeout |

---

## 7. Technology Stack

| Layer | Technology |
|-------|------------|
| Web Framework | FastAPI |
| Chat UI | LibreChat |
| MCP Protocol | `mcp` Python SDK (FastMCP + ClientSession) |
| DAB Transport | JSON-RPC 2.0 over SSE |
| HANA Transport | HTTP JSON-RPC (port 3100) |
| **LLM (Primary)** | **Google Gemini 3.5 Flash (OpenAI-compatible endpoint)** |
| Embeddings | Google Gemini Embedding-2 (free tier) |
| Vector DB | ChromaDB (persistent) |
| Semantic Search | NumPy + cosine similarity + keyword hybrid |
| Ingestion | PyMuPDF (fitz) + LlamaIndex + SentenceSplitter |
| Databases | SQL Server (via DAB / pyodbc) + SAP HANA (via hana-mcp-server) |
| HANA MCP Server | Node.js (`hana-mcp-server`) |
| Auth | PyJWT + YAML tenant mappings |
| Export | pandas + openpyxl |
| Charts | matplotlib (Agg) + Mermaid syntax |
| **Forecasting** | **nixtla (sandboxed subprocess, AST pre-flight + restricted builtins)** |
| **External data** | **requests + yfinance + pandas + nixtla** |
| Metrics | OpenTelemetry with `_NullMeter` fallback |
| Validation | jsonschema (Draft7Validator) |
| Testing | Test JWT generator (`tools/generate_test_jwt.py`) |

---

## 8. Files & Responsibilities

```
agent/
  main.py                    # FastAPI app, auth, rate limits, tool discovery
  config.py                  # Centralized env/config constants
  code_resolver.py           # Zero-touch codesetup reverse index

  core/
    agentic_executor.py      # Reflexive 4-stage pipeline (Intent -> Plan -> Execute -> Summarize)
    intent_classifier.py     # Two-stage LLM intent + tone classification with embedding exemplar retrieval
    intent_config.py         # Single source of truth loader for config/intents.yaml
    eval_intent_classifier.py # Per-intent precision/recall evaluation with CI gating hooks
    tool_planner.py          # Permission-aware dual-path planner
    prompts.py               # System prompt builders
    session_state.py         # Lightweight session state for follow-up actions
    guards.py                # L1/L2 intent guards
    sandbox.py               # Hardened subprocess sandbox

  integrations/
    dab_client.py            # MCP-DAB bridge client (SSE)
    hana_client.py           # SAP HANA MCP client (HTTP)
    dynamic_dashboard.py     # Dynamic dashboard FastAPI router + embedded HTML
    llm_client.py            # Unified LLM client (Gemini primary)
    rag_retriever.py         # ChromaDB + Gemini retriever
    schema_index.py          # In-memory semantic field index
    schema_registry.py       # Per-tenant HANA schema registry

  output/
    chart_generator.py       # matplotlib + Mermaid charts
    excel_exporter.py        # pandas Excel export
    export_service.py        # Export metadata + chart integration
    binning_orchestrator.py  # Client-side binning

  dab/                       # DRY utilities: response extraction, metrics, validation, OData normalization
  auth/                      # JWT, tenant resolution, permission enforcement
  summarizer/                # 3-stage response (Inform -> Assist -> Offer feedback)
  forecasting/               # External data + code generation + output parsing
  ingestion/                 # PDF -> ChromaDB pipeline

config/
  .env                       # Secrets
  dab-config.json            # DAB entity definitions
  finance_config.yaml        # Finance table discovery prefixes + fallback tables
  intents.yaml               # Single source of truth for intents, categories, keywords, exemplars
  intents_eval.yaml          # Intent classification eval dataset + CI gating thresholds
  tenant_mappings.yaml       # Multi-tenant RBAC + DAB URL routing

hana-mcp-server/             # SAP HANA MCP server (Node.js, upstream repo)
```


## 9. Current State Assessment

### What's Working Well

- Multi-tenant RBAC with permission-aware tool filtering
- OpenAI-compatible API with streaming
- Gemini 3.5 Flash primary LLM (8,000-token effective context window; see `CONTEXT_WINDOW_SIZE` in `agent/config.py`)
- DAB-based HR data abstraction + SAP HANA finance integration
- Config-driven intent classification with two-stage LLM routing and embedding-based exemplar retrieval
- Config-driven finance table discovery with YAML fallback
- Registry-driven executor with unified DAB/HANA execution contract
- RAG with Gemini embeddings and conservative rate limiting
- Tone-aware responses and source-of-truth discipline
- Excel export, chart generation, and chart artifact injection
- Session state for follow-up actions, code resolver, ambiguous response handling
- Forecasting engine with sandboxed execution and external market enrichment
- Dynamic dashboard builder served by the `/dashboard.html` router with direct DAB queries (no Superset MCP or preview-first workflow)

### Areas for Improvement

- DAB server startup lacks retry logic (fails silently if DB is down)
- Rate limiter duplication between `GeminiRateLimiter` and `GeminiGenAIEmbedder`
- Session state is in-memory only (lost on restart)
- Schema token budget is conservative relative to the model's context window
- No unit tests for forecasting modules
- yfinance is unofficial API; DOSM/WB have data lag limitations


*End of Architecture Document*


