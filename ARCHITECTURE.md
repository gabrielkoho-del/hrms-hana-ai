# HR AI Agent + DAB Data Layer — Architecture Document

> **Generated from live codebase analysis**  
> **Workspace:** `C:\Users\USER\Documents\hrms-hana-ai`  
> **Last updated:** 2026-07-29 (Manual update based on code review)

---

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
| **Dynamic finance table discovery** | SAP HANA schema scanning at startup |
| **Config-driven finance intent detection** | YAML-configured keywords and table patterns |

### Design Philosophy

> **1st priority: Production practice** — layered defense, declarative config, telemetry, strict validation, deterministic fallbacks.  
> **2nd priority: DRY consolidation** — shared utilities (`agent/dab/`), config-driven registries, no duplicated logic.  
> **3rd priority: Agentic / non-hardcoding** — LLM-driven reasoning, dynamic planning, self-healing.

---

## 2. High-Level Architecture

```
┌──────────────┐      ┌─────────────────────────────┐      ┌─────────────┐
│  LibreChat   │──────│  FastAPI (Port 8000)        │──────│  DAB Bridge │
│  (Chat UI)   │      │  • /v1/chat/completions     │      │  (Port 5000)│
└──────────────┘      │  • /health                  │      │  • MCP/SSE  │
                      │  • /charts/*                │      │  • OData    │
                      │  • /exports/*               │      └──────┬──────┘
                      └─────────────────────────────┘             │
                                                               │
                                                    ┌────────┴────────┐
                                                    │  Reflexive Agent │
                                                    │  (4-Stage Loop)  │
                                                    └────────┬────────┘
                                                             │
                                                    ┌────────┼──────────┐
                                                    │          │          │
                                            ┌───────┴─────┐┌───┴────┐┌────┴──────┐
                                            │  Intent   ││  RAG  ││  Schema   │
                                            │Classifier ││ChromaDB││Field Index│
                                            │  (Gemini) ││+Gemini││(Gemini+Num│
                                            └───────────┘└───────┘│Py hybrid) │
                                                                    └──────────┘
                                                                       │
                                                            ┌────────┴────────┐
                                                            │  SAP HANA MCP   │
                                                            │  (Port 3100)    │
                                                            │  • HTTP         │
                                                            │  • 4 finance    │
                                                            │    tools        │
                                                            └─────────────────┘
```

---

## 3. Component Inventory

### 3.1 LLM Client (`agent/llm_client.py`)

The **LLM client** uses Google Gemini 3.1 Flash Lite via OpenAI-compatible endpoint.

#### Model Tiers

```python
MODEL_TIERS = {
    "planner":   {"model": "gemini-3.1-flash-lite", "temperature": 0.1, "max_tokens": 8192},
    "executor":  {"model": "gemini-3.1-flash-lite", "temperature": 0.1, "max_tokens": 8192},
    "responder": {"model": "gemini-3.1-flash-lite", "temperature": 0.3, "max_tokens": 8192},
}
```

#### Key Features

- **Unified OpenAI-compatible format**: Uses Gemini 3.1 Flash Lite API
- **Tier-aware model selection**: Different temperature/token settings per pipeline stage
- **Token budget tracking**: Daily RPD tracking (500/day free tier) with actual usage from API response headers
- **Automatic fallback chain**:
  1. Model-not-found → fallback to `gemini-3.5-flash`
  2. Rate limit (429) → retry with `retry-after` sleep
  3. JSON mode error → retry without `response_format`

#### Budget & Rate Limits (Gemini Free Tier)

| Limit | Value | Safety Margin |
|-------|-------|---------------|
| TPM (tokens/minute) | 250,000 | Effectively unlimited for solo dev |
| RPM (requests/minute) | 15 | Tracked but not enforced |
| RPD (requests/day) | 500 | Warns at 400 (80%), hard stop at 500 |

---

### 3.2 DAB Client (`agent/dab_client.py`) / HANA Client (`agent/hana_client.py`)

The system has two structured-data sources, both exposed as MCP-style tools to the agent.

#### DAB Client

The **MCP-DAB Bridge client** connects to the DAB server via JSON-RPC over SSE transport.

##### Key Features

- **3-path response parsing**: Plain JSON → SSE with event/data markers → Raw JSON body fallback
- **JSON-RPC envelope unwrapping**: Automatically extracts `result` from `{"jsonrpc": "2.0", "result": {...}}`
- **Session-aware**: Extracts `Mcp-Session-Id` from initialize response, sends on subsequent calls
- **Exponential backoff**: `0.5 × 2^attempt` seconds between retries (max 3)
- **Error format alignment**: Returns `{"isError": True, "message": "..."}` consistently

##### DABTenantClientManager

Per-tenant client routing via `tenant_mappings.yaml`:

| Tenant ID → DAB URL → Client instance → Entity cache (5-min TTL) |

| Method | Purpose |
|--------|---------|
| `get_client(tenant_id)` | Sync lazy init (for startup/tool discovery) |
| `get_client_async(tenant_id)` | Async-safe init (for request handling) |
| `get_entities(tenant_id)` | Cached entity discovery with `describe_entities` |
| `invalidate_cache(tenant_id)` | Force refresh on schema changes |
| `close_all()` | Graceful per-client shutdown |

##### Exposed DAB Tools (3 tools)

| Tool | Purpose | Parameters |
|------|---------|------------|
| `read_records` | Query with OData filtering | `entity`, `select`, `filter`, `orderby`, `first` |
| `aggregate_records` | Grouped aggregations | `entity`, `function`, `field`, `groupby`, `having`, `filter`, `first` |
| `describe_entities` | Schema discovery | _(none)_ |

#### HANA Client

The **SAP HANA MCP client** connects to the `hana-mcp-server` (Node.js) via HTTP transport.

##### Key Features

- **HTTP transport**: Path is `POST {HANAMCP_HTTP_URL}/mcp` (JSON-RPC over HTTP), avoiding STDIO process-management anti-patterns inside FastAPI
- **Result normalization**: Converts HANA `{"columns": [...], "rows": [...]}` into agent-standard `{"result": [dicts], "message": "..."}` via `normalize_hana_result()`
- **HanaTenantClientManager**: Registry keyed by `db_name`, `schema_name`, and `tenant_id`, with lazy init and `close_all()` shutdown
- **Schema-agnostic downstream**: The executor writes HANA results into `state["tool_results"]` in the same shape as DAB, so summarizer/chart/export logic remains unchanged

##### Finance Table Discovery (NEW)

Finance tables are discovered dynamically from HANA schema at startup, not hardcoded:

1. `agentic_executor.py` calls `initialize_finance_tables(hana_client)` once per process lifetime
2. `intent_classifier.py` queries `hana_list_tables` and filters results by configurable prefixes (`FAGL`, `BKPF`, `BSEG`, `SKA`, `CSK`, `T001`, `TCUR`)
3. Discovered tables are cached in-memory with a TTL (default 1 hour, configurable via `schema_cache_ttl_seconds`)
4. If HANA is unavailable at startup, falls back to hardcoded table names from `config/finance_config.yaml`

This replaces the previous hardcoded `_HANA_FINANCE_TABLES` set with a dynamic, config-driven approach.

| Aspect | Detail |
|--------|--------|
| Discovery trigger | Once at startup, per process |
| Cache key | None needed — single global set per process |
| Cache TTL | 3600s (configurable) |
| Fallback | `config/finance_config.yaml` → `finance_table_names` |
| Config location | `config/finance_config.yaml` |

##### Finance Keyword Configuration (NEW)

Finance intent keywords are loaded from `config/finance_config.yaml` at import time, not hardcoded in source:

| Config Key | Purpose | Default |
|------------|---------|---------|
| `finance_keywords` | Substring-matched terms for fast-path intent detection | Narrowed list (see config file) |
| `finance_table_prefixes` | HANA table name prefixes for dynamic discovery | `FAGL`, `BKPF`, `BSEG`, `SKA`, `CSK`, `T001`, `TCUR` |
| `finance_table_names` | Fallback tables if schema discovery fails | `FAGLFLEXA`, `BKPF`, `BSEG`, `SKA1`, `SKAT`, `CSKS`, `CSKT`, `T001`, `TCURC`, `TCURR` |
| `schema_discovery_tool` | HANA tool used for discovery | `hana_list_tables` |
| `schema_cache_ttl_seconds` | How long to cache discovered tables | 3600 |

| Tool | Purpose | Parameters |
|------|---------|------------|
| `hana_execute_query` | Execute SQL against HANA (supports SELECT/WITH) | `query`, `maxRows`, `includeTotal` |
| `hana_describe_table` | Describe table columns/types | `table_name`, `schema_name`, `catalog_database` |
| `hana_list_tables` | List tables in a schema with optional prefix filter | `schema_name`, `prefix`, `limit`, `offset` |
| `hana_get_sample_data` | Fetch sample rows from a HANA table (SELECT TOP N) | `table_name`, `schema_name`, `limit` |

---

### 3.3 Reflexive Agent Pipeline (`agent/agentic_executor.py`)

The core execution engine. Four stages:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   L1 Guard      │ -> │ Stage 0: Intent  │ -> │ Stage 1: Plan   │ -> │ Stage 2: Execute│
│  Regex fast path│    │  + Tone classify │    │  Tool selection │    │  DAB + HANA     │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
                                                                               │
                                                                               v
                                                                      ┌─────────────────┐
                                                                      │ Stage 3: Summarize│
                                                                      │  NL response      │
                                                                      └─────────────────┘
```

#### L1 Guard

Deterministic regex matching for greetings/smalltalk. Returns immediately **without LLM calls** — saves tokens and latency.

#### Stage 0: Intent & Tone Classification (`agent/intent_classifier.py`)

Single Gemini call (~200 tokens) returns structured `IntentResult`:

```python
@dataclass
class IntentResult:
    intent: str                      # e.g., "leave_request", "salary_analysis"
    intent_category: str             # personal_data | aggregate_data | policy_info | action_request | emergency | grievance | greeting | finance_gl_analysis | finance_cost_analysis | finance_currency | finance_budget
    data_scope: str                  # individual | aggregate | none
    chart_eligible: bool             # True only for aggregate_data
    urgency_level: str               # routine | time_sensitive | urgent | distressed
    emotional_state: str             # neutral | anxious | frustrated | celebratory | grieving | confused
    topic_sensitivity: str           # low | medium | high
    needs_empathy: bool
    confidence: float
    action_oriented: bool
    wants_export: bool               # True if user explicitly asks to export/download
    is_ambiguous: bool               # True if "yes"/"ok" to multi-option offer
    finance_query: bool = False      # True if query involves SAP FI/CO finance data
```

**New capabilities**:
- **Follow-up affirmative detection**: Short responses like "yes", "ok", "sure" are resolved against conversation context (export offers, drill-down offers)
- **Ambiguous response handling**: When prior turn offered multiple options, forces clarification instead of guessing
- **Session state**: Caches last query, intent, chart data, export URL for follow-up turns

**Tone context propagates through the entire pipeline** — planner, executor, and summarizer all receive it.

#### Stage 1: Tool Planning (`agent/tool_planner.py`)

**Dual-path architecture** (production-grade fallback):

- **Path 1 (Primary)**: Native OpenAI-compatible tool calling — Gemini emits structured `tool_calls` for DAB tools (`read_records`, `aggregate_records`, `describe_entities`) and HANA tools (`hana_execute_query`, `hana_describe_table`, `hana_list_tables`, `hana_get_sample_data`).
- **Path 2 (Fallback)**: JSON text parsing with non-greedy regex extraction.

Key features:
- **Permission-aware tool filtering**: Only tools the user has permissions for are presented to the LLM
- **Token-budget-aware schema injection**: DAB and HANA schemas are merged under a shared token budget, with progressive pruning for large DAB schemas
- **Registry-driven tool discovery**: HANA schemas are appended at startup to `CACHED_TOOLS`, not fetched per-request
- **DAB filter validation**: `read:self` users must include their own `email`/`emp_id` in OData filters
- **Tone-aware guidance**: Contextual hints passed to the LLM (not hardcoded rules)
- **Dynamic binning inference**: If schema lacks pre-computed bins (e.g., `age_group`), planner fetches raw values and sets `client_side_binning` for runtime pandas binning
- **Schema token pruning**: Progressive pruning (descriptions → distinct values → fields → entities) to stay under token budget. With Gemini 3.1 Flash Lite's 1M context window, the schema budget is intentionally conservative for latency and tool-calling reliability; current value: 5,000 tokens.
- **Semantic schema retrieval**: For large DAB schemas (>25 fields), uses embedding-based field selection to reduce prompt size
- **Entity name normalization**: Schema-driven case-insensitive and fuzzy matching (no hardcoded maps)
- **Explicit chart type detection**: Detects "bar chart", "pie chart", "line graph" etc. from user query
- **Export intent inference**: Detects "export", "download", "Excel" keywords for aggregate queries
- **TTL cache**: Multi-tenant-safe schema formatting cache (5-min expiry, 100 entries)

#### Stage 2: Execution

For each planned step, the executor uses a **tool registry** (`_TOOL_REGISTRY` in `agentic_executor.py`) to route DAB and HANA calls through a unified interface:

1. **`normalize_odata_args`** (`agent/dab/odata_normalizer.py`) — Normalizes entity names, field names, and numeric types (DAB only)
2. **`enforce_tool_args`** — Pre-execution permission block (e.g., `aggregate_records` without self-filter blocked for `read:self`)
3. **`validate_dab_args`** (`agent/dab/validation.py`) — JSON Schema pre-validation against cached tool schemas (DAB only)
4. **`invoke_dab_tool_with_retry`** — SSE call to DAB bridge, 2 retries with exponential backoff (DAB only)
5. **`_execute_hana_tool_call`** — Dispatches to `hana_manager`, normalizes `{"columns", "rows"}` → `{"result": [dicts]}` via `normalize_hana_result()` (HANA only)
6. **`extract_payload`** (`agent/dab/dab_response.py`) — Unwraps MCP content wrapper, detects errors in text content (DAB only)
7. **`filter_tool_results`** — Post-execution row filtering by `email`/`emp_id` (defense in depth)
8. **Client-side dynamic binning** — Pandas applies runtime bins (age, salary, tenure) after data retrieval
9. **Code resolution** (`agent/code_resolver.py`) — Scans results for coded values, builds LLM context with descriptions
10. **Chart generation** — If aggregate + eligible, generates chart and injects raw markdown artifact into LLM context
11. **Excel export** — Chart-rich Excel with 5 worksheets (Executive Summary, Distribution Table, Chart, Raw Data, Metadata)

Agent state maintained across steps:
```python
state = {
    "tool_results": {},        # Unified dict for DAB + HANA results
    "tool_calls_made": [],
    "rag_context": "",
    "export_url": "",
}
```

#### Stage 3: Summarization (`agent/summarizer/` package)

The most sophisticated component. Modular prompt construction with **3-stage response structure**:

1. **Inform** — Present factual answer first, bold key numbers, use markdown tables
2. **Assist** — Suggest 1–3 relevant next steps (config-driven via `ASSIST_REGISTRY`)
3. **Offer feedback** — End with open invitation or escalation offer

**New capabilities**:
- **Ambiguous response override**: If `is_ambiguous=True`, Stage 3 forces clarification instead of generic feedback
- **Export-aware guidance**: When `wants_export=True`, suppresses full data walkthrough (Excel is primary deliverable)
- **Offer formatting rules**: Multiple actions must be bulleted/numbered, never combined with "or"
- **Code context injection**: Replaces raw code values with human-readable descriptions

Prompt blocks:
- **Identity**: "You are a helpful HR colleague"
- **Tone Block**: Empathy → Urgency → Frustration → Confusion → Celebration → Neutral
- **Data Protocol**: Source-of-truth rules, conflict handling, export mentions, large result handling
- **Zero-Row Guidance**: Never say "I don't have information" — explain what was checked and suggest next steps
- **Formatting Protocol**: Markdown tables, bold numbers, bullet observations, conversational tone
- **Conflict Detection**: Dynamic LLM-based comparison of RAG policy text vs. SQL facts
- **Chart Artifact Injection**: Pre-generated chart markdown is injected into the system prompt so the LLM embeds it verbatim

---

### 3.4 RAG System (`agent/rag_retriever.py`, `ingestion/`)

#### Architecture

```
PyMuPDF (fitz) → LlamaIndex Document → SentenceSplitter (1200 chars, 150 overlap)
                    ↓
            Gemini Embedding-2 (google-genai SDK)
                    ↓
            ChromaDB Persistent Vector Store
                    ↓
            Query: embed → similarity search → top-5 chunks
```

#### Rate Limiting (Conservative)

- **TPM**: 30,000 × 0.85 = 25,500 safe
- **RPM**: 100 × 0.80 = 80 safe
- **RPD**: 1,000 × 0.25 = 250 daily hard stop
- Minimum spacing: 60/80 = 0.75s between requests

`GeminiRateLimiter` is thread-safe with lock-based window tracking.

#### Ingestion Pipeline (`ingestion/ingestion.py` + `index.py`)

- Contextual chunking: prepends `Document: {title} | Section: {heading} | Page: {page}` to each chunk before embedding
- Section detection via regex: numbered headings (`1.0.`, `6.1.1.`) or ALL CAPS lines
- Footer stripping: removes "Sample Document - Malaysia HR Forum" and page numbers
- Batch insert: 10 chunks at a time
- Re-ingestion: deletes old collection, creates fresh one

---

### 3.5 Semantic Schema Index (`agent/schema_index.py`)

**Lightweight in-memory embedding index** for semantic field retrieval. No external vector DB needed — pure Python + NumPy for <500 fields.

#### Design

- **Batch-first embedding**: `embed_content(contents=LIST)` — single HTTP call for all fields (Gemini API batch limit: 100 items)
- **Per-item fallback**: Only on batch failure (429, payload too large)
- **Hybrid search**: 60% cosine similarity + 40% keyword overlap (Jaccard-like with exact-match bonus)
- **Disk cache**: Persisted as `.npz` files keyed by schema content hash, 1-hour TTL
- **Lazy init**: Loaded on first use, cached in memory per tenant
- **Warm at startup**: `warm_schema_index()` called during tool discovery to pre-build index off the request path

#### Usage

```python
from agent.schema_index import search_relevant_fields
fields = await search_relevant_fields(user_query, cached_schema, tenant_id, top_k=8)
# Returns: [{entity, field, embedding_score, keyword_score, hybrid_score}, ...]
```

---

### 3.6 Code Resolver (`agent/code_resolver.py`)

**Zero-touch code resolution** for multi-tenant DAB production. Fetches `codesetup` from each tenant's DAB and builds a reverse index for LLM context injection.

#### Design

- **Per-tenant singleton**: Each tenant has its own `CodeResolver` instance with isolated cache
- **Cursor-based pagination**: Handles 20K+ records via `$first` + `$after` (DAB cursor pagination)
- **TTL cache**: 300 seconds, auto-refresh on expiry
- **Heuristic filtering**: Skips IDs, dates, emails, currency, UUIDs — only resolves human-readable short codes
- **Ambiguity handling**: If code "A" maps to multiple types, all matches are shown; LLM uses field context to disambiguate

#### Usage

```python
from agent.dab.code_resolver import scan_for_codes
code_context_md = await scan_for_codes(tool_results, tenant_id)
# Injected into summarizer prompt before chart generation
```

---

### 3.7 Shared DAB Utilities (`agent/dab/`)

DRY-consolidated cross-cutting concerns for DAB/MCP operations.

#### `dab/dab_response.py` — DAB/MCP Response Extraction

Centralized response unwrapping.

| Function | Purpose |
|----------|---------|
| `extract_items(result)` | Deep extraction of list-of-dicts from any DAB response format |
| `extract_payload(result)` | Unwrap MCP `CallToolResult` to inner DAB payload; detects errors in text content |
| `is_dab_error(result)` | Boolean check for DAB error state |
| `extract_items_with_meta(result)` | Extract items + pagination metadata |
| `format_dab_items_context(...)` | Format DAB items into human-readable context lines for LLM |

#### `dab/metrics.py` — OpenTelemetry Metrics

Lazy-initialized OTel metrics with `_NullMeter` fallback.

| Metric | Type | Labels |
|--------|------|--------|
| `dab_tool_calls` | Counter | `tool`, `status` |
| `dab_errors` | Counter | `error_type`, `entity` |
| `charts_generated` | Counter | `chart_type`, `mode` |
| `entity_name_normalized` | Counter | `type`, `original`, `corrected` |

#### `dab/validation.py` — JSON Schema Validation

Pre-execution validation of DAB tool arguments against cached Groq-style tool schemas using `jsonschema.Draft7Validator`.

#### `dab/odata_normalizer.py` — OData Argument Normalization

Normalizes entity names, field names, and numeric types before DAB execution. Schema-driven, no hardcoded maps.

---

### 3.8 Auth System (`agent/auth/`)

#### JWT Validation (`auth/models.py`)

Supports dual algorithms:
- **HS256**: Local dev with `JWT_SECRET` from `.env`
- **RS256**: Production with issuer validation (`JWT_ISSUER_ALLOWLIST`)

Claims extracted:
- `tenant_id` / `tid` / `org_id`
- `groups` / `roles` (from Azure AD / Okta)
- `email` / `upn` / `preferred_username`
- `emp_id` (optional, for direct self-reference)
- `oid` / `sub` (user ID)

#### Tenant Resolution (`auth/tenant_resolver.py`)

YAML-based multi-tenant RBAC (`config/tenant_mappings.yaml`):

```yaml
tenants:
  XYZ:
    group_mappings:
      "Human Resources Team": HRMS_HR
      "Regional Managers": HRMS_MANAGER
      "Employees": HRMS_EMPLOYEE
  LOCALDEV:
    group_mappings:
      "TestHR": HRMS_HR
      "TestEmployee": HRMS_EMPLOYEE
```

Resolution chain: **External Groups → Internal Roles → Permissions**

#### Permission Enforcement

Two-layer defense:

1. **Pre-execution** (`enforce_tool_args()`): Blocks tool calls entirely
   - `aggregate_records` → blocked for `read:self` without self-filter
   - `read_records` → DAB filter must contain `email` or `emp_id` for `read:self`

2. **Post-execution** (`filter_tool_results()`): Filters result rows
   - For `read:self` users, strips rows where `email` != user's email or `emp_id` != user's emp_id
   - If no identifiable columns, allows through (assumes aggregate data)

#### Auth Modes (`auth/dependencies.py`)

| Mode | Behavior |
|------|----------|
| `test` | Falls back to `DEV_TEST_TOKEN` from `.env` if no header |
| `production` | Requires valid `Authorization: Bearer` header |
| `disabled` | Anonymous context — **no permissions granted** (not open access!) |

---

### 3.9 Visualization & Export

#### Excel Exporter (`agent/excel_exporter.py`)
- pandas + openpyxl
- **Chart-rich export**: 5 worksheets (Executive Summary, Distribution Table, Chart, Raw Data, Metadata)
- Auto-generated when result rows > `LARGE_RESULT_THRESHOLD` (default 20) or user explicitly requests export
- Served at `/exports/{filename}`
- Auto-cleanup after 24 hours

#### Chart Generator (`agent/chart_generator.py`)
- matplotlib (Agg backend) + pandas + Mermaid syntax
- Types: `bar`, `barh`, `pie`, `hist`, `line`, `box`
- **Environment-driven output modes**:
  - `CHART_MODE=matplotlib` → PNG files served via `/charts/{filename}`
  - `CHART_MODE=mermaid` → Mermaid syntax rendered natively by LibreChat
  - `CHART_MODE=auto` → Mermaid for simple, matplotlib for complex
  - `CHART_MODE=base64` → Inline base64 PNG (air-gapped)
- Privacy enforcement: blocks charts with individual employee identifiers (`emp_id`, `email`, `name`, etc.)
- Color-blind friendly palette (`COLORBLIND_PALETTE`)
- Auto-extracted from tool results via `agent/dab/dab_response.extract_items()`
- **Chart artifact injection**: Raw markdown is injected into the LLM context so the summarizer embeds it verbatim

---

## 4. Data Flow (Request Lifecycle)

```
┌────────┐
│  User  │  POST /v1/chat/completions
│        │  Authorization: Bearer <JWT>
└───┬────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  FastAPI Middleware                     │
│  • CORS (allow all origins)             │
│  • Rate limit: 30 req/min per IP        │
└───┬─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  verify_token()                         │
│  1. Extract Bearer token from header    │
│  2. Or fall back to DEV_TEST_TOKEN      │
│  3. Validate JWT (HS256/RS256)          │
│  4. Extract tenant_id, groups, email     │
│  5. Resolve groups → roles → permissions│
│  → returns AuthContext                  │
└───┬─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  run_agent(messages, auth_context)      │
│  ─────────────────────────────────────  │
│  1. Extract user_query from last msg    │
│  2. Build conversation_history (last 5  │
│     turns: user + assistant, 200 chars)  │
│                                         │
│  3. L1 GUARD: Regex check greeting?    │
│     → Yes: return greeting response     │
│                                         │
│  4. classify_intent(user_query)         │
│     → IntentResult (tone_context)        │
│     [LLM call #1 — Gemini]              │
│                                         │
│  5. build_tool_plan(user_query,         │
│     tone_context, cached_tools,          │
│     cached_schema, auth_context)         │
│     → JSON plan with steps, chart, rag  │
│     [LLM call #2 — Gemini]              │
│                                         │
│  6. If direct_answer: return it         │
│                                         │
│  7. For each step in plan:               │
  │     DAB path:                           │
  │     a. normalize_odata_args()           │
  │        → entity/field case normalization │
  │     b. enforce_tool_args(tool, args)    │
  │        → blocked? store error, skip      │
  │     c. validate_dab_args()              │
  │        → JSON Schema pre-validation      │
  │     d. invoke_dab_tool_with_retry()     │
  │        → SSE call to DAB bridge          │
  │     e. extract_payload()                 │
  │        → unwrap MCP content wrapper      │
  │     f. filter_tool_results()            │
  │        → post-execution row filtering    │
  │     g. parse JSON result                │
  │                                         │
  │     HANA path:                          │
  │     a. _execute_hana_tool_call()        │
  │        → HTTP dispatch                  │
  │     b. normalize_hana_result()          │
  │        → columns/rows → result: [dicts] │
  │     c. filter_tool_results()            │
  │        → post-execution row filtering    │
  │                                         │
  │     h. extract chartable data           │
  │        → generate_chart() if applicable  │
  │                                         │
  │  8. CLIENT-SIDE BINNING (if planned):   │
  │     → apply_client_side_binning()        │
  │                                         │
  │  9. CODE RESOLUTION (if codes found):   │
  │     → scan_for_codes() → LLM context     │
  │                                         │
  │  10. EXCEL EXPORT (if requested):       │
  │     → export_to_excel_with_chart()       │
  │                                         │
  │  11. If plan.rag: retrieve_policy_context()│
  │     → ChromaDB query with Gemini embed   │
  │                                         │
  │  12. summarize_results()                 │
  │     → build_response_structure()         │
  │     → build_tone_block(tone_context)      │
  │     → build_data_protocol()              │
  │     → detect_conflicts(rag vs sql)       │
  │     → build_zero_row_guidance()          │
  │     → build_formatting_protocol()        │
  │     → chart artifact injection           │
  │     [LLM call #3 — Gemini]              │
  │                                         │
  │  13. Save session state for follow-up   │
  │  14. Return final text                  │
└───┬─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Response Formatter                     │
│  • stream=true → SSE chunks (10 chars)  │
│  • stream=false → JSON OpenAI format    │
└───┬─────────────────────────────────────┘
    │
    ▼
┌────────┐
│  User  │  ← Natural language response
│        │  ← Optional: Excel link, Chart image
└────────┘
```

### LLM Call Count per Request

| Path | # LLM Calls | Purpose |
|------|------------|---------|
| Greeting/smalltalk | **0** | Regex guard, zero cost |
| Standard DB query | **3** | Intent + Plan + Summarize |
| With conflict detection | **3–4** | + optional `detect_conflicts` |

### Model Context Window

| Model | Context | Output |
|-------|---------|--------|
| **Gemini 3.1 Flash Lite** (primary) | **1,048,576 tokens (~1M)** | 65,536 tokens |

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
- **Export follow-up**: "yes" after "Would you like me to export this?" → generates Excel from cached data
- **Chart type preservation**: Follow-up exports use same chart type as original
- **Ambiguous resolution**: "yes" to multi-option offers forces clarification instead of guessing

### 5.4 Tone-Aware UX

The system treats **emotional state as a first-class concern**:

- **Distressed** (death, accident, hospital): Lead with empathy before policy
- **Urgent** (deadline today): Direct and efficient, actionable first
- **Frustrated**: Acknowledge briefly, then concrete steps
- **Confused**: Numbered steps, extra clarity
- **Celebratory**: Match positive energy

Tone context flows: `classify_intent` → `build_tool_plan` (hints) → `summarize_results` (tone rules in prompt).

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
| Tool plan parsing fails | Returns empty steps → agent asks for clarification |
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
| `CACHED_TOOLS` | Global | App startup (`discover_tools`) | — |
| `CACHED_TOOLS_PROMPT` | Global | App startup | — |
| `CACHED_SCHEMA` | Global | App startup + per-table | — |
| `distinct_values` | Per-column in schema | App startup (top 20 per categorical column) | — |
| `classify_intent` | Per-query | LRU cache (128 entries) | — |
| `schema formatting` | Per-tenant | On first use | 5 min |
| `DAB entities` | Per-tenant | On first use | 5 min |
| `HANA clients` | Per-tenant | Startup (`hana_manager.warmup`) | — |
| `SchemaFieldIndex` | Per-tenant | Startup (`warm_schema_index`) | 1 hr (disk) |
| `CodeResolver` | Per-tenant | On first use | 5 min |
| `Session state` | Per-user | Every turn | In-memory only |
| **Finance tables** | **Global** | **Startup (`initialize_finance_tables`)** | **1 hr (configurable)** |

---

## 6. Database Schema (Inferred)

From DAB `describe_entities` and schema discovery code:

### `employee` table
```
emp_id (PK)
first_name, last_name
email
job_title
department
salary
hire_date
status (Active/Inactive)
manager_id (FK to employee)
... other fields
```

### `leave_entitlement` table
```
entitlement_id (PK)
emp_id (FK)
leave_type (e.g., 'Annual Leave', 'Sick Leave', 'Maternity Leave')
year
entitled_days
taken_days
carried_forward
forfeited_days
balance_days
```

### `employee_leave` table
```
leave_id (PK)
emp_id (FK)
leave_type
start_date, end_date
days_requested
reason
status (Pending/Approved/Rejected)
requested_at
approved_at
approved_by
```

### `codesetup` table
```
codesetup_id (PK)
type (e.g., 'STATUS', 'GENDER', 'DEPARTMENT')
code (e.g., 'A', 'SL', 'ENG')
description (e.g., 'Active', 'Sick Leave', 'Engineering')
```

### `INFORMATION_SCHEMA` tables
Used for auto-discovery and schema caching.

---

## 7. Configuration Reference

| Env Var | Default | Purpose |
|---------|---------|---------|
| `GEMINI_API_KEY` | — | **Primary LLM API key** |
| `DB_CONN_STR` | ODBC to HRMSlocal | Legacy; DAB handles DB now |
| `MCP_SERVER_URL` | `http://localhost:5000` | DAB fallback URL |
| `HANAMCP_HTTP_URL` | `http://localhost:3100` | SAP HANA MCP HTTP endpoint |
| `CHROMA_DB_PATH` | `./chroma_db` | Vector store location |
| `CHROMA_COLLECTION_NAME` | `hr_policies` | Chroma collection |
| `SCHEMA_CACHE_DIR` | `data/schema_cache` | Schema index disk cache |
| `JWT_SECRET` | `local-dev-secret-change-me` | JWT signing |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |
| `DEV_TEST_TOKEN` | — | Test mode fallback token |
| `AUTH_MODE` | `test` | `test` / `production` / `disabled` |
| `AGENT_MODE` | `reflexive` | Agent execution mode |
| `DEFAULT_MAX_ROWS` | `100` | Default query limit |
| `LARGE_RESULT_THRESHOLD` | `20` | Auto-export trigger |
| `RATE_LIMIT_PER_MINUTE` | `30` | API rate limit |
| `EXPORT_DIR` | `./agent/output/exports` | Excel output directory |
| `CHART_OUTPUT_DIR` | `./agent/output` | Chart PNG output directory |
| `CHART_MODE` | `auto` | `matplotlib` / `mermaid` / `auto` / `base64` |
| `CHART_MAX_CATEGORIES` | `20` | Max categories in chart |
| `OTEL_SERVICE_NAME` | `hr-ai-agent` | OpenTelemetry service name |
| `DAB_DISCOVERY_TENANT` | `RDEMOROCKFORT` | Default tenant for tool discovery |
| `FINANCE_TABLE_CACHE_TTL` | `3600` | Finance table discovery cache TTL (seconds) |

---

## 8. Technology Stack

| Layer | Technology |
|-------|------------|
| Web Framework | FastAPI |
| Chat UI | LibreChat |
| MCP Protocol | `mcp` Python SDK (FastMCP + ClientSession) |
| DAB Transport | JSON-RPC 2.0 over SSE |
| HANA Transport | HTTP JSON-RPC (port 3100) |
| **LLM (Primary)** | **Google Gemini 3.1 Flash Lite (OpenAI-compatible endpoint)** |
| Embeddings | Google Gemini Embedding-2 (free tier) |
| Vector DB | ChromaDB (persistent) |
| Semantic Search | NumPy + cosine similarity + keyword hybrid |
| Ingestion | PyMuPDF (fitz) + LlamaIndex + SentenceSplitter |
| Databases | SQL Server (via DAB / pyodbc) + SAP HANA (via hana-mcp-server) |
| HANA MCP Server | Node.js (`hana-mcp-server`) |
| Auth | PyJWT + YAML tenant mappings |
| Export | pandas + openpyxl |
| Charts | matplotlib (Agg) + Mermaid syntax |
| Metrics | OpenTelemetry with `_NullMeter` fallback |
| Validation | jsonschema (Draft7Validator) |
| Testing | Test JWT generator (`tools/generate_test_jwt.py`) |

---

## 9. Files & Responsibilities

```
hrms-hana-ai/
├── app.py                    # Legacy API (direct SQL + Ollama Qwen) — NOT part of agent architecture
├── ARCHITECTURE.md           # This document
│
├── agent/
│   ├── main.py               # FastAPI app, orchestration, auth, rate limits, tool discovery
│   ├── code_resolver.py      # Zero-touch code resolution for multi-tenant DAB (codesetup reverse index)
│   ├── config.py             # Centralized env/config constants
│   │
│   ├── core/                 # Core agent pipeline
│   │   ├── agentic_executor.py   # Reflexive 4-stage pipeline
│   │   ├── intent_classifier.py  # LLM-based intent + tone classification
│   │   ├── tool_planner.py       # Permission-aware dual-path planner
│   │   ├── prompts.py            # System prompt builders
│   │   ├── session_state.py      # Lightweight session state for follow-up actions
│   │   ├── guards.py             # L1/L2 intent guards
│   │   └── __init__.py
│   │
│   ├── integrations/         # External service integrations
│   │   ├── dab_client.py         # MCP-DAB bridge client with SSE parsing
│   │   ├── hana_client.py        # SAP HANA MCP client
│   │   ├── llm_client.py         # Unified LLM client (Gemini primary)
│   │   ├── rag_retriever.py      # ChromaDB + Gemini retriever
│   │   ├── schema_index.py       # In-memory semantic field index
│   │   ├── schema_registry.py    # HANA schema registry
│   │   └── __init__.py
│   │
│   ├── output/               # Visualization & export
│   │   ├── chart_generator.py    # matplotlib + Mermaid charts
│   │   ├── excel_exporter.py     # pandas Excel export
│   │   ├── export_service.py     # Export metadata + chart integration
│   │   ├── binning.py            # Binning configuration and metadata inference
│   │   ├── binning_orchestrator.py # Client-side binning orchestration
│   │   ├── exports/              # Generated Excel files (served via /exports/*)
│   │   └── __init__.py
│   │
│   ├── dab/                  # DRY-consolidated DAB/MCP cross-cutting utilities
│   │   ├── dab_response.py   # DAB/MCP response extraction
│   │   ├── metrics.py        # OpenTelemetry metrics with lazy init + null fallback
│   │   ├── validation.py     # JSON Schema pre-validation for DAB tool arguments
│   │   ├── odata_normalizer.py # OData argument normalization
│   │   └── __init__.py
│   │
│   ├── auth/                 # Authentication and tenant resolution
│   │   ├── dependencies.py   # FastAPI verify_token dependency
│   │   ├── models.py         # JWT validation + AuthContext dataclass
│   │   ├── tenant_resolver.py # YAML group → role → permission mapping
│   │   ├── role_resolver.py  # Auth role resolution + row-level self-access check
│   │   └── __init__.py
│   │
│   ├── summarizer/           # 3-stage response structure
│   │   ├── response_summarizer.py
│   │   ├── aggregation.py
│   │   ├── tool_results.py
│   │   ├── prompt/           # Prompt construction modules
│   │   │   ├── tone.py
│   │   │   ├── structure.py
│   │   │   ├── registry.py
│   │   │   ├── formatting.py
│   │   │   ├── export_guidance.py
│   │   │   └── data_rules.py
│   │   ├── validators/       # Response validators
│   │   │   └── conflict.py
│   │   └── __init__.py
│   │
│   └── __init__.py
│   ├── ingestion/
│   │   ├── ingestion.py          # Per-file PDF ingestion script (PyMuPDF)
│   │   ├── index.py              # LlamaIndex + ChromaDB builder with contextual chunking
│   │   └── gemini_embedder.py   # LlamaIndex-compatible Gemini embedder with rate limiting
│   │
│   ├── config/
│   │   ├── .env                  # Environment variables (secrets)
│   │   ├── dab-config.json       # DAB entity definitions
│   │   ├── dab-config.Production.json
│   │   ├── dab-config.Rymnet.json
│   │   ├── dab-config.Rymnet.test.json
│   │   ├── finance_config.yaml   # (NEW) 
│   │                                │   discovery + intent keyword configuration
│   │   ├── suggestions.yaml
│   │   └── tenant_mappings.yaml  # Multi-tenant RBAC + DAB URL routing
│   │
│   ├── data/
│   │   ├── raw_docs/             # Source PDFs (e.g., Employee Handbook)
│   │   ├── chroma_db/            # Persistent ChromaDB vector store
│   │   └── schema_cache/         # Schema index disk cache (.npz files)
│   │
│   ├── sql/                      # SQL scripts and schema definitions
│   │
│   ├── tools/
│   │   └── generate_test_jwt.py # JWT generator for local testing
│   │
│   ├── venv311/                  # Python virtual environment
│   │
│   ├── hana-mcp-server/          # SAP HANA MCP server (Node.js)
│   │   ├── src/
│   │   │   ├── server/           # MCP server implementation
│   │   │   └── constants/        # Tool definitions, permissions
│   │   ├── hana-mcp-server.js   # Entry point
│   │   └── package.json
│   │
│   ├── requirements.txt          # Python dependencies
│   └── package.json              # Node.js dependencies (for hana-mcp-server)
│   
│   ---
│   
│   ## 10. Current State Assessment
│   
│   ### What's Working Well
│   
│   - ✅ Multi-tenant RBAC with permission-aware tool filtering
│   - ✅ OpenAI-compatible API with streaming support
│   - ✅ **Gemini 3.1 Flash Lite as primary LLM** (1M context, 64K output)
│   - ✅ DAB-based data abstraction for HR data (replaces custom MCP server)
│   - ✅ **Config-driven finance keywords** — loaded from `config/finance_config.yaml`, not hardcoded in source
│   - ✅ **SAP HANA integration** via HTTP-first MCP client with result normalization (`hana_client.py`)
│   - ✅ Registry-driven executor (`_TOOL_REGISTRY`) — idiomatic replacement for prefix string-matching
│   - ✅ Unified execution contract — DAB and HANA results share `state["tool_results"]` shape
│   - ✅ RAG with free Gemini embeddings and conservative rate limiting
│   - ✅ Tone-aware responses (empathy for distressed users)
│   - ✅ Source-of-truth discipline (DB over policy text for personal data)
│   - ✅ Excel export + chart generation for large results
│   - ✅ DAB OData filter validation (read-only by design)
│   - ✅ Caching of tools, schema, distinct values, and schema index at startup
│   - ✅ DRY consolidation: `agent/dab/` centralizes response extraction, metrics, validation, normalization
│   - ✅ Semantic schema index for field discovery without external vector DB
│   - ✅ OpenTelemetry metrics with graceful null fallback
│   - ✅ 3-stage response structure (Inform → Assist → Offer feedback)
│   - ✅ Client-side dynamic binning for multi-tenant schemas
│   - ✅ Chart artifact injection into LLM context (industry standard)
│   - ✅ **Session state management** for follow-up actions (export, drill-down)
│   - ✅ **Code resolver** for zero-touch multi-tenant code translation
│   - ✅ **Ambiguous response handling** prevents guessing on "yes"/"ok"
│   - ✅ **Finance intent detection** — config-driven keywords + dynamic HANA schema discovery at startup
│   - ✅ **Finance table discovery** — queries `hana_list_tables` at startup, caches with TTL, falls back to YAML config
│   - ✅ **Finance keyword narrowing** — removed broad terms (`company code`, `fiscal year`, `payroll`, `depreciation`) that caused false positives
│   - ✅ **Monolithic refactoring** — extracted `agent/prompts.py`, `agent/binning.py`, `agent/session_state.py` from `tool_planner.py` and `agentic_executor.py`
│   - ✅ **Finance intent consolidation** — removed duplicate `_HANA_FINANCE_TABLES` and `_FINANCE_KEYWORDS` from executor, consolidated into intent_classifier.py
│   
│   ### Areas for Improvement
│   
│   - ⚠️ No retry logic for DAB server startup (if DB is down at startup, `discover_tools` fails silently)
│   - ⚠️ `GeminiRateLimiter` and `GeminiGenAIEmbedder` have duplicated rate limiting logic — could be unified into `agent/dab/`
│   - ⚠️ `tenant_mappings.yaml` `default_role: null` means no access for unmapped groups — consider a read-only fallback
│   - ⚠️ `CHART_OUTPUT_DIR` default `/mnt/agents/output` was changed to `./agent/output` — verify in all environments
│   - ⚠️ Session state is in-memory only — not persisted across restarts
│   - ⚠️ Code resolver uses cursor-based pagination but `page_size` variable is referenced before assignment in `_extract_items`
│   - ⚠️ Schema token budget (`TOKEN_BUDGET = 1500`) is conservative for Gemini 3.1 Flash Lite's 1M context; consider raising to 4,000–6,000 tokens to reduce over-pruning
│   
│   ---
│   
│   *End of Architecture Document*