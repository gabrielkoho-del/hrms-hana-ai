"""
intent_classifier.py
Lightweight two-stage LLM-based intent & tone classification.

Industry standard:
 - Single structured call per stage (~100-200 tokens each).
 - Embedding-based exemplar retrieval for scalable intent matching.
 - Config-driven intents, categories, keywords (config/intents.yaml).
 - Per-intent precision/recall eval with CI gating hooks.

Stage 1 (coarse router): narrow broad category (10 options).
Stage 2 (fine classifier): specific intent + tone fields within that category.

Includes intent_category for robust, scalable 3-stage response handling.
Added ambiguous affirmation detection when prior turn offered multiple options.
"""
import asyncio
import json
import os
import re
import time
import logging
import threading
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache

import yaml
import numpy as np

from agent.core.intent_config import (
    get_intents_config,
    get_categories,
    get_intents,
    get_keywords,
    get_keyword_groups,
    get_full_info_patterns,
    get_intent_category,
    get_category_config,
    get_exemplars,
    get_all_intent_names,
    get_all_category_names,
    get_finance_keywords,
    get_aggregate_keywords,
    get_individual_keywords,
    get_forecast_keywords,
    get_export_keywords,
    get_export_offer_keywords,
    get_multi_option_keywords,
    get_affirmative_keywords,
)
from agent.integrations.llm_client import call_llm
from agent.integrations.schema_index import EMBEDDING_MODEL, EMBEDDING_DIM, _get_embed_client

logger = logging.getLogger("hr_agent")


# =======================================================================
# FINANCE TABLE CONFIG -- still loaded from finance_config.yaml

_INTENT_EXEMPLAR_CACHE_DIR = Path(os.getenv("INTENT_EXEMPLAR_CACHE_DIR", "data/intent_exemplar_cache"))
# =======================================================================

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_INTENT_EXEMPLAR_CACHE_DIR = Path(os.getenv('INTENT_EXEMPLAR_CACHE_DIR', 'data/intent_exemplar_cache'))
_FINANCE_CONFIG_PATH = os.path.join(_BASE_DIR, "config", "finance_config.yaml")

def _intent_exemplar_hash(exemplars):
    "Stable hash for exemplar dict."
    def _stable_dump(obj):
        if isinstance(obj, dict):
            return {k: _stable_dump(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            return sorted((_stable_dump(item) for item in obj), key=str)
        return obj
    return hashlib.md5(json.dumps(_stable_dump(exemplars), default=str).encode()).hexdigest()[:16]


def _load_finance_config() -> Dict:
    """Load finance table configuration from YAML file."""
    if not os.path.isfile(_FINANCE_CONFIG_PATH):
        logger.warning("Finance config not found at %s, using defaults", _FINANCE_CONFIG_PATH)
        return {}
    try:
        with open(_FINANCE_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config if isinstance(config, dict) else {}
    except Exception as e:
        logger.warning("Failed to load finance config from %s: %s", _FINANCE_CONFIG_PATH, e)
        return {}


_FINANCE_CONFIG = _load_finance_config()

# Table discovery config (keywords moved to intents.yaml)
_FINANCE_TABLE_PREFIXES: List[str] = _FINANCE_CONFIG.get("finance_table_prefixes", [
    "FAGL", "BKPF", "BSEG", "SKA", "CSK", "T001", "TCUR",
])

_FINANCE_FALLBACK_TABLES: Set[str] = set(_FINANCE_CONFIG.get("finance_table_names", [
    "FAGLFLEXA", "BKPF", "BSEG", "SKA1", "SKAT",
    "CSKS", "CSKT", "T001", "TCURC", "TCURR",
]))

_SCHEMA_CACHE_TTL = _FINANCE_CONFIG.get("schema_cache_ttl_seconds", 3600)
_schema_discovery_tool = _FINANCE_CONFIG.get("schema_discovery_tool", "hana_list_tables")


# =======================================================================
# DATA CLASSES
# =======================================================================

@dataclass
class IntentResult:
    intent: str                      # e.g., "leave_request", "policy_question", "finance_budget"
    intent_category: str             # "personal_data" | "aggregate_data" | "policy_info" | ...
    data_scope: str                  # "individual" | "aggregate" | "none"
    chart_eligible: bool             # True ONLY if aggregate data and not personal/action/emergery
    urgency_level: str               # "routine" | "time_sensitive" | "urgent" | "distressed"
    emotional_state: str             # "neutral" | "anxious" | "frustrated" | "celebratory" | ...
    topic_sensitivity: str           # "low" | "medium" | "high"
    needs_empathy: bool
    confidence: float                # 0.0-1.0
    action_oriented: bool            # True if user wants to DO something
    wants_export: bool = False       # True if user explicitly asks to export/download data
    is_ambiguous: bool = False       # True if user's affirmation is ambiguous (multiple prior options)
    finance_query: bool = False      # True if query involves SAP FI/CO finance data
    forecasting_query: bool = False  # True if query asks for forecasting/prediction/trend


# =======================================================================
# EXEMPLAR INDEX -- embedding-based intent retrieval with Jaccard fallback
# =======================================================================

class IntentExemplarIndex:
    """Lightweight index for embedding-based exemplar retrieval.

    Primary path: embed exemplar texts and query via Gemini, cosine similarity.
    Fallback: Jaccard token overlap if embedding API is unavailable.
    """

    def __init__(self, exemplars: Dict[str, List[str]]):
        self.exemplars = exemplars
        self._intent_embeddings: Dict[str, np.ndarray] = {}
        self._intent_token_sets: Dict[str, List[Set[str]]] = {}
        self._all_intents: List[str] = []
        self._all_embeddings: Optional[np.ndarray] = None
        self.embedding_dim: int = 768
        self.embedding_provider: str = "gemini"
        self._built = False
        self._use_embeddings = True

    async def build(self) -> None:
        """Build index. Tries embeddings first, falls back to Jaccard."""
        if self._built:
            return
        # Try disk cache first
        if await self.load_cache(self.exemplars):
            return
        try:
            await self._build_with_embeddings()
        except Exception as e:
            logger.warning("Embedding exemplar build failed (%s). Using Jaccard fallback.", e)
            self._use_embeddings = False
            self._build_jaccard_fallback()
        # Persist to disk cache
        if self._use_embeddings:
            self._save_cache(self.exemplars)
        self._built = True

    async def _build_with_embeddings(self) -> None:
        all_texts: List[str] = []
        intent_ranges: List[Tuple[str, int, int]] = []

        for intent, texts in self.exemplars.items():
            start = len(all_texts)
            all_texts.extend(texts)
            intent_ranges.append((intent, start, len(texts)))

        if not all_texts:
            self._all_intents = []
            self._all_embeddings = np.zeros((0, 768), dtype=np.float32)
            return

        # Batch embed all exemplar texts in as few API calls as possible.
        from agent.integrations.schema_index import _get_embed_client
        client = _get_embed_client()
        vectors = await client.embed(all_texts)

        self._all_intents = list(self.exemplars.keys())
        all_vecs: List[List[float]] = []
        for intent, start, count in intent_ranges:
            self._intent_embeddings[intent] = np.array(vectors[start:start + count], dtype=np.float32)
            all_vecs.extend(vectors[start:start + count])

        self._all_embeddings = np.array(all_vecs, dtype=np.float32) if all_vecs else np.zeros((0, len(vectors[0]) if vectors else 768))
        self.embedding_dim = int(self._all_embeddings.shape[1]) if len(self._all_embeddings) > 0 else 768
        # Record provider and lock the fallback client so query embeddings stay
        # dimension-aligned with this index (prevents Gemini/Cohere dim mismatch).
        self.embedding_provider = client.get_active_provider()
        if hasattr(client, "lock_provider"):
            client.lock_provider(self.embedding_provider)

    def _cache_path(self, exemplars: Dict[str, List[str]]) -> Path:
        _INTENT_EXEMPLAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        schema_hash = _intent_exemplar_hash(exemplars)
        provider_tag = getattr(self, "embedding_provider", "gemini")
        model_tag = EMBEDDING_MODEL.replace('/', '_').replace(':', '_')
        return _INTENT_EXEMPLAR_CACHE_DIR / ('intent_' + schema_hash + '_' + provider_tag + '_' + model_tag + '.npz')

    def _save_cache(self, exemplars: Dict[str, List[str]]) -> None:
        if self._all_embeddings is None:
            return
        path = self._cache_path(exemplars)
        model_tag = EMBEDDING_MODEL.replace('/', '_').replace(':', '_')
        provider_tag = getattr(self, "embedding_provider", "gemini")
        prefix = 'intent_'
        # Current format: intent_<hash>_<provider>_<model>.npz
        for f in _INTENT_EXEMPLAR_CACHE_DIR.glob(prefix + '*' + '_' + provider_tag + '_' + model_tag + '.npz'):
            if f != path and f.is_file():
                try:
                    f.unlink()
                except OSError:
                    pass
        # Old format with dim suffix: intent_<hash>_<model>_<dim>.npz
        for f in _INTENT_EXEMPLAR_CACHE_DIR.glob(prefix + '*' + '_' + model_tag + '_' + '[0-9]*.npz'):
            if f != path and f.is_file():
                try:
                    f.unlink()
                except OSError:
                    pass
        # Legacy format without model tag: intent_<hash>.npz
        for f in _INTENT_EXEMPLAR_CACHE_DIR.glob(prefix + '*.npz'):
            if f == path:
                continue
            fname = f.name
            if model_tag not in fname:
                try:
                    f.unlink()
                except OSError:
                    pass
        np.savez_compressed(
            path,
            embeddings=self._all_embeddings,
            intents=json.dumps(self._all_intents),
            intent_embeddings={k: v.tolist() for k, v in self._intent_embeddings.items()},
            exemplars=json.dumps(exemplars),
            built_at=time.time(),
            embedding_model=EMBEDDING_MODEL,
            embedding_dim=self.embedding_dim,
            embedding_provider=provider_tag,
        )
        logger.info('IntentExemplarIndex: cached to %s (dim=%d, provider=%s)', path, self.embedding_dim, provider_tag)

    async def load_cache(self, exemplars: Dict[str, List[str]]) -> bool:
        client = _get_embed_client()
        active_provider = client.get_active_provider()
        # Stamp the active provider before computing the cache path so the path
        # tag matches the provider that will actually be used to (re)build.
        self.embedding_provider = active_provider
        path = self._cache_path(exemplars)
        if not path.exists():
            return False
        try:
            data = np.load(path, allow_pickle=False)
            cached_dim = int(data['embedding_dim'])
            if cached_dim != EMBEDDING_DIM:
                logger.warning('Intent cache dim mismatch: cached=%d, configured=%d. Rebuilding.', cached_dim, EMBEDDING_DIM)
                return False
            # Provider validation: only reuse a cache built by the active provider.
            cached_provider = str(data.get('embedding_provider', 'gemini'))
            if cached_provider != active_provider:
                logger.warning('Intent cache provider mismatch: cached=%s, active=%s. Rebuilding.', cached_provider, active_provider)
                return False
            if int(data['embeddings'].shape[0]) != len(json.loads(str(data['intents']))):
                return False
            self._all_embeddings = data['embeddings']
            self._all_intents = json.loads(str(data['intents']))
            intent_embeddings_dict = json.loads(str(data['intent_embeddings']))
            self._intent_embeddings = {k: np.array(v, dtype=np.float32) for k, v in intent_embeddings_dict.items()}
            self.embedding_dim = cached_dim
            self.embedding_provider = cached_provider
            self._built = True
            self._use_embeddings = True
            if hasattr(client, 'lock_provider'):
                client.lock_provider(cached_provider)
            logger.info('IntentExemplarIndex: loaded %d intents from cache', len(self._all_intents))
            return True
        except Exception as e:
            logger.debug('IntentExemplarIndex: cache load failed: %s', e)
            return False

    def _build_jaccard_fallback(self) -> None:
        self._all_intents = list(self.exemplars.keys())
        for intent, texts in self.exemplars.items():
            self._intent_token_sets[intent] = [set(t.lower().split()) for t in texts]

    async def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return top-k (intent, score) pairs for the query."""
        if not self._built:
            await self.build()
        if not self._all_intents:
            return []

        if self._use_embeddings and self._all_embeddings is not None and len(self._all_embeddings) > 0:
            return await self._search_embedding(query, top_k)
        return self._search_jaccard(query, top_k)

    async def _search_embedding(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        from agent.integrations.schema_index import _get_embed_client
        client = _get_embed_client()
        query_vec = np.array((await client.embed([query]))[0], dtype=np.float32)

        if self._all_embeddings is None or len(self._all_embeddings) == 0:
            return []

        norms = np.linalg.norm(self._all_embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        cosine_scores = np.dot(self._all_embeddings, query_vec) / (norms * query_norm + 1e-8)

        intent_scores: List[Tuple[str, float]] = []
        idx = 0
        for intent in self._all_intents:
            count = len(self.exemplars.get(intent, []))
            if count == 0:
                intent_scores.append((intent, 0.0))
                continue
            intent_vecs = cosine_scores[idx:idx + count]
            best_score = float(np.max(intent_vecs))
            intent_scores.append((intent, best_score))
            idx += count

        intent_scores.sort(key=lambda x: x[1], reverse=True)
        return intent_scores[:top_k]

    def _search_jaccard(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        query_tokens = set(query.lower().split())
        intent_scores: List[Tuple[str, float]] = []
        for intent in self._all_intents:
            best = 0.0
            for token_set in self._intent_token_sets.get(intent, []):
                intersection = len(query_tokens & token_set)
                union = len(query_tokens | token_set)
                jaccard = intersection / union if union > 0 else 0.0
                best = max(best, jaccard)
            intent_scores.append((intent, best))
        intent_scores.sort(key=lambda x: x[1], reverse=True)
        return intent_scores[:top_k]


# -- Module-level exemplar index singleton ------------------------------

_intent_exemplar_index: Optional[IntentExemplarIndex] = None
_intent_exemplar_lock = threading.Lock()


async def get_intent_exemplar_index() -> Optional[IntentExemplarIndex]:
    """Return the global exemplar index, building it lazily on first call."""
    global _intent_exemplar_index
    if _intent_exemplar_index is not None:
        return _intent_exemplar_index
    with _intent_exemplar_lock:
        if _intent_exemplar_index is None:
            try:
                exemplars = get_exemplars()
                _intent_exemplar_index = IntentExemplarIndex(exemplars)
                await _intent_exemplar_index.build()
                logger.info("IntentExemplarIndex: built with %d intents", len(exemplars))
            except Exception as e:
                logger.warning("IntentExemplarIndex build failed: %s", e)
                _intent_exemplar_index = None
    return _intent_exemplar_index


# =======================================================================
# PROMPT BUILDERS -- coarse router + fine classifier
# =======================================================================

def _build_coarse_prompt(user_query: str, conversation_history: str = "") -> Tuple[str, str]:
    """Build (system_prompt, user_prompt) for stage 1: broad category classification."""
    categories = get_categories()
    cat_descriptions = "\n".join(
        f"- {name}: {cat.get('description', '')}"
        for name, cat in categories.items()
    )

    system_prompt = f"""You are an intent router for an HR AI assistant.
Classify the user's query into ONE of these broad categories:

{cat_descriptions}

Rules:
- Finance/admin queries (general ledger, cost center, budget, currency, exchange rate, invoice, vendor, payroll, accounts payable/receivable, profitability, cash flow, balance sheet, financial statements, forecast, FI/CO reporting) are aggregate_data, NOT data_discovery.
- Workforce productivity metrics (revenue per employee, absenteeism rate, overtime cost) are workforce_analytics.
- Dashboard creation requests are dashboard_building.
- Questions asking for specific counts, totals, sums, averages, or actual data values from known tables/entities are aggregate_data, not data_discovery.
- Data_discovery is ONLY when the user asks "what data do you have?", "what tables exist?", "what schemas are there?" -- questions about system capability, not data retrieval."""

    history_snippet = conversation_history[:200] if conversation_history else "None"
    user_prompt = f"""Recent context: "{history_snippet}"

Query: "{user_query}"

Return JSON only: {{"category": "<one of the above>", "confidence": 0.0-1.0}}"""

    return system_prompt, user_prompt


def _build_fine_prompt(
    user_query: str,
    category: str,
    exemplar_context: str,
    conversation_history: str = "",
) -> Tuple[str, str]:
    """Build (system_prompt, user_prompt) for stage 2: specific intent classification."""
    intents = get_intents()
    category_intents = [
        (name, defn)
        for name, defn in intents.items()
        if defn.get("category") == category
    ]

    intent_list = "\n".join(
        f"- {name}: {defn.get('description', '')}"
        for name, defn in category_intents
    )

    system_prompt = f"""You are an intent classifier for an HR AI assistant.
The broad category for this query is: {category}

Classify the user's query into one of the specific intents listed below.
Also determine urgency, emotional state, topic sensitivity, empathy need, confidence, and action orientation.

Rules:
- urgency_level: routine | time_sensitive | urgent | distressed
  - "distressed" = user mentions death, accident, hospital, severe illness, panic, "don't know what to do"
  - "urgent" = user needs something today/now, mentions deadlines, "emergency leave"
  - "time_sensitive" = needs action within days, mentions specific dates
  - "routine" = general questions, lookups, no time pressure
- topic_sensitivity: low | medium | high
  - "high" = medical, mental health, family crisis, harassment, termination, resignation
  - "medium" = leave disputes, salary issues, performance concerns, benefits
  - "low" = directory lookups, general policy questions, org chart queries, training info
- needs_empathy: true if user expresses distress, anxiety, urgency about personal matters
- action_oriented: true if user wants to TAKE ACTION (apply, request, change, escalate, submit, book, cancel, update, resign, enroll)
  - false if they just want to KNOW (check, see, find out, what is, how many, who is)"""

    history_snippet = ""
    if conversation_history and len(user_query.strip().split()) <= 5:
        history_snippet = f'Recent context (for follow-up resolution): "{conversation_history[:200]}"'

    user_prompt = f"""Possible intents in this category:
{intent_list}

Examples of similar queries and their intents:
{exemplar_context}

{history_snippet}

Query: "{user_query}"

Return JSON only with keys: intent, urgency_level, emotional_state, topic_sensitivity, needs_empathy, confidence, action_oriented"""

    return system_prompt, user_prompt


# =======================================================================
# CLASSIFIER STAGES
# =======================================================================

def _classify_coarse(user_query: str, conversation_history: str = "") -> Tuple[str, float]:
    """Stage 1: classify into a broad category. Returns (category, confidence)."""
    system_prompt, user_prompt = _build_coarse_prompt(user_query, conversation_history)
    try:
        choice = call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=100)
        if not choice:
            return "policy_info", 0.3
        content = choice.get("message", {}).get("content", "{}")
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        result = json.loads(content)
        category = result.get("category", "policy_info")
        confidence = float(result.get("confidence", 0.5))
        # Validate category
        if category not in get_all_category_names():
            logger.warning("Coarse classifier returned unknown category: %s", category)
            category = "policy_info"
        return category, confidence
    except Exception as e:
        logger.warning("Coarse classification failed: %s", e)
        return "policy_info", 0.0


def _classify_fine(user_query: str, category: str, exemplar_context: str, conversation_history: str = "") -> Dict[str, Any]:
    """Stage 2: classify into a specific intent within the category. Returns dict of tone fields."""
    system_prompt, user_prompt = _build_fine_prompt(user_query, category, exemplar_context, conversation_history)
    try:
        choice = call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=200)
        if not choice:
            return {"intent": "general_hr", "confidence": 0.3}
        content = choice.get("message", {}).get("content", "{}")
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        result = json.loads(content)
        intent = result.get("intent", "general_hr")
        # Validate intent belongs to the category
        intent_cat = get_intent_category(intent)
        if intent_cat != category:
            # Fallback: find the best matching intent in this category
            category_intents = [
                name for name, defn in get_intents().items()
                if defn.get("category") == category
            ]
            if category_intents:
                intent = category_intents[0]
            else:
                intent = "general_hr"
        return result
    except Exception as e:
        logger.warning("Fine classification failed: %s", e)
        category_intents = [
            name for name, defn in get_intents().items()
            if defn.get("category") == category
        ]
        return {"intent": category_intents[0] if category_intents else "general_hr", "confidence": 0.0}


async def _get_exemplar_context(user_query: str, category: str, top_k: int = 3) -> str:
    """Get exemplar context for the fine classifier prompt."""
    index = await get_intent_exemplar_index()
    if index is None:
        return "No exemplars available."

    try:
        results = await index.search(user_query, top_k=top_k)
    except Exception as e:
        logger.debug("Exemplar search failed: %s", e)
        return "No exemplars available."

    if not results:
        return "No similar exemplars found."

    lines = []
    for intent, score in results:
        exemplars = get_exemplars(intent_name=intent).get(intent, [])
        for ex in exemplars[:2]:
            lines.append(f"- [{intent}] {ex}")
    return "\n".join(lines) if lines else "No similar exemplars found."


# =======================================================================
# KEYWORD / PATTERN HELPERS (config-driven, local fallback)
# =======================================================================

_FULL_INFO_PATTERNS = [re.compile(p) for p in get_full_info_patterns()]


def _match_patterns(query: str, patterns: List[re.Pattern]) -> bool:
    q = query.strip()
    return any(p.search(q) for p in patterns)


def _is_short_affirmative(query: str) -> bool:
    """Detect very short affirmative responses (1-3 words)."""
    q = query.strip().lower().rstrip("!?.,")
    words = q.split()
    if len(words) > 3:
        return False
    affirmative = get_affirmative_keywords()
    return any(w in affirmative for w in words) or q in affirmative


def _conversation_has_export_offer(history: str) -> bool:
    if not history:
        return False
    export_offer = get_export_offer_keywords()
    return any(kw in history.lower() for kw in export_offer)


def _conversation_has_multi_option_offer(history: str) -> bool:
    if not history:
        return False
    multi_option = get_multi_option_keywords()
    return any(kw in history.lower() for kw in multi_option)


def _detect_forecasting_query(query: str) -> bool:
    q = query.lower()
    forecast_keywords = get_forecast_keywords()
    return any(kw in q for kw in forecast_keywords)


def _detect_export_intent(query: str) -> bool:
    q = query.lower()
    export_keywords = get_export_keywords()
    return any(kw in q for kw in export_keywords)


# =======================================================================
# FINANCE TABLE DISCOVERY (unchanged from legacy, uses finance_config.yaml)
# =======================================================================

_discovered_finance_tables: Optional[Set[str]] = None
_discovered_tables_timestamp: float = 0.0


def initialize_finance_tables(hana_client: Any = None) -> None:
    """Discover finance tables from HANA schema at startup."""
    global _discovered_finance_tables, _discovered_tables_timestamp

    if hana_client is None:
        _discovered_finance_tables = None
        _discovered_tables_timestamp = 0.0
        return

    try:
        result = hana_client.call_tool(_schema_discovery_tool, {})
        tables = set()
        if isinstance(result, dict) and "rows" in result:
            for row in result.get("rows", []):
                if isinstance(row, dict):
                    table_name = row.get("TABLE_NAME", row.get("table_name", ""))
                    if table_name:
                        tables.add(table_name)
                elif isinstance(row, str):
                    tables.add(row)
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, str):
                    tables.add(item)

        discovered = set()
        for table in tables:
            table_upper = table.upper()
            for prefix in _FINANCE_TABLE_PREFIXES:
                if table_upper.startswith(prefix.upper()):
                    discovered.add(table_upper)
                    break

        _discovered_finance_tables = discovered if discovered else None
        _discovered_tables_timestamp = time.time()
        logger.info("Finance table discovery: found %d finance tables from HANA schema", len(discovered or set()))
        _validate_finance_table_coverage(discovered)
    except Exception as e:
        logger.warning("Finance table discovery failed: %s. Using fallback tables.", e)
        _discovered_finance_tables = None
        _discovered_tables_timestamp = time.time()


def _validate_finance_table_coverage(discovered: Set[str]) -> None:
    """Warn at startup when configured finance tables are absent from the live registry."""
    if not discovered:
        return
    fallback = set(_FINANCE_FALLBACK_TABLES)
    missing = fallback - discovered
    if missing:
        logger.warning(
            "Finance config references %d tables not found in HANA registry: %s. "
            "Check config/finance_config.yaml or schema discovery.",
            len(missing), ", ".join(sorted(missing))
        )
    extra = discovered - fallback
    if extra:
        logger.info(
            "Finance discovery found %d tables not in config fallback: %s. "
            "Consider updating config/finance_config.yaml.",
            len(extra), ", ".join(sorted(extra))
        )


def _get_finance_tables() -> Set[str]:
    """Return discovered finance tables (cached) or fallback tables."""
    global _discovered_finance_tables, _discovered_tables_timestamp
    if _discovered_finance_tables is not None:
        cache_age = time.time() - _discovered_tables_timestamp
        if cache_age < _SCHEMA_CACHE_TTL:
            return _discovered_finance_tables
        logger.info("Finance table cache expired (%.0fs), re-discovery recommended", cache_age)
    return _FINANCE_FALLBACK_TABLES


def _detect_finance_query(query: str) -> bool:
    """Detect if query involves SAP FI/CO finance data (single config-driven source)."""
    q = query.lower()
    finance_tables = _get_finance_tables()
    for table in finance_tables:
        if table.lower() in q:
            return True
    finance_keywords = get_finance_keywords()
    return any(kw in q for kw in finance_keywords)


# =======================================================================
# MAIN CLASSIFIER -- two-stage with embedding exemplar retrieval
# =======================================================================

async def classify_intent(user_query: str, conversation_history: str = "",
                    pending_offers: tuple = ()) -> IntentResult:
    """
    Classify user intent and tone. Two-stage LLM approach:
      Stage 1: coarse router (category)
      Stage 2: fine classifier (specific intent + tone fields)

    Embedding exemplar retrieval augments the fine classifier prompt.
    Falls back gracefully if embeddings or LLM are unavailable.
    """
    # Initialize is_ambiguous early so it's always defined
    is_ambiguous = bool(pending_offers and _is_short_affirmative(user_query))

    try:
        # -- Stage 1: Coarse router --------------------------------------
        category, coarse_confidence = _classify_coarse(user_query, conversation_history)
        logger.debug("Coarse router: category=%s confidence=%.2f", category, coarse_confidence)

        # -- Stage 2: Fine classifier (or direct mapping for single-intent categories) --
        category_intents = [
            name for name, defn in get_intents().items()
            if defn.get("category") == category
        ]

        if len(category_intents) == 1:
            # Optimization: skip fine LLM call for single-intent categories
            intent = category_intents[0]
            fine_result = {"intent": intent, "confidence": coarse_confidence}
        else:
            exemplar_context = await _get_exemplar_context(user_query, category)
            fine_result = _classify_fine(user_query, category, exemplar_context, conversation_history)

        intent = fine_result.get("intent", "general_hr")
        confidence = float(fine_result.get("confidence", coarse_confidence))

        # -- Post-processing from category config ------------------------
        category_config = get_category_config(category)
        data_scope = category_config.get("data_scope", "none")
        chart_eligible = category_config.get("chart_eligible", False)

        # Tone fields with safe defaults
        urgency_level = fine_result.get("urgency_level", category_config.get("urgency_default", "routine"))
        emotional_state = fine_result.get("emotional_state", "neutral")
        topic_sensitivity = fine_result.get("topic_sensitivity", "low")
        needs_empathy = bool(fine_result.get("needs_empathy", False))
        action_oriented = bool(fine_result.get("action_oriented", False))

        # -- Local keyword fallbacks -------------------------------------
        wants_export = _detect_export_intent(user_query)
        if not wants_export and _match_patterns(user_query, _FULL_INFO_PATTERNS):
            wants_export = True
            logger.info("FULL_INFO_DETECTED: implicit export from query '%s'", user_query)

        # If personal/action/emergency and user asks for full info, export is implied
        if not wants_export and category in ("personal_data", "action_request", "emergency", "grievance"):
            if _match_patterns(user_query, _FULL_INFO_PATTERNS):
                wants_export = True
                logger.info("FULL_INFO_DETECTED: implicit export from query '%s'", user_query)

        finance_query = _detect_finance_query(user_query) or category in (
            "finance_gl_analysis", "finance_cost_analysis", "finance_currency",
            "finance_budget", "finance_revenue_analysis", "finance_profitability",
            "finance_cashflow", "finance_balance_sheet", "finance_financial_statement",
            "finance_cost_center", "finance_payroll_analysis",
            "finance_accounts_payable", "finance_accounts_receivable",
            "finance_invoice_analysis", "finance_vendor_analysis",
            "finance_budget_variance", "finance_forecast",
        )
        forecasting_query = _detect_forecasting_query(query=user_query) or intent in ("finance_forecast", "forecasting_query")

        # -- Confidence clamp --------------------------------------------
        confidence = max(0.0, min(1.0, float(confidence)))

        return IntentResult(
            intent=intent,
            intent_category=category,
            data_scope=data_scope,
            chart_eligible=chart_eligible,
            urgency_level=urgency_level,
            emotional_state=emotional_state,
            topic_sensitivity=topic_sensitivity,
            needs_empathy=needs_empathy,
            confidence=confidence,
            action_oriented=action_oriented,
            wants_export=wants_export,
            is_ambiguous=is_ambiguous,
            finance_query=finance_query,
            forecasting_query=forecasting_query,
        )
    except Exception as e:
        logger.warning("Intent classification failed: %s. Falling back to neutral.", e)
        return IntentResult(
            intent="general_hr",
            intent_category="policy_info",
            data_scope="none",
            chart_eligible=False,
            urgency_level="routine",
            emotional_state="neutral",
            topic_sensitivity="low",
            needs_empathy=False,
            confidence=0.0,
            action_oriented=False,
            wants_export=False,
            is_ambiguous=is_ambiguous,
            finance_query=False,
            forecasting_query=False,
        )
