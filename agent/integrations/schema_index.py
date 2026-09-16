"""
agent/integrations/schema_index.py

Schema field embedding index for semantic field retrieval.

The heavy lifting (embedding clients, cache lifecycle, base index class)
lives in `agent.integrations.embedding`. This module only defines the
schema-specific bits:
  - Field text construction (name + entity + type + description + examples)
  - Hybrid search (cosine + keyword overlap)
  - The public `search_relevant_fields` API used by `agent/core/prompts.py`

Backwards-compatible re-exports:
  This module still re-exports `EMBEDDING_MODEL`, `EMBEDDING_DIM`,
  `SCHEMA_INDEX_CACHE_DIR`, `embed_text_sync`, and the lower-level
  `_get_embed_client` / `_embed_texts_with_provider` helpers so that
  `agent.core.intent_classifier` and other callers continue to work
  without any import changes.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agent.integrations.embedding import (
    BaseEmbeddingIndex,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    embed_text_sync as _embed_text_sync,
    get_embed_client as _get_embed_client_impl,
    _embed_texts_with_provider,
    get_query_embedding,
    stable_hash as _stable_hash,
)

logger = logging.getLogger("hr_agent")

# Keep the legacy name for any external imports.
SCHEMA_INDEX_CACHE_DIR = Path(os.getenv("SCHEMA_CACHE_DIR", "data/schema_cache"))


# ── Backwards-compatible re-exports ─────────────────────────────────────
# `intent_classifier.py` and others import these names from this module.
# Keep them as aliases to the canonical implementations in `embedding.py`.
def _get_embed_client():
    """Backwards-compat alias for `get_embed_client` from embedding module."""
    return _get_embed_client_impl()


# `intent_classifier.py` does:
#   from agent.integrations.schema_index import _get_embed_client
# We re-export the implementation under that name.
get_embed_client = _get_embed_client_impl
embed_text_sync = _embed_text_sync


# ── SchemaFieldIndex ────────────────────────────────────────────────────
class SchemaFieldIndex(BaseEmbeddingIndex):
    """In-memory embedding index for DAB schema fields with hybrid keyword fallback.

    Inherits cache + build + load lifecycle from `BaseEmbeddingIndex`.
    The schema-specific behavior is:
      - `_texts_to_embed()` builds rich field text (name, type, description, examples)
      - `search()` does hybrid cosine + keyword overlap scoring

    Provider policy:
      Schema index is pinned to Cohere (dim=1024) to avoid Gemini free-tier
      rate limits at query time. Change `force_provider` to switch back.
    """

    cache_prefix = "schema"
    cache_dir = SCHEMA_INDEX_CACHE_DIR
    # Force Cohere for schema index so query-time embeddings always match the
    # cached index dimension. Gemini free-tier rate limits (1000 req/day) make
    # it unreliable for per-query embedding. Set to "gemini" to switch back.
    force_provider: str = "cohere"

    def __init__(self, schema: Dict, tenant_id: str = "default"):
        super().__init__(tenant_id=tenant_id)
        self.schema = schema
        self.field_keys: List[str] = []
        self.field_meta: List[Dict] = []

    # ── BaseEmbeddingIndex abstract impl ───────────────────────────────
    def _content_hash(self) -> str:
        return _stable_hash(self.schema)

    def _build_field_text(self, entity_name: str, field: Dict) -> str:
        """Rich text for embedding: combines name, description, type, examples."""
        parts = [f"Field name: {field.get('name', '')}"]
        parts.append(f"Entity: {entity_name}")
        parts.append(f"Data type: {field.get('type', 'unknown')}")

        if field.get("description"):
            parts.append(f"Description: {field['description']}")
        if field.get("distinct_values"):
            examples = ", ".join(str(v) for v in field["distinct_values"][:5])
            parts.append(f"Example values: {examples}")

        return " | ".join(parts)

    def _texts_to_embed(self) -> List[str]:
        texts: List[str] = []
        self.field_keys = []
        self.field_meta = []
        for entity_name, entity in self.schema.items():
            if not isinstance(entity, dict):
                continue
            fields = entity.get("fields", entity.get("columns", []))
            for field in fields:
                if not isinstance(field, dict):
                    continue
                key = f"{entity_name}.{field.get('name', '')}"
                self.field_keys.append(key)
                self.field_meta.append({
                    "entity": entity_name,
                    "field": field,
                    "key": key,
                })
                texts.append(self._build_field_text(entity_name, field))
        return texts

    def _extra_cache_fields(self) -> Dict[str, Any]:
        return {
            "keys": json.dumps(self.field_keys),
            "meta": json.dumps(self.field_meta),
        }

    def _restore_from_cache(self, data: Any) -> bool:
        self.field_keys = json.loads(str(data["keys"]))
        self.field_meta = json.loads(str(data["meta"]))
        self.embeddings = data["embeddings"]
        if self.embeddings.shape[0] != len(self.field_keys):
            return False
        return True

    def _find_existing_cache(self) -> Optional[Path]:
        """Find an existing cache file built by the forced provider only.

        Old caches built with a different provider (e.g. Gemini) are ignored
        so the index is rebuilt with the current forced provider (Cohere).
        """
        prefix = (
            f"{self.cache_prefix}_{self.tenant_id}_{self._content_hash()}_"
            f"{self.force_provider}_"
        )
        suffix = f"_{self._model_tag()}.npz"
        candidates = [
            p for p in self.cache_dir.glob(prefix + "*.npz")
            if p.name.endswith(suffix)
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    async def build(self) -> "SchemaFieldIndex":
        """Build index by embedding all fields with the forced provider."""
        texts = self._texts_to_embed()
        if not texts:
            logger.warning("SchemaFieldIndex: no fields to index")
            return self

        logger.info(
            "SchemaFieldIndex: embedding %d fields via %s...",
            len(texts), self.force_provider,
        )
        vectors = await _embed_texts_with_provider(texts, self.force_provider)
        self.embeddings = np.array(vectors, dtype=np.float32)

        if self.embeddings.shape[0] != len(texts):
            raise RuntimeError(
                f"SchemaFieldIndex: embedding count mismatch: expected "
                f"{len(texts)}, got {self.embeddings.shape[0]}"
            )
        self.embedding_dim = int(self.embeddings.shape[1])
        self.embedding_provider = self.force_provider
        self._built_at = time.time()
        self._save_cache()
        return self

    # ── Search ─────────────────────────────────────────────────────────
    def search(self, query: str, query_embedding: List[float], top_k: int = 8) -> List[Dict]:
        """Hybrid search: embedding similarity + keyword overlap."""
        if self.embeddings is None or len(self.field_keys) == 0:
            return []

        n_fields = len(self.field_keys)

        # ── Shape validation ─────────────────────────────────────────
        if self.embeddings.shape[0] != n_fields:
            logger.error(
                "Search shape mismatch: embeddings.shape[0]=%d, field_keys=%d",
                self.embeddings.shape[0], n_fields,
            )
            return []

        k = min(top_k, n_fields)

        query_vec = np.array(query_embedding, dtype=np.float32)

        # ── Dimension alignment check ────────────────────────────────
        # Schema vectors and the query vector MUST share a dimension. A mismatch
        # means the query embedding used a different model than the schema build.
        # Fail loud (return no semantic results) rather than silently degrading,
        # so the underlying model misconfiguration is surfaced immediately.
        if query_vec.shape[0] != self.embedding_dim:
            logger.error(
                "Embedding retrieval failed: query dim=%d != index dim=%d. "
                "Query and schema embeddings use different models.",
                query_vec.shape[0], self.embedding_dim,
            )
            return []

        # 1. Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        cosine_scores = np.dot(self.embeddings, query_vec) / (norms * query_norm)

        # 2. Keyword overlap (Jaccard-like)
        query_tokens = set(query.lower().split())
        keyword_scores = np.zeros(n_fields)
        for i, meta in enumerate(self.field_meta):
            field_text = (
                f"{meta['field'].get('name', '')} "
                f"{meta['field'].get('description', '')}"
            ).lower()
            field_tokens = set(field_text.split())
            overlap = len(query_tokens & field_tokens)
            exact_bonus = 3 if meta["field"].get("name", "").lower() in query.lower() else 0
            keyword_scores[i] = overlap + exact_bonus

        # 3. Hybrid fusion
        c_min, c_max = cosine_scores.min(), cosine_scores.max()
        if c_max > c_min:
            cosine_norm = (cosine_scores - c_min) / (c_max - c_min)
        else:
            cosine_norm = cosine_scores

        k_max = keyword_scores.max()
        keyword_norm = keyword_scores / k_max if k_max > 0 else keyword_scores

        final_scores = 0.6 * cosine_norm + 0.4 * keyword_norm

        # 4. Top-k (safe with k = min(top_k, n_fields))
        top_indices = np.argpartition(final_scores, -k)[-k:]
        top_indices = top_indices[np.argsort(-final_scores[top_indices])]

        results = []
        for idx in top_indices:
            results.append({
                **self.field_meta[idx],
                "embedding_score": float(cosine_scores[idx]),
                "keyword_score": float(keyword_scores[idx]),
                "hybrid_score": float(final_scores[idx]),
            })
        return results


# ── Singleton Registry ──────────────────────────────────────────────────
_index_registry: Dict[str, SchemaFieldIndex] = {}
_registry_lock = asyncio.Lock()


async def get_schema_index(schema: Dict, tenant_id: str = "default") -> SchemaFieldIndex:
    """Get or build schema index for tenant. Cached in memory with stable hash + lock."""
    cache_key = f"{tenant_id}:{_stable_hash(schema)}"

    if cache_key in _index_registry:
        return _index_registry[cache_key]

    async with _registry_lock:
        if cache_key in _index_registry:
            return _index_registry[cache_key]
        _index_registry[cache_key] = await SchemaFieldIndex.load_or_build(schema, tenant_id=tenant_id)
        return _index_registry[cache_key]


async def warm_schema_index(schema: Dict, tenant_id: str = "default") -> Optional[SchemaFieldIndex]:
    """Build/load the schema embedding index at startup (off the request path)."""
    if not schema:
        logger.info("warm_schema_index: empty schema, skipping")
        return None
    try:
        index = await get_schema_index(schema, tenant_id)
        logger.info(
            "warm_schema_index: ready (%d fields, dim=%d, model=%s)",
            len(index.field_keys), index.embedding_dim, EMBEDDING_MODEL,
        )
        return index
    except Exception as e:
        logger.error("warm_schema_index: failed to build index: %s", e)
        return None


async def search_relevant_fields(
    user_query: str,
    schema: Dict,
    tenant_id: str = "default",
    top_k: int = 8,
) -> List[Dict]:
    """Public API: embed query (cached) and return top-k relevant fields."""
    index = await get_schema_index(schema, tenant_id)
    # Pin the query embedding to the index's provider to keep dimensions aligned.
    query_embedding = await get_query_embedding(user_query, provider=index.embedding_provider)
    return index.search(user_query, query_embedding, top_k)
