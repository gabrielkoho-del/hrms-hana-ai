"""
Lightweight in-memory schema embedding index for semantic field retrieval.
No external vector DB — pure Python + NumPy for <500 fields.
Shares Gemini API config with ingestion.gemini_embedder but avoids LlamaIndex overhead.

BATCH-FIRST DESIGN:
- embed_content(contents=LIST) is the primary path (1 HTTP call for N fields)
- On transient batch failure (e.g., 429 rate limit), retry the SAME batch with
  backoff that honors the API's RetryInfo.retryDelay. We deliberately do NOT fall
  back to per-item calls: splitting a batch into N single requests multiplies the
  request count and pushes us further over the per-minute quota.
"""

import os
import json
import math
import time
import asyncio
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("hr_agent")

# ── Suppress noisy HTTP client logs from google-genai SDK ──────────────
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ── Config ──────────────────────────────────────────────────────────────
# Schema (field) embeddings AND query embeddings MUST use the SAME model so
# their vector dimensions align. Both paths below derive from these two values,
# so changing the model here keeps schema build + query embed consistent.
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Expected output dimension for the configured model. Used for validation only;
# the true dim is also read back from the API response at build time.
EMBEDDING_DIM = int(os.getenv("GEMINI_EMBEDDING_DIM", "768"))  # gemini-embedding-2 = 768
SCHEMA_INDEX_CACHE_DIR = Path(os.getenv("SCHEMA_CACHE_DIR", "data/schema_cache"))
BATCH_SIZE = 100

# ── Cohere fallback config ──────────────────────────────────────────────
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_EMBEDDING_MODEL = os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
COHERE_EMBEDDING_DIM = int(os.getenv("COHERE_EMBEDDING_DIM", "1024"))
COHERE_BATCH_SIZE = 96  # Cohere max inputs per embed call

# ── Module-level singletons ─────────────────────────────────────────────
_embed_client: Optional["_GeminiEmbeddingClient"] = None
_embed_client_lock = asyncio.Lock()

_query_embedding_cache: Dict[str, Tuple[List[float], float]] = {}
_QUERY_CACHE_TTL_SECONDS = 300


def _get_embed_client() -> "_FallbackEmbeddingClient":
    """Get or create the module-level embedding client singleton.

    Returns a fallback-aware client: primary is Gemini, fallback is Cohere
    (only when COHERE_API_KEY is configured). On a fatal rate-limit/quota
    error from the primary, the client switches to the fallback provider
    for the rest of the session so embeddings remain dimension-consistent.
    """
    global _embed_client
    if _embed_client is None:
        primary = _GeminiEmbeddingClient(GEMINI_API_KEY, EMBEDDING_MODEL)
        fallback = None
        if COHERE_API_KEY:
            fallback = _CohereEmbeddingClient(
                api_key=COHERE_API_KEY,
                model=COHERE_EMBEDDING_MODEL,
            )
        _embed_client = _FallbackEmbeddingClient(primary, fallback)
    return _embed_client


def embed_text_sync(text: str, model: str = EMBEDDING_MODEL, api_key: str = GEMINI_API_KEY) -> List[float]:
    """Synchronous single-text embedding for low-latency sync paths (e.g., intent exemplar retrieval).

    Uses the fallback-aware client so dimensions stay aligned with the async batch path.
    """
    client = _get_embed_client()
    return client.embed_sync(text)


def _schema_hash(schema: Dict) -> str:
    """Stable hash for schema dict. Sorts lists to ensure deterministic output."""
    def _stable_dump(obj):
        if isinstance(obj, dict):
            return {k: _stable_dump(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            return sorted((_stable_dump(item) for item in obj), key=str)
        return obj
    return hashlib.md5(
        json.dumps(_stable_dump(schema), default=str).encode()
    ).hexdigest()[:16]


# ── Gemini Embedding Client ────────────────────────────────────────────
class _GeminiEmbeddingClient:
    """Minimal async embedding client. Reuses API key from gemini_embedder config.

    BATCH-FIRST ARCHITECTURE:
       1. Primary: batch via explicit Content objects (avoids SDK aggregation bug)
       2. Transient batch failure (e.g., 429): retry the SAME batch with backoff
          that honors the server's RetryInfo.retryDelay. No per-item fan-out.
    """

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._client = None  # Lazy init

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def _embed_single(self, text: str) -> List[float]:
        """Embed a single text — used for the single-item path (e.g. queries)."""
        client = self._get_client()

        def _call():
            return client.models.embed_content(
                model=self.model,
                contents=text,  # SDK auto-wraps string into Content/Part
            )

        resp = await asyncio.to_thread(_call)
        return resp.embeddings[0].values

    def _is_rate_limited(self, exc: Exception) -> bool:
        """Return True if the exception represents a 429 / quota-exhausted error."""
        msg = str(exc)
        return "429" in msg or "RESOURCE_EXHAUSTED" in msg

    def _is_daily_quota_exhausted(self, exc: Exception) -> bool:
        """Return True if the error indicates a DAILY (RPD) quota exhaustion.

        A daily cap resets next day, not in seconds, so retrying is futile and
        only delays the fallback embedder (Cohere) from taking over. Detect via the
        quotaId containing 'PerDay' / 'Per Day' / 'daily', or an implausibly large
        retry-delay hint (a per-minute cap uses tens of seconds, not hours).
        """
        msg = str(exc)
        if "PerDay" in msg or "Per Day" in msg or "daily" in msg.lower():
            return True
        import re
        match = re.search(r"retry in\s+([\d.]+)\s*s", msg, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1)) > 3600
            except ValueError:
                return False
        return False

    def _retry_delay_seconds(self, exc: Exception, default: float) -> float:
        """Extract the server's recommended retry delay (RetryInfo.retryDelay).

        The Gemini API embeds e.g. "Please retry in 43.984572047s." in the error
        message. Fall back to `default` if it cannot be parsed.
        """
        import re
        match = re.search(r"retry in\s+([\d.]+)\s*s", str(exc), re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return default
        return default

    async def _embed_batch(self, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        """Embed a batch of texts in a SINGLE API call, retrying on transient failure.

        PATCH: google-genai SDK's embed_content(contents=list_of_strings)
        silently aggregates all strings into a single multi-part Content,
        returning 1 embedding instead of N. Fix: wrap each text in an
        explicit Content object so the SDK treats them as separate items.

        On a 429 / quota-exhausted error we retry the SAME batch after sleeping
        for the duration recommended by the server (RetryInfo.retryDelay), with
        capped exponential backoff. We never fan the batch out into per-item
        calls, since that multiplies request count and worsens rate limiting.
        """
        if not texts:
            return []

        client = self._get_client()

        def _call():
            from google.genai import types
            # Explicit Content wrapping — each text becomes its own Content
            # so the SDK sends N separate items to the batch embedding API
            contents = [
                types.Content(parts=[types.Part.from_text(text=t)])
                for t in texts
            ]
            return client.models.embed_content(
                model=self.model,
                contents=contents,
            )

        for attempt in range(max_retries):
            try:
                resp = await asyncio.to_thread(_call)
                embeddings = [emb.values for emb in resp.embeddings]
                # ── Defensive count validation ─────────────────────────
                if len(embeddings) != len(texts):
                    raise RuntimeError(
                        f"Batch embedding count mismatch: expected {len(texts)}, got {len(embeddings)}. "
                        f"This indicates an SDK aggregation bug."
                    )
                return embeddings
            except Exception as e:
                # Daily (RPD) quota exhaustion: the limit resets next day, not in
                # seconds, so retrying only wastes time. Fail fast so the fallback
                # client can switch to Cohere immediately instead of burning retries.
                if self._is_daily_quota_exhausted(e):
                    logger.warning(
                        "Batch embedding DAILY quota exhausted (%d items) - failing fast to fallback: %s",
                        len(texts), e,
                    )
                    raise
                if self._is_rate_limited(e) and attempt < max_retries - 1:
                    delay = self._retry_delay_seconds(e, 2.0 ** attempt * 5)
                    logger.warning(
                        "Batch embedding rate-limited (%d items), attempt %d/%d — "
                        "sleeping %.1fs before retry: %s",
                        len(texts), attempt + 1, max_retries, delay, e,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.warning("Batch embedding failed for %d items: %s", len(texts), e)
                raise

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with a batch-first strategy.

        1. A single text goes through the single-item path (used by queries).
        2. Multiple texts are chunked (BATCH_SIZE per chunk) and each chunk is
           embedded in one batch API call. Transient batch failures (e.g., 429)
           are retried with backoff by `_embed_batch` — there is no per-item
           fallback, which would multiply request count and worsen rate limits.
        Schema cache persists to disk so this cost only applies on first run.
        """
        if not texts:
            return []

        if len(texts) == 1:
            return [await self._embed_single(texts[0])]

        all_embeddings = []
        chunks = [
            texts[i:i + BATCH_SIZE]
            for i in range(0, len(texts), BATCH_SIZE)
        ]

        for chunk in chunks:
            chunk_embeddings = await self._embed_batch(chunk)
            all_embeddings.extend(chunk_embeddings)
            logger.info("Batch embedded %d/%d fields", len(all_embeddings), len(texts))

        return all_embeddings

    def embed_sync(self, text: str) -> List[float]:
        """Synchronous single-text embedding for low-latency sync paths (e.g., intent exemplars)."""
        client = self._get_client()

        def _call():
            return client.models.embed_content(
                model=self.model,
                contents=text,
            )

        resp = _call()
        return resp.embeddings[0].values


# ── Cohere Embedding Client ─────────────────────────────────────────────
class _CohereEmbeddingClient:
    """Minimal async embedding client for Cohere's /v2/embed endpoint.

    Uses the same batch-first strategy as the Gemini client:
      - Primary path: batch up to 96 inputs per HTTP call
      - Retries on transient 429 / rate-limit errors
      - No per-item fan-out (would multiply request count)
    """

    def __init__(self, api_key: str, model: str = COHERE_EMBEDDING_MODEL):
        self.api_key = api_key
        self.model = model
        self._dim = COHERE_EMBEDDING_DIM

    def _is_rate_limited(self, exc: Exception) -> bool:
        msg = str(exc)
        return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate limit" in msg.lower()

    async def _embed_batch(self, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        if not texts:
            return []

        def _call():
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Client-Name": "hr-ai-agent",
            }
            payload = {
                "model": self.model,
                "input_type": "search_document",
                "texts": texts,
                "embedding_types": ["float"],
            }
            resp = requests.post(
                "https://api.cohere.com/v2/embed",
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", {}).get("float", [])
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Cohere batch embedding count mismatch: expected {len(texts)}, got {len(embeddings)}."
                )
            return embeddings

        for attempt in range(max_retries):
            try:
                return await asyncio.to_thread(_call)
            except Exception as e:
                if self._is_rate_limited(e) and attempt < max_retries - 1:
                    delay = 2.0 ** attempt * 5
                    logger.warning(
                        "Cohere batch embedding rate-limited (%d items), attempt %d/%d — "
                        "sleeping %.1fs before retry: %s",
                        len(texts), attempt + 1, max_retries, delay, e,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.warning("Cohere batch embedding failed for %d items: %s", len(texts), e)
                raise

    async def _embed_single(self, text: str) -> List[float]:
        embeddings = await self._embed_batch([text])
        return embeddings[0]

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if len(texts) == 1:
            return [await self._embed_single(texts[0])]

        all_embeddings = []
        chunks = [
            texts[i:i + COHERE_BATCH_SIZE]
            for i in range(0, len(texts), COHERE_BATCH_SIZE)
        ]

        for chunk in chunks:
            chunk_embeddings = await self._embed_batch(chunk)
            all_embeddings.extend(chunk_embeddings)
            logger.info("Cohere batch embedded %d/%d inputs", len(all_embeddings), len(texts))

        return all_embeddings

    def embed_sync(self, text: str) -> List[float]:
        def _call():
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Client-Name": "hr-ai-agent",
            }
            payload = {
                "model": self.model,
                "input_type": "search_query",
                "texts": [text],
                "embedding_types": ["float"],
            }
            resp = requests.post(
                "https://api.cohere.com/v2/embed",
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", {}).get("float", [])
            if not embeddings:
                raise RuntimeError("Cohere returned no embeddings for query.")
            return embeddings[0]

        return _call()


# ── Fallback Embedding Client ──────────────────────────────────────────
class _FallbackEmbeddingClient:
    """Wraps a primary and fallback embedder.

    Tries the primary client first. If the primary fails with a rate-limit /
    quota-exhausted error, switches to the fallback for the remainder of the
    session so we do not mix providers within a single index build (which
    would create dimension-mismatched numpy arrays).

    PROVIDER LOCKING:
    Once an index is built (or loaded from cache) with a specific provider,
    callers must lock the client to that provider via `lock_provider()` so
    that subsequent query embeddings stay dimension-aligned with the index.
    This prevents a mid-session fallback switch from producing vectors of a
    different dimension than the cached index.
    """

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self._active = primary
        self._locked = False

    def _provider_name(self, client) -> str:
        if self.fallback is not None and client is self.fallback:
            return "cohere"
        return "gemini"

    def get_active_provider(self) -> str:
        """Return 'gemini' or 'cohere' for the currently active client."""
        return self._provider_name(self._active)

    def lock_provider(self, provider_name: str) -> None:
        """Lock the client to a specific provider (called after index build/load).

        Once locked, rate-limit failures will NOT trigger a provider switch,
        guaranteeing query embeddings use the same model/dimension as the
        cached index.
        """
        self._locked = True
        if provider_name == "cohere" and self.fallback is not None:
            self._active = self.fallback
        else:
            self._active = self.primary

    def _is_fatal(self, exc: Exception) -> bool:
        msg = str(exc)
        return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()

    async def _try_active(self, coro_factory):
        try:
            return await coro_factory()
        except Exception as e:
            if self._locked:
                raise
            if self._active is self.primary and self.fallback and self._is_fatal(e):
                logger.warning(
                    "Primary embedder failed (%s). Switching to fallback provider for this session.", e
                )
                self._active = self.fallback
                return await coro_factory()
            raise

    async def embed(self, texts: List[str]) -> List[List[float]]:
        return await self._try_active(lambda: self._active.embed(texts))

    async def _embed_single(self, text: str) -> List[float]:
        return (await self._try_active(lambda: self._active.embed([text])))[0]

    def embed_sync(self, text: str) -> List[float]:
        try:
            return self._active.embed_sync(text)
        except Exception as e:
            if self._locked:
                raise
            if self._active is self.primary and self.fallback and self._is_fatal(e):
                logger.warning(
                    "Primary embedder failed (%s). Switching to fallback provider for this session.", e
                )
                self._active = self.fallback
                return self._active.embed_sync(text)
            raise


# ── Schema Field Index ──────────────────────────────────────────────────
class SchemaFieldIndex:
    """In-memory embedding index for schema fields with hybrid keyword fallback."""

    def __init__(self, schema: Dict, tenant_id: str = "default"):
        self.schema = schema
        self.tenant_id = tenant_id
        self.field_keys: List[str] = []
        self.field_meta: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_dim: int = EMBEDDING_DIM
        self.embedding_provider: str = "gemini"
        self._built_at: float = 0.0

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

    async def build(self, embedding_client: _GeminiEmbeddingClient) -> "SchemaFieldIndex":
        """Build index by embedding all fields. Idempotent."""
        fields_to_embed = []

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
                fields_to_embed.append(self._build_field_text(entity_name, field))

        if not fields_to_embed:
            logger.warning("SchemaFieldIndex: no fields to index")
            return self

        logger.info("SchemaFieldIndex: embedding %d fields via batch API...", len(fields_to_embed))
        vectors = await embedding_client.embed(fields_to_embed)
        self.embeddings = np.array(vectors, dtype=np.float32)

        # ── Shape validation ─────────────────────────────────────────
        if self.embeddings.shape[0] != len(self.field_keys):
            raise RuntimeError(
                f"Embedding shape mismatch: expected ({len(self.field_keys)}, {self.embedding_dim}), "
                f"got {self.embeddings.shape}. Aborting to prevent corrupted cache."
            )
        # Record the ACTUAL dimension the model produced. This is what we stamp
        # into the cache so a later model/dim change forces a rebuild.
        self.embedding_dim = int(self.embeddings.shape[1])
        # Record which provider produced these embeddings and lock the fallback
        # client to it so query embeddings stay dimension-aligned with this index.
        self.embedding_provider = embedding_client.get_active_provider()
        if hasattr(embedding_client, "lock_provider"):
            embedding_client.lock_provider(self.embedding_provider)

        self._built_at = time.time()
        self._save_cache()
        return self

    def _cache_path(self) -> Path:
        SCHEMA_INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        schema_hash = _schema_hash(self.schema)
        provider_tag = getattr(self, "embedding_provider", "gemini")
        # Include model AND provider in the cache key so switching embedding models
        # OR providers forces a clean rebuild instead of loading a dimension-mismatched
        # cache. Dimension is intentionally omitted: the true dim is read from the API
        # response at build time and validated on load.
        model_tag = EMBEDDING_MODEL.replace("/", "_").replace(":", "_")
        return SCHEMA_INDEX_CACHE_DIR / f"schema_{self.tenant_id}_{schema_hash}_{provider_tag}_{model_tag}.npz"

    def _save_cache(self):
        if self.embeddings is None:
            return
        path = self._cache_path()

        # Clean up any other cache files for the same tenant+schema_hash+model,
        # including old-format files that embedded the dimension in the filename
        # and legacy files that predate the model-in-filename format entirely.
        model_tag = EMBEDDING_MODEL.replace("/", "_").replace(":", "_")
        provider_tag = getattr(self, "embedding_provider", "gemini")
        prefix = f"schema_{self.tenant_id}_"
        # Current format: schema_<tenant>_<hash>_<provider>_<model>.npz
        for f in SCHEMA_INDEX_CACHE_DIR.glob(prefix + "*" + f"_{provider_tag}_{model_tag}.npz"):
            if f != path and f.is_file():
                try:
                    f.unlink()
                    logger.info("SchemaFieldIndex: removed stale cache %s", f.name)
                except OSError as e:
                    logger.warning("SchemaFieldIndex: failed to remove stale cache %s: %s", f.name, e)
        # Old format with dim suffix: schema_<tenant>_<hash>_<model>_<dim>.npz
        for f in SCHEMA_INDEX_CACHE_DIR.glob(prefix + "*" + f"_{model_tag}_" + "[0-9]*.npz"):
            if f != path and f.is_file():
                try:
                    f.unlink()
                    logger.info("SchemaFieldIndex: removed old-format stale cache %s", f.name)
                except OSError as e:
                    logger.warning("SchemaFieldIndex: failed to remove old-format stale cache %s: %s", f.name, e)
        # Legacy format without model tag: schema_<tenant>_<hash>.npz
        for f in SCHEMA_INDEX_CACHE_DIR.glob(prefix + "*.npz"):
            if f == path:
                continue
            fname = f.name
            # If it doesn't contain the model tag, it's legacy
            if model_tag not in fname:
                try:
                    f.unlink()
                    logger.info("SchemaFieldIndex: removed legacy format cache %s", fname)
                except OSError as e:
                    logger.warning("SchemaFieldIndex: failed to remove legacy cache %s: %s", fname, e)

        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            keys=json.dumps(self.field_keys),
            meta=json.dumps(self.field_meta),
            built_at=self._built_at,
            embedding_model=EMBEDDING_MODEL,
            embedding_dim=self.embedding_dim,
            embedding_provider=provider_tag,
        )
        logger.info("SchemaFieldIndex: cached to %s (dim=%d, provider=%s)", path, self.embedding_dim, provider_tag)

    @classmethod
    async def load_or_build(
        cls,
        schema: Dict,
        tenant_id: str = "default",
        max_age_seconds: int = 31536000,  # 1 year — schema hash is the real invalidation
    ) -> "SchemaFieldIndex":
        """Load from cache if fresh and dimension-matched, otherwise build.

        The cache is keyed by (tenant, schema hash, model). Dimension is validated
        from the stored array shape on load, so a model swap that changes output
        dims forces a clean rebuild and the stale file is removed.
        """
        instance = cls(schema, tenant_id)
        client = _get_embed_client()
        active_provider = client.get_active_provider()
        # Stamp the active provider before computing the cache path so the path
        # tag matches the provider that will actually be used to (re)build.
        instance.embedding_provider = active_provider
        cache_path = instance._cache_path()

        if cache_path.exists():
            try:
                data = np.load(cache_path, allow_pickle=False)
                age = time.time() - float(data["built_at"])
                if age < max_age_seconds:  # Safety cap; schema hash handles real invalidation
                    cached_dim = int(data["embedding_dim"])
                    # ── Dimension validation on cache load ──────────────
                    # If the cached dim differs from the actual array shape, the
                    # model changed — drop the cache and rebuild so searches align.
                    actual_dim = int(data["embeddings"].shape[1]) if hasattr(data["embeddings"], "shape") else cached_dim
                    if actual_dim != EMBEDDING_DIM:
                        logger.warning(
                            "Cache dim mismatch: cached=%d, configured=%d. Rebuilding.",
                            actual_dim, EMBEDDING_DIM,
                        )
                        raise ValueError("Cache dimension mismatch")

                    # ── Provider validation on cache load ───────────────
                    # Gemini and Cohere produce different output dimensions; loading
                    # a cache from the wrong provider would silently break search.
                    cached_provider = str(data.get("embedding_provider", "gemini"))
                    if cached_provider != active_provider:
                        logger.warning(
                            "Cache provider mismatch: cached=%s, active=%s. Rebuilding.",
                            cached_provider, active_provider,
                        )
                        raise ValueError("Cache provider mismatch")

                    instance.embeddings = data["embeddings"]
                    instance.field_keys = json.loads(str(data["keys"]))
                    instance.field_meta = json.loads(str(data["meta"]))
                    instance.embedding_dim = actual_dim
                    instance.embedding_provider = cached_provider
                    instance._built_at = float(data["built_at"])

                    # Lock the fallback client to the provider that built this cache
                    # so query embeddings stay dimension-aligned with the index.
                    if hasattr(client, "lock_provider"):
                        client.lock_provider(cached_provider)

                    # ── Shape validation on cache load ─────────────────
                    if instance.embeddings.shape[0] != len(instance.field_keys):
                        logger.warning(
                            "Cache shape mismatch: embeddings=%s, field_keys=%d. Rebuilding.",
                            instance.embeddings.shape, len(instance.field_keys)
                        )
                        raise ValueError("Cache shape mismatch")

                    logger.info(
                        "SchemaFieldIndex: loaded %d fields from cache (age=%.0fs, dim=%d)",
                        len(instance.field_keys), age, actual_dim
                    )
                    return instance
            except Exception as e:
                logger.warning("SchemaFieldIndex: cache load failed, rebuilding: %s", e)

        client = _get_embed_client()
        return await instance.build(client)

    def search(self, query: str, query_embedding: List[float], top_k: int = 8) -> List[Dict]:
        """Hybrid search: embedding similarity + keyword overlap."""
        if self.embeddings is None or len(self.field_keys) == 0:
            return []

        n_fields = len(self.field_keys)

        # ── Shape validation ─────────────────────────────────────────
        if self.embeddings.shape[0] != n_fields:
            logger.error(
                "Search shape mismatch: embeddings.shape[0]=%d, field_keys=%d",
                self.embeddings.shape[0], n_fields
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
            field_text = f"{meta['field'].get('name', '')} {meta['field'].get('description', '')}".lower()
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


# ── Singleton Registry ─────────────────────────────────────────────────
_index_registry: Dict[str, SchemaFieldIndex] = {}
_registry_lock = asyncio.Lock()


async def get_schema_index(schema: Dict, tenant_id: str = "default") -> SchemaFieldIndex:
    """Get or build schema index for tenant. Cached in memory with stable hash + lock."""
    cache_key = f"{tenant_id}:{_schema_hash(schema)}"

    if cache_key in _index_registry:
        return _index_registry[cache_key]

    async with _registry_lock:
        # Double-check after acquiring lock
        if cache_key in _index_registry:
            return _index_registry[cache_key]
        _index_registry[cache_key] = await SchemaFieldIndex.load_or_build(schema, tenant_id)
        return _index_registry[cache_key]


async def warm_schema_index(schema: Dict, tenant_id: str = "default") -> Optional[SchemaFieldIndex]:
    """Build/load the schema embedding index at startup (off the request path).

    Schema is static for a given tenant, so we eagerly warm the in-memory index
    and on-disk cache during app startup rather than on the first user query.
    This avoids first-query latency and 429 exposure, and guarantees the cached
    dimension matches the currently configured embedding model.
    """
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


async def _get_query_embedding(user_query: str) -> List[float]:
    """Get query embedding with simple in-memory TTL cache."""
    cache_key = user_query.lower().strip()
    now = time.time()

    if cache_key in _query_embedding_cache:
        embedding, cached_at = _query_embedding_cache[cache_key]
        if now - cached_at < _QUERY_CACHE_TTL_SECONDS:
            return embedding

    client = _get_embed_client()
    embedding = (await client.embed([user_query]))[0]
    _query_embedding_cache[cache_key] = (embedding, now)
    return embedding


async def search_relevant_fields(
    user_query: str,
    schema: Dict,
    tenant_id: str = "default",
    top_k: int = 8,
) -> List[Dict]:
    """Public API: embed query (cached) and return top-k relevant fields."""
    index = await get_schema_index(schema, tenant_id)
    query_embedding = await _get_query_embedding(user_query)
    return index.search(user_query, query_embedding, top_k)