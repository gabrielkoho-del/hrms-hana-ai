"""
agent/integrations/embedding.py

Shared embedding infrastructure used by both SchemaFieldIndex and
IntentExemplarIndex. Owns:
  - The three embedding clients (Gemini primary, Cohere fallback, Fallback wrapper)
  - A module-level singleton for ad-hoc embedding calls
  - A dedicated-client helper for per-index provider-locked embedding
  - A BaseEmbeddingIndex class encapsulating the shared cache + build + load
    lifecycle so concrete indexes (schema fields, intent exemplars) only
    implement the domain-specific bits (text construction, search).

BATCH-FIRST DESIGN (preserved from the original schema_index.py):
- embed_content(contents=LIST) is the primary path (1 HTTP call for N texts)
- On transient batch failure (e.g., 429 rate limit), retry the SAME batch with
  backoff that honors the API's RetryInfo.retryDelay. We deliberately do NOT
  fall back to per-item calls: splitting a batch into N single requests
  multiplies the request count and pushes us further over the per-minute quota.

PROVIDER ISOLATION (fix for the dim=3072 != dim=1024 regression):
- Each BaseEmbeddingIndex tracks its own `embedding_provider` and `embedding_dim`
- Query-time embedding for an index uses `_embed_texts_with_provider(...)` with
  the index's stored provider, so a different provider being used by another
  index cannot cross-contaminate dimensions.
- The global fallback client no longer gets `_locked` to a specific provider
  on index build, eliminating the previous "last loader wins" failure mode.
"""

import os
import json
import time
import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("hr_agent")

# ── Suppress noisy HTTP client logs from google-genai SDK ──────────────
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ── Config ──────────────────────────────────────────────────────────────
# All embed indexes in this codebase MUST use the same model so dimensions align.
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Expected output dimension for the configured model. Used for validation only;
# the true dim is also read back from the API response at build time.
EMBEDDING_DIM = int(os.getenv("GEMINI_EMBEDDING_DIM", "3072"))  # gemini-embedding-2 = 3072
BATCH_SIZE = 100

# ── Cohere fallback config ──────────────────────────────────────────────
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_EMBEDDING_MODEL = os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
COHERE_EMBEDDING_DIM = int(os.getenv("COHERE_EMBEDDING_DIM", "1024"))
COHERE_BATCH_SIZE = 96  # Cohere max inputs per embed call

# ── Module-level singletons ─────────────────────────────────────────────
_embed_client: Optional["_FallbackEmbeddingClient"] = None
_embed_client_lock = asyncio.Lock()

# Query embedding cache keyed by f"{provider or 'global'}:{query}"
_query_embedding_cache: Dict[str, Tuple[List[float], float]] = {}
_QUERY_CACHE_TTL_SECONDS = 300


# ── Public helpers ──────────────────────────────────────────────────────
def get_embed_client() -> "_FallbackEmbeddingClient":
    """Get or create the module-level embedding client singleton.

    Returns a fallback-aware client: primary is Gemini, fallback is Cohere
    (only when COHERE_API_KEY is configured). The singleton is intentionally
    NOT locked to a specific provider anymore — each index now tracks its own
    provider and uses `_embed_texts_with_provider` for query-time isolation.
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
    """Synchronous single-text embedding for low-latency sync paths.

    Uses the fallback-aware singleton so dimensions stay aligned with the
    async batch path. Do NOT use this for query-time embedding when the
    index expects a specific provider — use `_embed_texts_with_provider`
    instead, or pass `provider=` to the async query helpers.
    """
    client = get_embed_client()
    return client.embed_sync(text)


async def _embed_texts_with_provider(texts: List[str], provider: str) -> List[List[float]]:
    """Embed texts using a dedicated client for the specified provider.

    This is the safe path for index query-time embedding: it constructs a
    fresh FallbackEmbeddingClient with `_active` set to the requested
    provider, so a different provider being used by another index cannot
    cross-contaminate dimensions.
    """
    primary = _GeminiEmbeddingClient(GEMINI_API_KEY, EMBEDDING_MODEL)
    fallback = None
    if COHERE_API_KEY:
        fallback = _CohereEmbeddingClient(
            api_key=COHERE_API_KEY,
            model=COHERE_EMBEDDING_MODEL,
        )
    client = _FallbackEmbeddingClient(primary, fallback)
    if provider == "cohere" and fallback is not None:
        client._active = fallback
    elif provider == "gemini" or provider is None:
        client._active = primary
    return await client.embed(texts)


async def get_query_embedding(user_query: str, provider: Optional[str] = None) -> List[float]:
    """Get a query embedding, optionally pinned to a specific provider.

    When `provider` is given, uses a dedicated client (per-index isolation).
    Cache key includes the provider so an embedding for one provider is never
    served for a different provider.
    """
    cache_key = f"{provider or 'global'}:{user_query.lower().strip()}"
    now = time.time()

    if cache_key in _query_embedding_cache:
        embedding, cached_at = _query_embedding_cache[cache_key]
        if now - cached_at < _QUERY_CACHE_TTL_SECONDS:
            return embedding

    if provider:
        vectors = await _embed_texts_with_provider([user_query], provider)
    else:
        client = get_embed_client()
        vectors = await client.embed([user_query])

    embedding = vectors[0]
    _query_embedding_cache[cache_key] = (embedding, now)
    return embedding


# ── Hashing helper used by both indexes for cache key stability ────────
def stable_hash(obj: Any) -> str:
    """Stable hash for an arbitrary dict/list structure. Sorts to be deterministic."""
    def _stable_dump(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: _stable_dump(v) for k, v in sorted(o.items())}
        if isinstance(o, list):
            return sorted((_stable_dump(item) for item in o), key=str)
        return o
    return hashlib.md5(
        json.dumps(_stable_dump(obj), default=str).encode()
    ).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════
# Embedding clients
# ═══════════════════════════════════════════════════════════════════════

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
        msg = str(exc)
        return "429" in msg or "RESOURCE_EXHAUSTED" in msg

    def _is_daily_quota_exhausted(self, exc: Exception) -> bool:
        """A daily cap resets next day, not in seconds, so retrying is futile."""
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
        import re
        match = re.search(r"retry in\s+([\d.]+)\s*s", str(exc), re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return default
        return default

    async def _embed_batch(self, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        """Embed a batch of texts in a SINGLE API call, retrying on transient failure."""
        if not texts:
            return []

        client = self._get_client()

        def _call():
            from google.genai import types
            # Explicit Content wrapping — each text becomes its own Content
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
                if len(embeddings) != len(texts):
                    raise RuntimeError(
                        f"Batch embedding count mismatch: expected {len(texts)}, "
                        f"got {len(embeddings)}. This indicates an SDK aggregation bug."
                    )
                return embeddings
            except Exception as e:
                if self._is_daily_quota_exhausted(e):
                    logger.warning(
                        "Batch embedding DAILY quota exhausted (%d items) - "
                        "failing fast to fallback: %s",
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
        if not texts:
            return []

        if len(texts) == 1:
            return [await self._embed_single(texts[0])]

        all_embeddings: List[List[float]] = []
        chunks = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

        for chunk in chunks:
            chunk_embeddings = await self._embed_batch(chunk)
            all_embeddings.extend(chunk_embeddings)
            logger.info("Batch embedded %d/%d fields", len(all_embeddings), len(texts))

        return all_embeddings

    def embed_sync(self, text: str) -> List[float]:
        client = self._get_client()

        def _call():
            return client.models.embed_content(
                model=self.model,
                contents=text,
            )

        resp = _call()
        return resp.embeddings[0].values


class _CohereEmbeddingClient:
    """Minimal async embedding client for Cohere's /v2/embed endpoint.

    Uses the same batch-first strategy as the Gemini client.
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
                    f"Cohere batch embedding count mismatch: expected {len(texts)}, "
                    f"got {len(embeddings)}."
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

        all_embeddings: List[List[float]] = []
        chunks = [texts[i:i + COHERE_BATCH_SIZE] for i in range(0, len(texts), COHERE_BATCH_SIZE)]

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


class _FallbackEmbeddingClient:
    """Wraps a primary and fallback embedder.

    Tries the primary client first. If the primary fails with a rate-limit /
    quota-exhausted error, switches to the fallback for the remainder of the
    session so we do not mix providers within a single index build (which
    would create dimension-mismatched numpy arrays).

    Note: callers that need provider isolation should use
    `_embed_texts_with_provider(...)` instead of relying on this singleton.
    """

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self._active = primary

    def _provider_name(self, client) -> str:
        if self.fallback is not None and client is self.fallback:
            return "cohere"
        return "gemini"

    def get_active_provider(self) -> str:
        """Return 'gemini' or 'cohere' for the currently active client."""
        return self._provider_name(self._active)

    def _is_fatal(self, exc: Exception) -> bool:
        msg = str(exc)
        return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()

    async def _try_active(self, coro_factory):
        try:
            return await coro_factory()
        except Exception as e:
            if self._active is self.primary and self.fallback and self._is_fatal(e):
                logger.warning(
                    "Primary embedder failed (%s). Switching to fallback provider "
                    "for this session.", e
                )
                self._active = self.fallback
                return await coro_factory()
            raise

    async def embed(self, texts: List[str]) -> List[List[float]]:
        return await self._try_active(lambda: self._active.embed(texts))

    def embed_sync(self, text: str) -> List[float]:
        try:
            return self._active.embed_sync(text)
        except Exception as e:
            if self._active is self.primary and self.fallback and self._is_fatal(e):
                logger.warning(
                    "Primary embedder failed (%s). Switching to fallback provider "
                    "for this session.", e
                )
                self._active = self.fallback
                return self._active.embed_sync(text)
            raise


# ═══════════════════════════════════════════════════════════════════════
# BaseEmbeddingIndex — shared cache + build + load lifecycle
# ═══════════════════════════════════════════════════════════════════════

class BaseEmbeddingIndex:
    """Base class for embedding-based indexes (schema fields, intent exemplars, ...).

    Concrete subclasses MUST implement:
      - `_content_hash() -> str`            : stable hash of the content (used in cache key)
      - `_texts_to_embed() -> List[str]`    : texts to send to the embedding API on build
      - `_extra_cache_fields() -> Dict`     : subclass-specific fields to persist in the npz
      - `_restore_from_cache(data: np.lib.npyio.NpzFile) -> bool`: restore subclass state

    The base class handles:
      - Embedding via the shared `_embed_texts_with_provider` path
      - Cache file naming: <prefix>_<tenant>_<hash>_<provider>_<model>.npz
      - Cache load with dimension validation
      - Per-index `embedding_provider` and `embedding_dim` tracking
      - Stale-cache cleanup (current format, old dim-suffix format, legacy no-model format)
    """

    # Subclass overrides
    cache_prefix: str = "base"  # e.g. "schema" or "intent"
    cache_dir: Path = Path("data/embedding_cache")

    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_dim: int = EMBEDDING_DIM
        self.embedding_provider: str = "gemini"
        self._built_at: float = 0.0

    # ── Abstract interface (subclasses MUST implement) ────────────────
    def _content_hash(self) -> str:
        raise NotImplementedError

    def _texts_to_embed(self) -> List[str]:
        raise NotImplementedError

    def _extra_cache_fields(self) -> Dict[str, Any]:
        """Return subclass-specific fields to persist in the npz cache."""
        return {}

    def _restore_from_cache(self, data: Any) -> bool:
        """Restore subclass state from the loaded npz. Return True on success."""
        raise NotImplementedError

    # ── Cache path / search helpers ────────────────────────────────────
    def _model_tag(self) -> str:
        return EMBEDDING_MODEL.replace("/", "_").replace(":", "_")

    def _cache_path(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / (
            f"{self.cache_prefix}_{self.tenant_id}_{self._content_hash()}_"
            f"{self.embedding_provider}_{self._model_tag()}.npz"
        )

    def _find_existing_cache(self) -> Optional[Path]:
        """Find an existing cache file for this index, regardless of provider tag.

        This lets us reuse a cache built by a different provider (e.g., Cohere)
        across restarts instead of forcing a rebuild every time.
        """
        prefix = f"{self.cache_prefix}_{self.tenant_id}_{self._content_hash()}_"
        suffix = f"_{self._model_tag()}.npz"
        candidates = [
            p for p in self.cache_dir.glob(prefix + "*.npz")
            if p.name.endswith(suffix)
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    def _save_cache(self) -> None:
        if self.embeddings is None:
            return
        path = self._cache_path()
        model_tag = self._model_tag()
        provider_tag = self.embedding_provider
        prefix = f"{self.cache_prefix}_{self.tenant_id}_"

        # Current format: <prefix>_<tenant>_<hash>_<provider>_<model>.npz
        for f in self.cache_dir.glob(prefix + "*" + f"_{provider_tag}_{model_tag}.npz"):
            if f != path and f.is_file():
                try:
                    f.unlink()
                    logger.info("%s: removed stale cache %s", self.cache_prefix, f.name)
                except OSError as e:
                    logger.warning("%s: failed to remove stale cache %s: %s",
                                   self.cache_prefix, f.name, e)
        # Old format with dim suffix: <prefix>_<tenant>_<hash>_<model>_<dim>.npz
        for f in self.cache_dir.glob(prefix + "*" + f"_{model_tag}_" + "[0-9]*.npz"):
            if f != path and f.is_file():
                try:
                    f.unlink()
                    logger.info("%s: removed old-format stale cache %s",
                                self.cache_prefix, f.name)
                except OSError as e:
                    logger.warning("%s: failed to remove old-format stale cache %s: %s",
                                   self.cache_prefix, f.name, e)
        # Legacy format without model tag
        for f in self.cache_dir.glob(prefix + "*.npz"):
            if f == path:
                continue
            if model_tag not in f.name:
                try:
                    f.unlink()
                    logger.info("%s: removed legacy format cache %s",
                                self.cache_prefix, f.name)
                except OSError as e:
                    logger.warning("%s: failed to remove legacy cache %s: %s",
                                   self.cache_prefix, f.name, e)

        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            built_at=self._built_at,
            embedding_model=EMBEDDING_MODEL,
            embedding_dim=self.embedding_dim,
            embedding_provider=provider_tag,
            **self._extra_cache_fields(),
        )
        logger.info(
            "%s: cached to %s (dim=%d, provider=%s)",
            self.cache_prefix, path, self.embedding_dim, provider_tag,
        )

    # ── Build / Load lifecycle ─────────────────────────────────────────
    async def build(self) -> "BaseEmbeddingIndex":
        """Build the index by embedding the texts. Idempotent."""
        texts = self._texts_to_embed()
        if not texts:
            logger.warning("%s: no texts to index", self.cache_prefix)
            return self

        # Use the global singleton for the initial build. The singleton's
        # fallback mechanism may switch to Cohere mid-build; we record which
        # provider actually produced the embeddings so query-time embedding
        # uses the same provider via _embed_texts_with_provider.
        client = get_embed_client()
        logger.info("%s: embedding %d texts via batch API...",
                    self.cache_prefix, len(texts))
        vectors = await client.embed(texts)
        self.embeddings = np.array(vectors, dtype=np.float32)

        if self.embeddings.shape[0] != len(texts):
            raise RuntimeError(
                f"{self.cache_prefix}: embedding count mismatch: expected "
                f"{len(texts)}, got {self.embeddings.shape[0]}"
            )
        self.embedding_dim = int(self.embeddings.shape[1])
        self.embedding_provider = client.get_active_provider()
        self._built_at = time.time()
        self._save_cache()
        return self

    @classmethod
    async def load_or_build(
        cls,
        *args: Any,
        max_age_seconds: int = 31536000,  # 1 year; content hash handles real invalidation
        **kwargs: Any,
    ) -> "BaseEmbeddingIndex":
        """Load from cache if fresh and dimension-matched, otherwise build."""
        instance = cls(*args, **kwargs)
        # Set the provider from the active client BEFORE finding the cache,
        # so _find_existing_cache can match files built by the same provider.
        client = get_embed_client()
        instance.embedding_provider = client.get_active_provider()

        path = instance._find_existing_cache()
        if path and path.exists():
            try:
                data = np.load(path, allow_pickle=False)
                age = time.time() - float(data["built_at"])
                if age >= max_age_seconds:
                    logger.info("%s: cache age %.0fs exceeds TTL, rebuilding",
                                instance.cache_prefix, age)
                else:
                    cached_dim = int(data["embedding_dim"])
                    actual_dim = (
                        int(data["embeddings"].shape[1])
                        if hasattr(data["embeddings"], "shape") else cached_dim
                    )
                    if cached_dim != actual_dim:
                        raise ValueError("Cache dimension mismatch")
                    # Let subclass restore its own state
                    if not instance._restore_from_cache(data):
                        raise ValueError("Subclass restore_from_cache failed")
                    instance.embedding_dim = actual_dim
                    instance.embedding_provider = str(
                        data.get("embedding_provider", "gemini")
                    )
                    instance._built_at = float(data["built_at"])

                    logger.info(
                        "%s: loaded from cache (age=%.0fs, dim=%d, provider=%s)",
                        instance.cache_prefix, age, actual_dim, instance.embedding_provider,
                    )
                    return instance
            except Exception as e:
                logger.warning("%s: cache load failed, rebuilding: %s",
                               instance.cache_prefix, e)

        return await instance.build()
