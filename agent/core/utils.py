"""agent/core/utils.py

Shared utilities used across the agent core:
  - Token counting via tiktoken with safe fallback
  - Lightweight TTL cache for prompt/schema caching
"""
import logging
import time
from typing import Any, Dict, Optional, Tuple

import tiktoken

logger = logging.getLogger("hr_agent")


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (OpenAI-compatible) with safe fallback.

    Uses cl100k_base encoding, which matches GPT-4/OpenAI tool-calling models.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text, disallowed_special=()))
    except Exception:
        # Fallback: conservative approximation (slightly overestimates)
        return int(len(text) / 3.0)


class TTLCache:
    """Lightweight TTL cache with no external dependencies."""

    def __init__(self, ttl: int = 300, maxsize: int = 100):
        self.ttl = ttl
        self.maxsize = maxsize
        self._store: Dict[str, Tuple[str, float]] = {}

    def get(self, key: str) -> Optional[str]:
        if key not in self._store:
            return None
        value, ts = self._store[key]
        if time.time() - ts > self.ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: str):
        if len(self._store) >= self.maxsize:
            oldest_key = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest_key]
        self._store[key] = (value, time.time())

    def clear(self):
        self._store.clear()


# Shared schema cache instance (5 min TTL, 100 entries)
schema_cache = TTLCache(ttl=300, maxsize=100)
