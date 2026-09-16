"""agent/output/chart_metadata.py

Hybrid metric extraction for multi-metric chart reports.

Uses the shared metric registry (config/metrics.yaml) as a single source of truth.
Fast path: pre-compiled regex word-boundary patterns + fuzzy typo recovery.
Falls back to LLM extraction when the gate is ambiguous. Results cached in a
TTL cache to reduce latency for repeated queries.
"""
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from agent.core.utils import TTLCache
from agent.integrations.llm_client import call_llm
from agent.core.metric_registry import (
    ALIAS_TO_CANONICAL,
    extract_metrics_fast_gate,
    allowed_metrics_prompt_text,
)

logger = logging.getLogger("hr_agent")

_METRIC_LLM_PROMPT = """\
Extract all metrics mentioned in this query. Return ONLY JSON:
{{"metrics": ["revenue", "expense", "profit"], "time_period": "month", "year": 2025}}

Allowed metrics (use these exact lowercase canonical forms):
{allowed_metrics}

Query: {query}
"""

_METRIC_EXTRACTION_CACHE = TTLCache(ttl=300, maxsize=256)


def _metric_cache_key(query: str) -> str:
    normalized = query.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def should_fallback_to_llm(fast_metrics: List[str], query: str) -> bool:
    q = query.lower()
    if not fast_metrics:
        return True
    mentioned = [a for a in ALIAS_TO_CANONICAL if a in q]
    if len(mentioned) >= 2 and len(fast_metrics) < 2:
        return True
    return False


async def extract_metrics_llm(query: str) -> List[str]:
    prompt = _METRIC_LLM_PROMPT.format(query=query, allowed_metrics=allowed_metrics_prompt_text())
    try:
        result = call_llm(
            system_prompt="You are a financial/HR query analyzer. Extract metrics precisely using the allowed canonical forms.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=256,
            json_mode=True,
            tier="executor",
            estimated_tokens=500,
        )
        if not result or not isinstance(result, dict):
            return []
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return []
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        parsed = json.loads(content.strip())
        raw = parsed.get("metrics", [])
        return [ALIAS_TO_CANONICAL.get(m.lower().strip(), m.lower().strip()) for m in raw if isinstance(m, str)]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        logger.warning("CHART_DEBUG_HYBRID: LLM metric extraction failed: %s", e)
        return []


async def extract_metrics_hybrid(user_query: str) -> List[str]:
    """Hybrid extraction: fast gate -> cache -> LLM fallback."""
    cache_key = _metric_cache_key(user_query)
    cached = _METRIC_EXTRACTION_CACHE.get(cache_key)
    if cached is not None:
        logger.info("CHART_DEBUG_HYBRID: cache hit for query key=%s metrics=%s", cache_key[:8], cached)
        return cached

    fast_metrics = extract_metrics_fast_gate(user_query)
    logger.info("CHART_DEBUG_HYBRID: fast gate metrics=%s", fast_metrics)

    if not should_fallback_to_llm(fast_metrics, user_query):
        _METRIC_EXTRACTION_CACHE.set(cache_key, fast_metrics)
        return fast_metrics

    logger.info("CHART_DEBUG_HYBRID: falling back to LLM extraction")
    llm_metrics = await extract_metrics_llm(user_query)
    logger.info("CHART_DEBUG_HYBRID: LLM metrics=%s", llm_metrics)

    final_metrics = llm_metrics if len(llm_metrics) >= len(fast_metrics) else fast_metrics
    _METRIC_EXTRACTION_CACHE.set(cache_key, final_metrics)
    return final_metrics
