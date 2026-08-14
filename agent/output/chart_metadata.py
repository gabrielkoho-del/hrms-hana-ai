"""agent/output/chart_metadata.py

Hybrid metric extraction for multi-metric chart reports.

Uses a fast regex gate with longest-match-first ordering, falling back
to LLM extraction when the gate is ambiguous. Results are cached in a
TTL cache to reduce latency for repeated queries.
"""
import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional

from agent.core.utils import TTLCache
from agent.integrations.llm_client import call_llm

logger = logging.getLogger("hr_agent")

_METRIC_FAST_GATE_KEYWORDS = [
    "gross profit", "gross_profit", "grossprofit",
    "net profit", "net_profit", "netprofit",
    "operating income",
    "revenue", "expense", "profit", "cost", "margin",
    "ebitda", "income", "loss", "budget", "actual",
    "profitability",
]

_METRIC_LLM_PROMPT = """\
Extract all financial metrics mentioned in this query. Return ONLY JSON:
{{"metrics": ["revenue", "expense", "profit"], "time_period": "month", "year": 2025}}

Allowed metrics (use these exact lowercase forms):
revenue, expense, profit, gross_profit, net_profit, ebitda, cost, margin, income, loss, budget, actual, profitability

Query: {query}
"""

_METRIC_EXTRACTION_CACHE = TTLCache(ttl=300, maxsize=256)


def _metric_cache_key(query: str) -> str:
    normalized = query.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def extract_metrics_fast_gate(query: str) -> List[str]:
    """Regex word-boundary extraction with longest-match-first ordering.

    Returns deduplicated list preserving match order (most specific first).
    """
    q = query.lower()
    found: List[str] = []
    # Sort by length descending so "gross profit" matches before "profit"
    for kw in sorted(_METRIC_FAST_GATE_KEYWORDS, key=len, reverse=True):
        pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
        if pattern.search(q) and kw not in found:
            found.append(kw)
    return found


def should_fallback_to_llm(fast_metrics: List[str], query: str) -> bool:
    """Heuristic: when is the fast gate likely wrong or incomplete?"""
    q = query.lower()
    if not fast_metrics:
        return True
    # If user mentions multiple distinct financial terms but gate found only one
    financial_terms = ["revenue", "expense", "profit", "cost", "margin", "income", "loss", "budget", "actual", "ebitda"]
    mentioned = [t for t in financial_terms if t in q]
    if len(mentioned) >= 2 and len(fast_metrics) < 2:
        return True
    return False


async def extract_metrics_llm(query: str) -> List[str]:
    """LLM-based metric extraction using structured JSON response."""
    prompt = _METRIC_LLM_PROMPT.format(query=query)
    try:
        result = await call_llm(
            system_prompt="You are a financial query analyzer. Extract metrics precisely.",
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
        # Strip markdown code fences if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        parsed = json.loads(content.strip())
        metrics = parsed.get("metrics", [])
        return [m.lower().strip() for m in metrics if isinstance(m, str)]
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

    # Prefer LLM result if it found more metrics, otherwise use fast gate
    final_metrics = llm_metrics if len(llm_metrics) >= len(fast_metrics) else fast_metrics
    _METRIC_EXTRACTION_CACHE.set(cache_key, final_metrics)
    return final_metrics
