"""agent/output/metric_registry.py

Single source of truth for metric keyword extraction (financial + HR).

Loaded from config/metrics.yaml (fail-open to built-in defaults if the file is
missing or malformed). Builds, ONCE at import time:
  - PATTERNS            : Dict[alias, re.Pattern]  (pre-compiled word-boundary regex)
  - ALIAS_TO_CANONICAL  : Dict[alias, canonical]    (canonical normalization, point #5)
  - CANONICAL_DOMAIN    : Dict[canonical, domain]
  - CHART_ALIASES       : aliases that are real chart metrics (drives fast gate + LLM prompt)
  - FINANCIAL_ALIASES   : all financial-domain aliases (drives temporal.is_financial_metric_query)
  - CANONICAL_LIST      : canonical chart-metric names (single source for the LLM prompt)

Fuzzy / typo tolerance (point #4, #6): after the exact regex pass, a token-level
fuzzy scan (rapidfuzz if installed, else stdlib difflib) recovers misspellings
before escalating to the (expensive) LLM.
"""
import logging
import os
import re
from typing import Dict, List

import yaml

logger = logging.getLogger("hr_agent")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_METRICS_YAML = os.path.join(_BASE_DIR, "config", "metrics.yaml")

# Fail-open built-in defaults (mirror of config/metrics.yaml)
_DEFAULT_METRICS: List[dict] = [
    {"canonical": "revenue", "domain": "financial", "chart_metric": True, "aliases": ["revenue", "sales"]},
    {"canonical": "expense", "domain": "financial", "chart_metric": True, "aliases": ["expense"]},
    {"canonical": "profit", "domain": "financial", "chart_metric": True, "aliases": ["profit"]},
    {"canonical": "gross_profit", "domain": "financial", "chart_metric": True, "aliases": ["gross profit", "gross_profit", "grossprofit"]},
    {"canonical": "net_profit", "domain": "financial", "chart_metric": True, "aliases": ["net profit", "net_profit", "netprofit"]},
    {"canonical": "operating_income", "domain": "financial", "chart_metric": True, "aliases": ["operating income"]},
    {"canonical": "ebitda", "domain": "financial", "chart_metric": True, "aliases": ["ebitda"]},
    {"canonical": "cost", "domain": "financial", "chart_metric": True, "aliases": ["cost"]},
    {"canonical": "margin", "domain": "financial", "chart_metric": True, "aliases": ["margin"]},
    {"canonical": "income", "domain": "financial", "chart_metric": True, "aliases": ["income"]},
    {"canonical": "loss", "domain": "financial", "chart_metric": True, "aliases": ["loss"]},
    {"canonical": "budget", "domain": "financial", "chart_metric": True, "aliases": ["budget"]},
    {"canonical": "actual", "domain": "financial", "chart_metric": True, "aliases": ["actual"]},
    {"canonical": "profitability", "domain": "financial", "chart_metric": True, "aliases": ["profitability"]},
    {"canonical": "headcount", "domain": "hr", "chart_metric": True, "aliases": ["headcount", "head count", "head_count", "workforce", "staffing", "fte"]},
    {"canonical": "hires", "domain": "hr", "chart_metric": True, "aliases": ["hires", "hire", "hiring", "new hires", "new joiners", "joiners", "onboarding", "recruitment"]},
    {"canonical": "terminations", "domain": "hr", "chart_metric": True, "aliases": ["terminations", "termination", "resignations", "resignation"]},
    {"canonical": "attrition", "domain": "hr", "chart_metric": True, "aliases": ["attrition", "employee turnover", "staff turnover"]},
    {"canonical": "retention", "domain": "hr", "chart_metric": True, "aliases": ["retention", "retention_rate", "retention rate"]},
    {"canonical": "tenure", "domain": "hr", "chart_metric": True, "aliases": ["tenure"]},
    {"canonical": "leave", "domain": "hr", "chart_metric": True, "aliases": ["leave", "leave balance", "leave_balance", "leave taken", "absenteeism", "absence", "leave entitlement"]},
    {"canonical": "salary", "domain": "hr", "chart_metric": True, "aliases": ["salary", "basic_salary", "basic salary", "compensation", "remuneration", "wages", "payroll"]},
]

# Fuzzy backend: prefer rapidfuzz (production), fall back to stdlib difflib.
try:
    from rapidfuzz import fuzz as _rfuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _rfuzz = None
    _HAS_RAPIDFUZZ = False

FUZZY_THRESHOLD = 0.88


def _ratio(a: str, b: str) -> float:
    if _HAS_RAPIDFUZZ:
        return _rfuzz.ratio(a, b) / 100.0
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def _load_metrics() -> List[dict]:
    if not os.path.isfile(_METRICS_YAML):
        logger.warning("metrics.yaml not found at %s; using built-in defaults", _METRICS_YAML)
        return _DEFAULT_METRICS
    try:
        with open(_METRICS_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        metrics = (data or {}).get("metrics", [])
        if not isinstance(metrics, list) or not metrics:
            logger.warning("metrics.yaml has no metrics list; using built-in defaults")
            return _DEFAULT_METRICS
        return metrics
    except Exception as e:
        logger.warning("Failed to load metrics.yaml (%s); using built-in defaults", e)
        return _DEFAULT_METRICS


# Build pre-compiled structures ONCE at import.
_RAW = _load_metrics()

PATTERNS: Dict[str, "re.Pattern"] = {}
ALIAS_TO_CANONICAL: Dict[str, str] = {}
CANONICAL_DOMAIN: Dict[str, str] = {}
CHART_ALIASES: List[str] = []
FINANCIAL_ALIASES: List[str] = []
CANONICAL_LIST: List[str] = []

for _m in _RAW:
    _canon = _m["canonical"]
    _domain = _m.get("domain", "financial")
    _chart = bool(_m.get("chart_metric", True))
    _substring = bool(_m.get("substring", False))
    CANONICAL_DOMAIN[_canon] = _domain
    for _alias in _m.get("aliases", []):
        _a = str(_alias).lower().strip()
        if not _a:
            continue
        if _chart:
            if not _substring:
                PATTERNS[_a] = re.compile(r"\b" + re.escape(_a) + r"\b", re.IGNORECASE)
            ALIAS_TO_CANONICAL[_a] = _canon
            CHART_ALIASES.append(_a)
        if _domain == "financial":
            FINANCIAL_ALIASES.append(_a)
    if _chart:
        CANONICAL_LIST.append(_canon)


def normalize_to_canonical(token: str) -> str:
    return ALIAS_TO_CANONICAL.get(token.lower().strip(), token.lower().strip())


def extract_metrics_fast_gate(query: str) -> List[str]:
    q = query.lower()
    found: List[str] = []
    for _alias in sorted(PATTERNS.keys(), key=len, reverse=True):
        if PATTERNS[_alias].search(q) and ALIAS_TO_CANONICAL[_alias] not in found:
            found.append(ALIAS_TO_CANONICAL[_alias])
    if not found:
        found.extend(_fuzzy_scan(q))
    return found


def _fuzzy_scan(q: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9_]+", q)
    results: List[str] = []
    for _alias, _canon in ALIAS_TO_CANONICAL.items():
        if _canon in results:
            continue
        best = 0.0
        for _tok in tokens:
            if abs(len(_tok) - len(_alias)) > 2:
                continue
            r = _ratio(_alias, _tok)
            if r > best:
                best = r
        if best >= FUZZY_THRESHOLD and _canon not in results:
            results.append(_canon)
    return results


def financial_keywords() -> List[str]:
    return list(FINANCIAL_ALIASES)


def allowed_metrics_prompt_text() -> str:
    return ", ".join(sorted(set(CANONICAL_LIST)))
