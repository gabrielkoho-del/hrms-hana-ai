"""
intent_classifier.py
Lightweight LLM-based intent & tone classification.
Industry standard: single structured call, ~200 tokens.
Includes intent_category for robust, scalable 3-stage response handling.
Added ambiguous affirmation detection when prior turn
offered multiple options. Forces clarification instead of guessing.
"""
import json
import logging
import os
import re
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass

import yaml

from agent.integrations.llm_client import call_llm

from functools import lru_cache
logger = logging.getLogger("hr_agent")


# ═══════════════════════════════════════════════════════════════════════
# FINANCE CONFIG — loaded from YAML at startup
# ═══════════════════════════════════════════════════════════════════════

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FINANCE_CONFIG_PATH = os.path.join(_BASE_DIR, "config", "finance_config.yaml")

def _load_finance_config() -> Dict:
    """Load finance configuration from YAML file."""
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

# Finance keywords loaded from config (narrowed to avoid false positives)
_FINANCE_KEYWORDS: Set[str] = set(_FINANCE_CONFIG.get("finance_keywords", [
    "gl account", "general ledger", "bank details",
    "exchange rate", "currency conversion",
    "salary cost", "compensation cost", "benefits cost",
    "overhead", "personnel cost", "labor cost", "fte cost",
]))

# Table-name prefixes used to discover finance tables from HANA schema
_FINANCE_TABLE_PREFIXES: List[str] = _FINANCE_CONFIG.get("finance_table_prefixes", [
    "FAGL", "BKPF", "BSEG", "SKA", "CSK", "T001", "TCUR",
])

# Explicit finance table names (fallback if HANA schema discovery fails)
_FINANCE_FALLBACK_TABLES: Set[str] = set(_FINANCE_CONFIG.get("finance_table_names", [
    "FAGLFLEXA", "BKPF", "BSEG", "SKA1", "SKAT",
    "CSKS", "CSKT", "T001", "TCURC", "TCURR",
]))

# Schema discovery cache
_discovered_finance_tables: Optional[Set[str]] = None
_discovered_tables_timestamp: float = 0.0
_SCHEMA_CACHE_TTL = _FINANCE_CONFIG.get("schema_cache_ttl_seconds", 3600)
_schema_discovery_tool = _FINANCE_CONFIG.get("schema_discovery_tool", "hana_list_tables")


@dataclass
class IntentResult:
    intent: str                      # e.g., "leave_request", "policy_question", "profile_lookup", "data_discovery"
    intent_category: str             # "personal_data" | "aggregate_data" | "policy_info" | "data_discovery" | "action_request" | "emergency" | "grievance" | "greeting"
    data_scope: str                  # "individual" | "aggregate" | "none" — determines chart eligibility
    chart_eligible: bool             # True ONLY if aggregate data and not personal/action/emergency
    urgency_level: str               # "routine" | "time_sensitive" | "urgent" | "distressed"
    emotional_state: str             # "neutral" | "anxious" | "frustrated" | "celebratory" | "grieving" | "confused"
    topic_sensitivity: str           # "low" | "medium" | "high"
    needs_empathy: bool
    confidence: float                # 0.0-1.0
    action_oriented: bool            # True if user wants to DO something
    wants_export: bool = False       # True if user explicitly asks to export/download data
    is_ambiguous: bool = False       # True if user's affirmation is ambiguous (multiple prior options)
    finance_query: bool = False      # True if query involves SAP FI/CO finance data


# Intent -> Category mapping (deterministic, no LLM needed for this)
# This is the ROBUST layer: new intents map to categories automatically
INTENT_CATEGORY_MAP = {
    # Personal data lookups (individual scope, NEVER chart)
    "leave_request": "personal_data",
    "leave_balance": "personal_data",
    "profile_lookup": "personal_data",
    "salary_question": "personal_data",
    "medical": "personal_data",
    "emergency": "emergency",
    "complaint": "grievance",
    # Action requests (individual scope, NEVER chart)
    "resignation": "action_request",
    "benefits_enrollment": "action_request",
    "profile_update": "action_request",
    "training_request": "action_request",
    # Aggregate / org data (aggregate scope, CHART ELIGIBLE)
    "org_hierarchy": "aggregate_data",
    "department_count": "aggregate_data",
    "gender_distribution": "aggregate_data",
    "hiring_trend": "aggregate_data",
    "salary_analysis": "aggregate_data",
    "turnover_analysis": "aggregate_data",
    "age_distribution": "aggregate_data",
    "performance_distribution": "aggregate_data",
    "leave_analysis": "aggregate_data",
    "mc_trend": "aggregate_data",
    "attendance_analysis": "aggregate_data",
    "recruitment_funnel": "aggregate_data",
    "compensation_ratio": "aggregate_data",
    "headcount_budget": "aggregate_data",
    "diversity_hiring": "aggregate_data",
    "engagement_scores": "aggregate_data",
    "skills_gap": "aggregate_data",
    "payroll_distribution": "aggregate_data",
    "attrition_risk": "aggregate_data",
    "revenue_analysis": "aggregate_data",
    # Workforce analytics (aggregate scope, CHART ELIGIBLE)
    "workforce_productivity": "workforce_analytics",
    "revenue_per_employee": "workforce_analytics",
    "absenteeism_rate": "workforce_analytics",
    "overtime_cost": "workforce_analytics",
    # Finance / GL data (aggregate scope, CHART ELIGIBLE)
    "finance_gl_analysis": "aggregate_data",
    "finance_cost_analysis": "aggregate_data",
    "finance_currency": "aggregate_data",
    "finance_budget": "aggregate_data",
    "finance_revenue_analysis": "aggregate_data",
    "finance_profitability": "aggregate_data",
    "finance_cashflow": "aggregate_data",
    "finance_balance_sheet": "aggregate_data",
    "finance_financial_statement": "aggregate_data",
    "finance_cost_center": "aggregate_data",
    "finance_payroll_analysis": "aggregate_data",
    "finance_accounts_payable": "aggregate_data",
    "finance_accounts_receivable": "aggregate_data",
    "finance_invoice_analysis": "aggregate_data",
    "finance_vendor_analysis": "aggregate_data",
    "finance_budget_variance": "aggregate_data",
    "finance_forecast": "aggregate_data",
    # Policy / info (no data scope, NEVER chart)
    "policy_question": "policy_info",
    "general_hr": "policy_info",
    "data_discovery": "data_discovery",
    # Greeting
    "greeting": "greeting",
    "smalltalk": "greeting",
}

# Categories that trigger personal-data source-of-truth rules
PERSONAL_DATA_CATEGORIES = {"personal_data", "emergency", "grievance", "action_request"}

# Categories that trigger action-oriented assist stage
ACTION_CATEGORIES = {"action_request", "emergency", "grievance"}

# Chart eligibility rules (Guideline: Privacy First + Context-Aware)
CHART_ELIGIBLE_CATEGORIES = {"aggregate_data", "workforce_analytics"}
NEVER_CHART_CATEGORIES = {"personal_data", "emergency", "grievance", "action_request", "greeting"}

# Data scope inference from intent keywords
AGGREGATE_KEYWORDS = ("count", "average", "avg", "sum", "total", "distribution", "breakdown",
                      "trend", "rate", "analysis", "by department", "per department", "per team",
                      "gender", "age", "salary range", "turnover", "attrition", "hiring",
                      "headcount", "budget", "actual", "funnel", "engagement", "skills",
                      "compensation", "compa", "ratio", "payroll", "overtime", "absenteeism",
                      "time-to-fill", "source", "effectiveness", "utilization",
                      "revenue per employee", "hr-to-employee ratio", "workforce productivity",
                      "productivity index", "absenteeism rate", "overtime cost",
                      "revenue", "revenue analysis", "total revenue", "revenue for",
                      "profit", "loss", "net income", "cash flow", "cashflow",
                      "balance sheet", "financial statement", "income statement",
                      "p&l", "pnl", "profitability", "margin",
                      "cost center", "cost centre", "controlling", "overhead",
                      "budget variance", "forecast", "financial forecast",
                      "invoice", "invoices", "vendor", "vendors", "billing",
                      "accounts payable", "accounts receivable", "account payable", "account receivable",
                      "fico", "fi data", "co data", "sap fi", "sap co",
                      "financial reporting", "finance report", "bank details")
INDIVIDUAL_KEYWORDS = ("my ", "i ", "me ", "myself", "john", "jane", "who is", "profile of",
                       "salary of", "leave of", "balance of", "entitlement of")

# ── FOLLOW-UP AFFIRMATIVE DETECTION ──
_AFFIRMATIVE_WORDS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "pls",
                      "go ahead", "do it", "send it", "export it", "download it", "yes please",
                      "yes pls", "yess", "yas", " affirmative", "confirm"}
_EXPORT_OFFER_KEYWORDS = ("export", "excel", "spreadsheet", "xlsx", "workbook", "download",
                          "save as", "send me the file", "get a file")
_MULTI_OPTION_KEYWORDS = ("or would you prefer", "or", "instead", "which would you like",
                           "choose one", "would you like me to", "shall i", "do you want")

# SAP FI/CO tables available in HANA for finance queries
_HANA_FINANCE_TABLES: Set[str] = {
    "FAGLFLEXA", "BKPF", "BSEG", "SKA1", "SKAT",
    "CSKS", "CSKT", "T001", "TCURC", "TCURR",
}

# Finance domain keywords (narrowed to avoid false positives from generic business terms)
_FINANCE_KEYWORDS = {
    # GL / accounting
    "gl account", "general ledger", "bank details", "chart of accounts",
    "accounting", "account payable", "account receivable", "accounts payable", "accounts receivable",
    # Revenue / P&L / balance sheet / cash flow
    "revenue", "sales revenue", "total revenue", "revenue analysis",
    "profit", "loss", "net income", "operating income", "ebitda",
    "profitability", "margin", "gross margin", "net margin",
    "cash flow", "cashflow", "operating cash flow", "free cash flow",
    "balance sheet", "assets", "liabilities", "equity",
    "financial statement", "income statement", "p&l", "pnl",
    # Cost / budget / controlling
    "cost center", "cost centre", "controlling", "overhead",
    "budget", "budget variance", "forecast", "financial forecast",
    "salary cost", "compensation cost", "benefits cost",
    "personnel cost", "labor cost", "fte cost", "headcount cost",
    # Currency / banking / vendors / invoices
    "exchange rate", "currency conversion", "forex",
    "vendor", "vendors", "invoice", "invoices", "billing",
    # FI/CO references
    "fico", "fi data", "co data", "sap fi", "sap co",
    "financial reporting", "finance report",
}

_FULL_INFO_PATTERNS = [
    re.compile(r"\b(pull|get|show|give|fetch|send)\s+(me\s+)?(my\s+)?(full|complete|all|entire)\s+(the\s+)?(info|information|details|record|records|data|profile)\b", re.I),
    re.compile(r"\b(my\s+full|my\s+complete|all\s+my|full\s+profile|complete\s+profile)\b", re.I),
    re.compile(r"\b(full|complete|all)\s+the\s+(record|records|info|information|details)\b", re.I),
    re.compile(r"\b(full|complete|all)\s+(record|records|info|information|details)\b", re.I),
    re.compile(r"\b(everything|raw\s+data|all\s+fields|all\s+columns)\s+(on\s+file|in\s+your\s+system|you\s+have)\b", re.I),
]


def initialize_finance_tables(hana_client: Any = None) -> None:
    """Discover finance tables from HANA schema at startup.

    Queries HANA for tables matching finance prefixes, then caches the result.
    If HANA is unavailable, falls back to the hardcoded fallback table list.
    Called from agentic_executor.py at startup.
    """
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

        # Filter tables by finance prefixes
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

        # ── Startup validation: warn on configured tables missing from discovery ──
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
    if _discovered_finance_tables is not None:
        cache_age = time.time() - _discovered_tables_timestamp
        if cache_age < _SCHEMA_CACHE_TTL:
            return _discovered_finance_tables
        # Cache expired, try rediscovery
        logger.info("Finance table cache expired (%.0fs), re-discovery recommended", cache_age)
    return _FINANCE_FALLBACK_TABLES


def _detect_finance_query(query: str, finance_tables: Optional[Set[str]] = None) -> bool:
    """Detect if query involves SAP FI/CO finance data from HANA tables."""
    q = query.lower()
    tables = finance_tables if finance_tables is not None else _get_finance_tables()
    for table in tables:
        if table.lower() in q:
            return True
    for kw in _FINANCE_KEYWORDS:
        if kw in q:
            return True
    return False


def _match_patterns(query: str, patterns: list) -> bool:
    q = query.strip()
    return any(p.search(q) for p in patterns)


def _is_short_affirmative(query: str) -> bool:
    """Detect very short affirmative responses (1-3 words)."""
    q = query.strip().lower().rstrip("!?.,")
    words = q.split()
    if len(words) > 3:
        return False
    return any(w in _AFFIRMATIVE_WORDS for w in words) or q in _AFFIRMATIVE_WORDS


def _conversation_has_export_offer(history: str) -> bool:
    if not history:
        return False
    return any(kw in history.lower() for kw in _EXPORT_OFFER_KEYWORDS)


def _conversation_has_multi_option_offer(history: str) -> bool:
    if not history:
        return False
    return any(kw in history.lower() for kw in _MULTI_OPTION_KEYWORDS)


_CLASSIFIER_SYSTEM = """You are an intent classifier for an HR AI assistant. 
Analyze the user's query and classify it into structured categories.

 Return ONLY a JSON object with these exact keys:
  - intent: One of [greeting, smalltalk, leave_request, leave_balance, policy_question, profile_lookup, salary_question, org_hierarchy, emergency, medical, complaint, resignation, benefits_enrollment, profile_update, training_request, general_hr, department_count, gender_distribution, hiring_trend, salary_analysis, turnover_analysis, age_distribution, performance_distribution, leave_analysis, mc_trend, attendance_analysis, recruitment_funnel, compensation_ratio, headcount_budget, diversity_hiring, engagement_scores, skills_gap, payroll_distribution, attrition_risk, revenue_analysis, workforce_productivity, revenue_per_employee, absenteeism_rate, overtime_cost, finance_gl_analysis, finance_cost_analysis, finance_currency, finance_budget, finance_revenue_analysis, finance_profitability, finance_cashflow, finance_balance_sheet, finance_financial_statement, finance_cost_center, finance_payroll_analysis, finance_accounts_payable, finance_accounts_receivable, finance_invoice_analysis, finance_vendor_analysis, finance_budget_variance, finance_forecast, data_discovery]
- data_scope: One of [individual, aggregate, none]. "individual" = about a specific person (me, John, my profile). "aggregate" = about groups, departments, trends, distributions. "none" = no data needed (policy, greeting, action steps).
- chart_eligible: boolean. TRUE only if the query is about aggregate data (group comparisons, distributions, trends, proportions) AND not about a specific person. FALSE for personal lookups, single values, action requests, policy questions, greetings.
- urgency_level: One of [routine, time_sensitive, urgent, distressed]
- emotional_state: One of [neutral, anxious, frustrated, celebratory, grieving, confused]
- topic_sensitivity: One of [low, medium, high]
- needs_empathy: boolean (true if user expresses distress, anxiety, urgency about personal matters)
- confidence: number 0.0-1.0
- action_oriented: boolean (true if the user wants to TAKE ACTION: apply, request, change, escalate, submit, book, cancel, update, resign, enroll. False if they just want to KNOW: check, see, find out, what is, how many, who is)
- wants_export: boolean (true if the user explicitly asks to export, download, save to Excel, spreadsheet, or get a file with the data)

Rules:
- "distressed" = user mentions death, accident, hospital, severe illness, panic, "don't know what to do"
- "urgent" = user needs something today/now, mentions deadlines, "emergency leave"
- "time_sensitive" = needs action within days, mentions specific dates
- "routine" = general questions, lookups, no time pressure
- "high" topic_sensitivity = medical, mental health, family crisis, harassment, termination, resignation
- "medium" = leave disputes, salary issues, performance concerns, benefits
- "low" = directory lookups, general policy questions, org chart queries, training info
  - data_scope = "aggregate" when user asks about: counts per department, gender distribution, hiring trends, average salary by level, turnover rate, age distribution, performance ratings, leave analysis, MC trends, headcount, recruitment funnel, engagement scores, skills gap, total revenue, revenue by period, revenue analysis, etc.
  - Finance/admin queries such as general ledger, cost center, budget, currency, exchange rate, invoice, vendor, payroll cost, accounts payable/receivable, profitability, cash flow, balance sheet, financial statements, forecast, and FI/CO reporting are aggregate_data queries, NOT data_discovery.
  - Workforce productivity metrics such as revenue per employee, hr-to-employee ratio, absenteeism rate, and overtime cost are aggregate_data/workforce_analytics queries, NOT data_discovery.
 - "How many X are in table Y?", "Count X in Y", or "Total X in Y" where Y is a database table is an AGGREGATE query with intent=aggregate_data or a finance intent like finance_gl_analysis/finance_budget. It is NOT data_discovery.
- Data_discovery is ONLY when the user asks: "what data do you have?", "what's available?", "show me what you can access", "what tables exist?", "what schemas are there?" — questions about system capability, not data retrieval.
- Questions asking for specific counts, totals, sums, averages, or actual data values from known tables/entities are aggregate_data, not data_discovery.
- data_scope = "individual" when user asks about: my leave, John's salary, my profile, who is my manager, my balance, my department, my team.
- data_scope = "none" for policy questions, how-to steps, greetings, small talk.
- chart_eligible = TRUE when data_scope is "aggregate" AND the result would have multiple rows/categories to compare. FALSE when data_scope is "individual" or "none".
- action_oriented = True when user says: "apply for leave", "request time off", "change my address", "update my profile", "escalate this", "book a meeting", "submit resignation", "enroll in benefits", "file a complaint"
- action_oriented = False when user says: "what is my balance", "how many days", "show me", "who is my manager", "what is the policy", "tell me about", "how many employees per department"
- wants_export = True when user says: "export to Excel", "download this data", "save as spreadsheet", "send me the file", "give me the data in Excel", "workbook", "xlsx"
- wants_export = False for casual queries that don't explicitly request a file export

FOLLOW-UP RESOLUTION (context-dependent intent):
- If the user's query is very short ("yes", "no", "ok", "sure", "please",
  "nope", "nah", "go ahead", "do it"), examine the RECENT CONTEXT to resolve
  what the user is responding to.
- If the Assistant previously offered to EXPORT data ("export to Excel",
  "download", "save as spreadsheet"), "yes"/"ok"/"sure"/"please"/"go ahead"
  means the user ACCEPTS the export offer. Set wants_export=True and preserve
  the original intent/data_scope from the previous turn.
- If the Assistant offered to SHOW MORE DETAILS ("break down by department",
  "drill down", "filter by"), "yes"/"ok"/"sure" means the user wants the
  drill-down. Preserve the original intent and adjust parameters.
- "no"/"nope"/"nah" means the user DECLINES the offer. Set action_oriented=False
  and map intent to "general_hr".
- For follow-ups like "by department", "per team", "break it down", resolve
  against the previous query intent and set data_scope="aggregate".
- If the context does not clarify the user's intent, set confidence below 0.5
  so the system can ask for clarification.
"""

@lru_cache(maxsize=128)
def classify_intent(user_query: str, conversation_history: str = "",
                    pending_offers: tuple = ()) -> IntentResult:
    """
    Classify user intent and tone. Single LLM call, ~200 tokens.
    Returns IntentResult with intent_category mapped deterministically.
    """
    history_snippet = conversation_history[:200] if conversation_history else "None"
    prompt = f'Query: "{user_query}"\nRecent context: "{history_snippet}"\n\nClassify this query. Return JSON only.'

    try:
        # Initialize is_ambiguous early so it's always defined even if exceptions occur
        is_ambiguous = False

        choice = call_llm(_CLASSIFIER_SYSTEM, prompt, temperature=0.0, max_tokens=300)
        content = choice.get("message", {}).get("content", "{}") if choice else "{}"

        # Extract JSON from potential markdown fences
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

        # ROBUST: Map intent to category deterministically
        # If intent not in map, infer from action_oriented + personal keywords
        category = INTENT_CATEGORY_MAP.get(intent)
        if not category:
            if result.get("action_oriented", False):
                category = "action_request"
            elif intent == "data_discovery":
                category = "data_discovery"
            elif any(k in intent for k in ("salary", "profile", "leave", "benefit", "medical", "emergency")):
                category = "personal_data"
            elif any(k in intent for k in ("workforce", "productivity", "revenue_per_employee", "absenteeism_rate", "overtime_cost")):
                category = "workforce_analytics"
            elif any(k in intent for k in ("finance", "revenue", "budget", "cost", "currency", "exchange", "fx", "gl", "invoice", "vendor", "payroll", "payable", "receivable", "profit", "cashflow", "balance", "forecast", "fico", "controlling")):
                category = "aggregate_data"
            else:
                category = "policy_info"

        # Determine data_scope and chart_eligibility
        data_scope = result.get("data_scope", "")
        if not data_scope:
            query_lower = user_query.lower()
            if any(k in query_lower for k in AGGREGATE_KEYWORDS) or category == "aggregate_data":
                data_scope = "aggregate"
            elif any(k in query_lower for k in INDIVIDUAL_KEYWORDS) or category in ("personal_data", "emergency", "grievance", "action_request"):
                data_scope = "individual"
            else:
                data_scope = "none"

        chart_eligible = result.get("chart_eligible", False)
        if not isinstance(chart_eligible, bool):
            chart_eligible = (category in CHART_ELIGIBLE_CATEGORIES and data_scope == "aggregate")

        # wants_export: prefer LLM result, fallback to keyword detection
        wants_export = result.get("wants_export", False)
        if not isinstance(wants_export, bool):
            wants_export = False
        if not wants_export:
            export_keywords = ("export", "download", "save", "excel", "spreadsheet",
                               "xlsx", "workbook", "file", "send me", "give me the data")
            wants_export = any(kw in user_query.lower() for kw in export_keywords)

        if not wants_export and category in ("personal_data", "action_request", "emergency", "grievance"):
            if _match_patterns(user_query, _FULL_INFO_PATTERNS):
                wants_export = True
                logger.info("FULL_INFO_DETECTED: implicit export from query '%s'", user_query)

        return IntentResult(
            intent=intent,
            intent_category=category,
            data_scope=data_scope,
            chart_eligible=chart_eligible,
            urgency_level=result.get("urgency_level", "routine"),
            emotional_state=result.get("emotional_state", "neutral"),
            topic_sensitivity=result.get("topic_sensitivity", "low"),
            needs_empathy=result.get("needs_empathy", False),
            confidence=result.get("confidence", 0.5),
            action_oriented=result.get("action_oriented", False),
            wants_export=wants_export,
            is_ambiguous=is_ambiguous,
            finance_query=_detect_finance_query(user_query),
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
            is_ambiguous=False,
            finance_query=False,
        )