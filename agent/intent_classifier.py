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
import re
from typing import Dict, Any
from dataclasses import dataclass

from agent.llm_client import call_llm

from functools import lru_cache
logger = logging.getLogger("hr_agent")


@dataclass
class IntentResult:
    intent: str                      # e.g., "leave_request", "policy_question", "profile_lookup"
    intent_category: str             # "personal_data" | "aggregate_data" | "policy_info" | "action_request" | "emergency" | "grievance" | "greeting"
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
    # Policy / info (no data scope, NEVER chart)
    "policy_question": "policy_info",
    "general_hr": "policy_info",
    # Greeting
    "greeting": "greeting",
    "smalltalk": "greeting",
}

# Categories that trigger personal-data source-of-truth rules
PERSONAL_DATA_CATEGORIES = {"personal_data", "emergency", "grievance", "action_request"}

# Categories that trigger action-oriented assist stage
ACTION_CATEGORIES = {"action_request", "emergency", "grievance"}

# Chart eligibility rules (Guideline: Privacy First + Context-Aware)
CHART_ELIGIBLE_CATEGORIES = {"aggregate_data"}
NEVER_CHART_CATEGORIES = {"personal_data", "emergency", "grievance", "action_request", "greeting"}

# Data scope inference from intent keywords
AGGREGATE_KEYWORDS = ("count", "average", "avg", "sum", "total", "distribution", "breakdown",
                      "trend", "rate", "analysis", "by department", "per department", "per team",
                      "gender", "age", "salary range", "turnover", "attrition", "hiring",
                      "headcount", "budget", "actual", "funnel", "engagement", "skills",
                      "compensation", "compa", "ratio", "payroll", "overtime", "absenteeism",
                      "time-to-fill", "source", "effectiveness", "utilization")
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

_FULL_INFO_PATTERNS = [
    re.compile(r"\b(pull|get|show|give|fetch|send)\s+(me\s+)?(my\s+)?(full|complete|all|entire)\s+(the\s+)?(info|information|details|record|records|data|profile)\b", re.I),
    re.compile(r"\b(my\s+full|my\s+complete|all\s+my|full\s+profile|complete\s+profile)\b", re.I),
    re.compile(r"\b(full|complete|all)\s+the\s+(record|records|info|information|details)\b", re.I),
    re.compile(r"\b(full|complete|all)\s+(record|records|info|information|details)\b", re.I),
    re.compile(r"\b(everything|raw\s+data|all\s+fields|all\s+columns)\s+(on\s+file|in\s+your\s+system|you\s+have)\b", re.I),
]


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
- intent: One of [greeting, smalltalk, leave_request, leave_balance, policy_question, profile_lookup, salary_question, org_hierarchy, emergency, medical, complaint, resignation, benefits_enrollment, profile_update, training_request, general_hr, department_count, gender_distribution, hiring_trend, salary_analysis, turnover_analysis, age_distribution, performance_distribution, leave_analysis, mc_trend, attendance_analysis, recruitment_funnel, compensation_ratio, headcount_budget, diversity_hiring, engagement_scores, skills_gap, payroll_distribution, attrition_risk]
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
- data_scope = "aggregate" when user asks about: counts per department, gender distribution, hiring trends, average salary by level, turnover rate, age distribution, performance ratings, leave analysis, MC trends, headcount, recruitment funnel, engagement scores, skills gap, etc.
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
            # Fallback inference: never fails
            if result.get("action_oriented", False):
                category = "action_request"
            elif any(k in intent for k in ("salary", "profile", "leave", "benefit", "medical", "emergency")):
                category = "personal_data"
            elif any(k in intent for k in ("org", "department", "team", "count", "aggregate", "distribution", "trend", "analysis", "rate", "funnel", "engagement", "skills", "compensation", "payroll", "attrition", "headcount", "budget", "hiring")):
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
        )