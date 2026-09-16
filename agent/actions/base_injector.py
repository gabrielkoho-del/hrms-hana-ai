"""Base injector utilities for action step injection.

Provides shared logic for:
- Entity config loading from YAML
- Action keyword detection
- Schema field extraction
- Step validation
- $ref chain building
- LLM-based field extraction fallback

This is the foundation for the unified action framework.
"""
import asyncio
import json
import logging
import re
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from agent.core.config_loader import get_action_keywords_for_tenant

logger = logging.getLogger("hr_agent")


# =============================================================================
# Config Loading
# =============================================================================

def _load_entity_actions_config() -> Dict[str, Any]:
    """Load entity_actions config from entity_actions.yaml."""
    import os
    import yaml
    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "config")
    )
    path = os.path.join(config_dir, "entity_actions.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("entity_actions", {})


@lru_cache(maxsize=1)
def get_entity_config(entity_name: str) -> Optional[Dict[str, Any]]:
    """Get action config for a specific entity (cached)."""
    configs = _load_entity_actions_config()
    return configs.get(entity_name)


# =============================================================================
# Keyword Detection
# =============================================================================

def is_action_request(
    user_query: str,
    action_keywords: Optional[tuple] = None,
    tenant_id: Optional[str] = None,
) -> bool:
    """Check if user query contains action keywords."""
    if action_keywords is None:
        action_keywords = get_action_keywords_for_tenant(tenant_id or "default")
    q = user_query.lower()
    return any(kw in q for kw in action_keywords)


def is_update_intent(
    user_query: str,
    entity_config: Dict[str, Any],
    action_keywords: Optional[tuple] = None,
    tenant_id: Optional[str] = None,
) -> bool:
    """Check if query is specifically an update/edit intent for an entity.

    Args:
        user_query: The user's query
        entity_config: Entity config from entity_actions.yaml
        action_keywords: Override action keywords (otherwise loaded from config)
        tenant_id: Tenant ID for action keyword loading
    """
    if action_keywords is None:
        action_keywords = get_action_keywords_for_tenant(tenant_id or "default")

    q = user_query.lower()
    update_words = {"update", "edit", "change", "modify", "set", "correct", "change to", "update to"}
    entity_name = entity_config.get("_entity_name", "").lower()

    # Check if query mentions the entity (e.g., "update my profile", "edit employee_general")
    mentions_entity = entity_name in q or _entity_mentioned(q, entity_name)

    # Check for action keywords
    has_action = any(kw in q for kw in action_keywords)

    # Check for update-specific words
    has_update_word = any(kw in q for kw in update_words)

    return mentions_entity and (has_action or has_update_word)


def _entity_mentioned(query: str, entity_name: str) -> bool:
    """Check if entity is mentioned in query via common variations."""
    # Handle employee_general variations
    if entity_name == "employee_general":
        variations = ["profile", "my profile", "my information", "my info", "personal info", "personal information"]
        return any(v in query for v in variations)
    return entity_name in query


# =============================================================================
# Schema Utilities
# =============================================================================

def get_entity_fields_from_schema(
    cached_schema: Dict,
    entity_name: str
) -> List[str]:
    """Extract field names for an entity from cached DAB schema."""
    entity = cached_schema.get(entity_name, {})
    if not isinstance(entity, dict):
        return []
    fields = entity.get("fields", entity.get("columns", []))
    if isinstance(fields, list):
        return [f.get("name") or f.get("column_name") or str(f) for f in fields if isinstance(f, dict)]
    return list(entity.keys())


def get_pk_fields_from_schema(
    cached_schema: Dict,
    entity_name: str
) -> List[str]:
    """Extract primary key field names from cached DAB schema."""
    entity = cached_schema.get(entity_name, {})
    if not isinstance(entity, dict):
        return []
    fields = entity.get("fields", entity.get("columns", []))
    if not isinstance(fields, list):
        return []
    pk_fields = []
    for f in fields:
        if isinstance(f, dict) and f.get("primary-key"):
            pk_fields.append(f.get("name", ""))
    return pk_fields


# =============================================================================
# Field Value Parsing
# =============================================================================

# =============================================================================
# Field Value Parsing — handles both "field: value" and "update X to value"
# =============================================================================

def parse_field_from_query(
    user_query: str,
    field_name: str,
    field_patterns: Optional[Dict[str, List[str]]] = None
) -> Optional[Any]:
    """Parse a field value from user query using pattern matching.

    Supports two patterns:
    1. "field: value" — primary pattern with colon delimiter
    2. "update field to value" — preposition pattern (update X to Y)

    Args:
        user_query: Lowercased user query
        field_name: Field to extract
        field_patterns: Optional override patterns

    Returns:
        Extracted value or None if not found
    """
    if field_patterns is None:
        field_patterns = DEFAULT_FIELD_PATTERNS

    patterns = field_patterns.get(field_name, [])
    for pattern in patterns:
        m = re.search(pattern, user_query, re.IGNORECASE)
        if m:
            captured = m.group(1)
            # Guard against optional capture group returning None
            if captured is None:
                return None
            return captured.strip('.,!?"\'')

    # Fallback: try "update <field> to <value>" pattern
    value = _extract_value_from_preposition(user_query, field_name)
    if value is not None:
        return value

    return None


def _extract_value_from_preposition(
    user_query: str,
    field_name: str
) -> Optional[str]:
    """Extract value from 'update <field> to <value>' or 'change <field> to <value>' pattern.

    This handles queries like:
    - "update marital status to married"
    - "change email to new@example.com"
    - "update my phone number to 0123456789"
    """
    q = user_query.lower()

    # Build patterns for different action verbs
    action_verbs = ["update", "change", "set", "edit", "correct"]
    field_pattern = field_name.lower().replace("_", r"\s+")

    for verb in action_verbs:
        # Pattern: "<verb> <field_pattern> to <value>"
        # e.g., "update marital status to married"
        #       "change email to x@y.com"
        pattern = rf'{verb}\s+{field_pattern}\s+to\s+(\S+)'
        m = re.search(pattern, q)
        if m:
            return m.group(1).strip('.,!?"\'')

        # Pattern: "<field_pattern> to <value>" (no verb)
        # e.g., "marital status to married"
        pattern = rf'{field_pattern}\s+to\s+(\S+)'
        m = re.search(pattern, q)
        if m:
            return m.group(1).strip('.,!?"\'')

    return None


def resolve_field_value(
    field_name: str,
    raw_value: str,
    tenant_id: str,
) -> str:
    """Resolve a human-readable value to a code using codesetup.

    For coded fields (marital_status, gender, confirmation_status, etc.),
    converts descriptions to codes:
    - "married" → "M"
    - "single" → "S"

    For non-coded fields, returns the raw value unchanged.

    Args:
        field_name: Field name (e.g., "marital_status")
        raw_value: Human value (e.g., "married")
        tenant_id: Tenant for codesetup lookup

    Returns:
        Code value if resolved, otherwise the original value
    """
    # Map field names to codesetup type names
    CODE_TYPE_MAP = {
        "marital_status": "MARITAL_STATUS",
        "gender": "GENDER",
        "confirmation_status": "CONFIRMATION_STATUS",
        "employee_status": "STATUS",
        "employment_category": "EMPLOYMENT_CATEGORY",
    }

    code_type = CODE_TYPE_MAP.get(field_name)
    if not code_type:
        return raw_value

    try:
        from agent.dab.code_resolver import CodeResolver
        resolver = CodeResolver(tenant_id)
        # Forward lookup: "married" → code
        matches = resolver.resolve_description(raw_value)
        for matched_type, code in matches:
            if matched_type.upper() == code_type.upper():
                return code
    except Exception:
        pass

    return raw_value


def parse_date_from_query(
    user_query: str,
    field_descriptions: List[str]
) -> Optional[str]:
    """Parse a date value from user query.

    Args:
        user_query: User query
        field_descriptions: List of field names/desc to look for

    Returns:
        ISO date string (yyyy-mm-dd) or None
    """
    q = user_query.lower()
    date_pattern = r'(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})'

    for desc in field_descriptions:
        if desc.lower() not in q:
            continue
        # Look for date near the field name
        idx = q.find(desc.lower())
        if idx >= 0:
            search_range = q[idx:idx+100]
            m = re.search(date_pattern, search_range)
            if m:
                return _normalize_date(m.group(1))
            # Also try standalone patterns
            m = re.search(r'(\d{4}-\d{2}-\d{2})', q)
            if m:
                return m.group(1)

    return None


def _normalize_date(date_str: str) -> str:
    """Convert various date formats to yyyy-mm-dd."""
    date_str = date_str.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    # Try month name format
    try:
        return datetime.strptime(date_str, "%d %B %Y").strftime("%Y-%m-%d")
    except ValueError:
        pass
    try:
        return datetime.strptime(date_str, "%d %b %Y").strftime("%Y-%m-%d")
    except ValueError:
        pass
    return date_str


# =============================================================================
# LLM-Based Field Extraction -- Agentic fallback for ambiguous cases
# Called when regex patterns miss and we need semantic understanding.
# =============================================================================

_FIELD_EXTRACTION_SYSTEM_PROMPT = """You are a field extraction assistant for an HR self-service AI assistant.

Given a user message and a list of allowed fields for an entity, extract the field name and
the value the user wants to set. If the user mentions a field but provides no value, indicate
that clarification is needed.

Coded fields and their values (use these for resolution):
- marital_status: "married"→"M", "single"→"S", "divorced"→"D", "widowed"→"W", "separated"→"SP"
- gender: "male"→"M", "female"→"F"
- confirmation_status: "confirmed"→"C", "probation"→"P", "terminated"→"T"

Return ONLY JSON with this exact schema:
{
  "field": "field_name or null",
  "value": "resolved_value or null",
  "needs_clarification": true|false,
  "clarification_needed_for": "field_name or null",
  "reason": "explanation when needs_clarification is true"
}

Rules:
- If the field is mentioned AND a value is provided → set field and value (resolve codes)
- If the field is mentioned but NO value provided → needs_clarification=true, set clarification_needed_for
- If no field mentioned at all → field=null, value=null, needs_clarification=false
- Resolve human descriptions to codes: "married"→"M", "male"→"M", etc.
- For date values, return ISO format (yyyy-mm-dd)
- Keep $ref chains as-is: do not resolve {"$ref": "..."} — those are filled by the executor

Examples:
- "update marital status to married" → {"field": "marital_status", "value": "M", "needs_clarification": false, "clarification_needed_for": null, "reason": null}
- "change my phone number" → {"field": "mobile_no", "value": null, "needs_clarification": true, "clarification_needed_for": "mobile_no", "reason": "no value provided"}
- "my email is test@example.com" → {"field": "personal_email", "value": "test@example.com", "needs_clarification": false, "clarification_needed_for": null, "reason": null}
- "show me headcount" → {"field": null, "value": null, "needs_clarification": false, "clarification_needed_for": null, "reason": null}"""


def _llm_extract_field_value(
    user_query: str,
    allowed_fields: List[str],
    conversation_history: str = "",
    pending_field: Optional[str] = None,
    tenant_id: str = "default",
) -> Dict[str, Any]:
    """Extract field+value using LLM when regex patterns miss.

    This is the agentic fallback: instead of silently returning nothing,
    we call the LLM with schema context so it can handle ambiguous cases
    like "I'm married now" (no keyword "marital status" mentioned directly).

    Args:
        user_query: The user's current query
        allowed_fields: Self-updateable fields for this entity
        conversation_history: Prior turns for context
        pending_field: If set, the LLM should resolve this specific field
        tenant_id: Tenant (unused here, reserved for future code lookup)

    Returns:
        Dict with keys: field, value, needs_clarification, clarification_needed_for, reason
    """
    from agent.integrations.llm_client import call_llm

    history_snippet = conversation_history[-500:] if conversation_history else "None"

    pending_context = (
        f"NOTE: The user previously mentioned '{pending_field}' but did not provide a value. "
        f"Their current reply should be treated as the value for '{pending_field}'.\n\n"
        if pending_field else ""
    )

    user_prompt = (
        f"{pending_context}"
        f"Recent conversation:\n{history_snippet}\n\n"
        f"Allowed fields: {', '.join(allowed_fields)}\n"
        f"User message: {user_query}\n\n"
        f"Return JSON only."
    )

    default_result = {
        "field": None,
        "value": None,
        "needs_clarification": False,
        "clarification_needed_for": None,
        "reason": None,
    }

    try:
        choice = call_llm(
            _FIELD_EXTRACTION_SYSTEM_PROMPT,
            user_prompt,
            temperature=0.1,
            max_tokens=300,
            tier="planner",
        )
        if not choice:
            logger.warning("_llm_extract_field_value: empty LLM response")
            return default_result

        content = choice.get("message", {}).get("content", "{}").strip()
        # Strip markdown code fences
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        result = json.loads(content)

        return {
            "field": result.get("field"),
            "value": result.get("value"),
            "needs_clarification": bool(result.get("needs_clarification", False)),
            "clarification_needed_for": result.get("clarification_needed_for"),
            "reason": result.get("reason"),
        }

    except json.JSONDecodeError as exc:
        logger.warning("_llm_extract_field_value: JSON parse failed (%s): %s", exc, content[:200])
        return default_result
    except Exception as exc:
        logger.warning("_llm_extract_field_value: LLM call failed (%s)", exc)
        return default_result


# =============================================================================
# Default Field Patterns for employee_general
# =============================================================================

DEFAULT_FIELD_PATTERNS: Dict[str, List[str]] = {
    # Contact fields - anchor to distinguish from emergency_*
    "mobile_no": [
        r'(?:mobile|phone|mobile\s*no)[:\s]+(\S+)',
        r'mobile\s*(?:no|number)?[:\s]*(\d[\d\s-]{7,})',
    ],
    "contact_no": [
        r'(?<!emergency\s)(?:contact|contact\s*no)[:\s]+(\S+)',
    ],
    "personal_email": [
        r'personal\s*email[:\s]+(\S+@\S+)',
    ],
    "mail_address": [
        r'(?:mail(?:ing)?\s+)?address(?:[:\s]+(.+))?(?=\.|,|$|and|emergency)',
    ],
    "emergency_name": [
        r'emergency\s+name[:\s]+(.+?)(?:\.|,|$)',
        r'emergency\s+contact\s+name[:\s]+(.+?)(?:\.|,|$)',
    ],
    "emergency_contact_no": [
        r'emergency\s+(?:contact\s+)?(?:number|phone)[:\s]+(\S+)',
    ],
    "emergency_relation": [
        r'emergency\s+relation(?:ship)?[:\s]+(.+?)(?:\.|,|$)',
    ],
    "emergency_email": [
        r'emergency\s+email[:\s]+(\S+@\S+)',
    ],
    "emergency_address": [
        r'emergency\s+address(?:[:\s]+(.+))?(?=\.|,|$)',
    ],
    "passport_no": [
        r'passport\s+(?:no|number)[:\s]+(\S+)',
    ],
    "passport_expiry": [
        r'passport\s+expir(?:y|ing)[:\s]+(\S+)',
    ],
    "visa_no": [
        r'visa\s+(?:no|number)?[:\s]+(\S+)',
    ],
    "work_permit_no": [
        r'work\s+permit\s+(?:no|number)[:\s]+(\S+)',
    ],
    "work_permit_expiry": [
        r'work\s+permit\s+expir(?:y|ing)[:\s]+(\S+)',
    ],
    "national_id": [
        r'national\s+id(?:entification)?[:\s]+(\S+)',
        r'\bIC[:\s]+(\S+)',
    ],
    "nickname": [
        r'nickname[:\s]+(.+?)(?:\.|,|$)',
    ],
    "personal_title": [
        r'(?:title|salutation)[:\s]+(.+?)(?:\.|,|$)',
    ],
    "name_ext": [
        r'(?:suffix|name\s*ext)[:\s]+(.+?)(?:\.|,|$)',
    ],
    "forward_email": [
        r'forward\s+email[:\s]+(\S+@\S+)',
    ],
    "driving_licence": [
        r'driving\s+licence(?:\s+no)?[:\s]+(\S+)',
    ],
    "place_of_birth": [
        r'place\s+of\s+birth[:\s]+(.+?)(?:\.|,|$)',
    ],
    "state_of_birth": [
        r'state\s+of\s+birth[:\s]+(.+?)(?:\.|,|$)',
    ],
    "marital_status": [
        r'marital\s+status[:\s]+(\w+)',
    ],
}


# =============================================================================
# Step Validation
# =============================================================================

def validate_steps(
    steps: List[Dict],
    cached_tools: Optional[List[Dict]]
) -> bool:
    """Validate injected steps against DAB tool schemas.

    Returns True if all steps are valid or we can't determine (fail-open).
    """
    if not cached_tools:
        return True

    try:
        from agent.dab.validation import validate_dab_args
        for step in steps:
            tool = step.get("tool")
            args = step.get("args", {})
            _, err = validate_dab_args(tool, args, cached_tools)
            if err:
                logger.warning("ACTION_INJECTOR: step validation failed for %s: %s", tool, err)
                return False
    except Exception as exc:
        logger.warning("ACTION_INJECTOR: validation error: %s", exc)

    return True


# =============================================================================
# Pending-slot value plausibility check
# =============================================================================
# Shared by agentic_executor._handle_pending_update and
# simple_update_strategy.inject_simple_update to decide whether a user reply
# is a bare value for a pending field (e.g. "married") or a new request
# (e.g. "I want to apply 1 day leave"). Without this guard, a stale pending
# action hijacks an unrelated query by binding the entire message as the
# field value.

_QUESTION_STARTERS = frozenset({
    "what", "when", "where", "who", "why", "how", "which", "whose", "whom",
    "can", "could", "do", "does", "did", "is", "are", "was", "were",
    "show", "list", "tell", "find", "get", "give", "any", "anyway",
})


def is_plausible_slot_value(reply: str) -> bool:
    """Heuristic: is the reply plausibly a bare value for a pending field?

    Excludes question-shaped replies, long multi-clause requests, and
    empty input. The LLM extraction (with pending_field context) is the
    final arbiter, but this gate prevents obvious non-values from being
    bound as field values.

    Examples that return True (bare values):
      "married"
      "M"
      "0123456789"

    Examples that return False (new requests / questions):
      "I want to apply 1 day leave"   (7 words — too long)
      "What's my leave balance?"      (question)
      "Can I update my email?"        (question-shaped)
    """
    q = reply.strip().lower()
    if not q or len(q) < 2:
        return False
    if "?" in q:
        return False
    first_word = q.split()[0] if q.split() else ""
    if first_word in _QUESTION_STARTERS:
        return False
    if len(q.split()) > 6:
        return False
    return True


# =============================================================================
# Import lru_cache for get_entity_config
# =============================================================================
