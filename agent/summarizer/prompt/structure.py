from typing import Dict, List

from agent.summarizer.prompt.registry import ACTION_CATEGORIES, PERSONAL_DATA_CATEGORIES, ASSIST_REGISTRY


def _get_response_structure_guidance(intent: str, stage: str, default: str) -> str:
    """Lookup response structure guidance from registry by intent."""
    guidance = ASSIST_REGISTRY.get("response_structure", {}).get(intent, {})
    return guidance.get(stage, default)


def build_response_structure(category: str, has_data: bool, has_empty_result: bool,
                              action_oriented: bool, tone_context: Dict) -> str:
    """
    Enforce the Inform -> Assist -> Offer feedback three-stage response structure.
    Uses category (not hardcoded intent) for robust, scalable handling.
    """
    parts = []
    emotional = tone_context.get("emotional_state", "neutral")
    urgency = tone_context.get("urgency_level", "routine")
    intent = tone_context.get("intent", "")

    # STAGE 1: INFORM
    parts.append(
        "STAGE 1 - INFORM: Present the factual answer FIRST, before any policy context. "
        "Bold key numbers. Use tables for multiple records. "
        "Never bury the answer under preamble or policy text. "
        "Get straight to the point: the user asked a question, answer it immediately."
    )

    # STAGE 2: ASSIST
    if has_empty_result:
        stage2_guidance = _get_response_structure_guidance(
            intent,
            "stage2_empty",
            "STAGE 2 - ASSIST: No data found. Explain what you checked (specific entity, filter), "
            "then offer 2-3 NUMBERED next steps the user can choose from. Be specific. "
            "Example: 'I checked your 2025 leave records and found no matches. You could: "
            "(1) check all years, (2) verify your employee ID with HR, (3) ask me to search with a different filter. "
            "Which would you like?'"
        )
        parts.append(stage2_guidance)

    # Unified pending-slots handler — replaces hardcoded missing_date / missing_leave_type.
    # Any injector can produce a pending_slot; the summarizer asks generically from user_question.
    # When multiple slots are missing (e.g. both date and leave type for a leave application),
    # ask for ALL of them in a single turn rather than one at a time to avoid round-trips.
    pending_slots = tone_context.get("pending_slots", [])
    if pending_slots:
        # Deduplicate by (entity, field) in case injectors produced duplicates.
        seen_slots: set = set()
        unique_slots: List[Dict] = []
        for slot in pending_slots:
            key = (slot.get("entity", ""), slot.get("field", ""))
            if key not in seen_slots:
                seen_slots.add(key)
                unique_slots.append(slot)

        if len(unique_slots) == 1:
            slot = unique_slots[0]
            user_question = slot.get("user_question", "Could you provide more details?")
            entity = slot.get("entity", "the record")
            parts.append(
                f"STAGE 2 - ASSIST: The user wants to update {entity} but did not provide a required value. "
                f"After presenting any relevant context, ask for the missing information. "
                f"Example: '{user_question}'"
            )
        else:
            # Multiple missing slots — ask for all in one turn.
            questions = [s.get("user_question", "Could you provide more details?") for s in unique_slots]
            entities = sorted(set(s.get("entity", "the record") for s in unique_slots))
            parts.append(
                f"STAGE 2 - ASSIST: The user wants to update {', '.join(entities)} but did not provide "
                f"required values for {len(unique_slots)} fields. After presenting any relevant context, "
                f"ask for ALL missing information in a single turn (do not ask one at a time). "
                f"Example: 'I have your leave entitlement above. To apply, I need: "
                f"{'; '.join(questions)}'"
            )
    elif tone_context.get("chart_intent_clarification_needed"):
        # Ambiguous chart/dashboard request detected by LLM classifier
        reason = tone_context.get("chart_intent_clarification_reason", "your request could mean different things")
        parts.append(
            f"STAGE 2 - ASSIST: The user's chart request is ambiguous ({reason}). "
            f"Ask them to clarify EXACTLY what they want to see. "
            f"Offer 2-3 specific options as a numbered list. "
            f"Example: 'I can show you a single chart or a full dashboard. "
            f"Would you like: (1) just headcount by department, "
            f"(2) a full workforce overview with multiple charts, or "
            f"(3) just the key numbers without charts? Which works best for you?'"
        )
    elif has_data and (action_oriented or category in ACTION_CATEGORIES):
        stage2_guidance = _get_response_structure_guidance(
            intent,
            "stage2_has_data",
            "STAGE 2 - ASSIST: The user wants to take action or this is an action-oriented topic. "
            "After presenting the facts, suggest 1-3 relevant next actions they can take. "
            "Keep suggestions brief and tied to the data you just showed. "
            "Examples: 'Would you like me to help you apply for this leave?', "
            "'Shall I check your manager's approval status?', "
            "'I can draft an email to HR if you'd like.'"
        )
        parts.append(stage2_guidance)
    elif has_data and category in PERSONAL_DATA_CATEGORIES:
        parts.append(
            "STAGE 2 - ASSIST: After presenting personal data, suggest 1-2 relevant "
            "next actions. Example: 'Would you like me to check your leave history, "
            "or help you plan time off?'"
        )
    else:
        parts.append(
            "STAGE 2 - ASSIST: Even for informational queries, suggest how this "
            "information might be useful or what related action they might consider. "
            "Keep it brief and relevant."
        )

    # Workforce analytics override
    if category == "workforce_analytics":
        parts.append(
            "STAGE 1 - INFORM: Present the KPI result clearly. "
            "When multiple KPIs are available, use a markdown table with headers: KPI | Current Value | Target | Status. "
            "Bold the KPI names. Use clear status wording such as On Track, At Risk, or Off Track. "
            "If a KPI cannot be calculated from available data, say 'N/A - requires [data source]'. "
            "Then add a short 'Why it matters' line explaining the business impact."
        )

    # STAGE 3: OFFER FEEDBACK
    # If user response is ambiguous, override Stage 3 to force clarification
    if tone_context.get("is_ambiguous"):
        parts.append(
            "STAGE 3 - CLARIFICATION REQUIRED: The user's response ('yes', 'ok', 'sure') is ambiguous "
            "because the previous turn offered multiple options. "
            "You MUST ask which option they meant. Do NOT guess. "
            "Present the pending options as a numbered list and invite them to choose."
        )
        return "\n\n".join(parts)

    # STAGE 3: OFFER FEEDBACK
    if category == "emergency" or emotional in ("anxious", "distressed", "grieving") or urgency == "urgent":
        parts.append(
            "STAGE 3 - OFFER FEEDBACK: End by asking if this is what they needed "
            "and offering to escalate. 'Is this the right info? I can loop in HR now if needed.'"
        )
    elif action_oriented or category in ACTION_CATEGORIES:
        stage3_guidance = _get_response_structure_guidance(
            intent,
            "stage3",
            "STAGE 3 - OFFER FEEDBACK: End by offering to help with the next step. "
            "'Ready when you are - just let me know what you'd like to do next.'"
        )
        parts.append(stage3_guidance)
    else:
        parts.append(
            "STAGE 3 - OFFER FEEDBACK: End with an open invitation. "
            "'Let me know if you need anything else, or if something looks off.'"
        )

    return "\n\n".join(parts)
