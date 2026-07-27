from typing import Dict

from agent.summarizer.prompt.registry import ACTION_CATEGORIES, PERSONAL_DATA_CATEGORIES


def build_response_structure(category: str, has_data: bool, has_empty_result: bool,
                              action_oriented: bool, tone_context: Dict) -> str:
    """
    Enforce the Inform -> Assist -> Offer feedback three-stage response structure.
    Uses category (not hardcoded intent) for robust, scalable handling.
    """
    parts = []
    emotional = tone_context.get("emotional_state", "neutral")
    urgency = tone_context.get("urgency_level", "routine")

    # STAGE 1: INFORM
    parts.append(
        "STAGE 1 - INFORM: Present the factual answer FIRST, before any policy context. "
        "Bold key numbers. Use tables for multiple records. "
        "Never bury the answer under preamble or policy text. "
        "Get straight to the point: the user asked a question, answer it immediately."
    )

    # STAGE 2: ASSIST
    if has_empty_result:
        parts.append(
            "STAGE 2 - ASSIST: No data found. Explain what you checked (specific entity, filter), "
            "then offer 2-3 NUMBERED next steps the user can choose from. Be specific. "
            "Example: 'I checked your 2025 leave records and found no matches. You could: "
            "(1) check all years, (2) verify your employee ID with HR, (3) ask me to search with a different filter. "
            "Which would you like?'"
        )
    elif has_data and (action_oriented or category in ACTION_CATEGORIES):
        parts.append(
            "STAGE 2 - ASSIST: The user wants to take action or this is an action-oriented topic. "
            "After presenting the facts, suggest 1-3 relevant next actions they can take. "
            "Keep suggestions brief and tied to the data you just showed. "
            "Examples: 'Would you like me to help you apply for this leave?', "
            "'Shall I check your manager's approval status?', "
            "'I can draft an email to HR if you'd like.'"
        )
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
        parts.append(
            "STAGE 3 - OFFER FEEDBACK: End by offering to help with the next step. "
            "'Ready when you are - just let me know what you'd like to do next.'"
        )
    else:
        parts.append(
            "STAGE 3 - OFFER FEEDBACK: End with an open invitation. "
            "'Let me know if you need anything else, or if something looks off.'"
        )

    return "\n\n".join(parts)
