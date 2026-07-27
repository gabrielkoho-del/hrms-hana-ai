from typing import Dict, List

from agent.summarizer.prompt.registry import ACTION_CATEGORIES


def build_tone_block(tone_context: Dict) -> List[str]:
    if not tone_context:
        return [
            "CRITICAL TONE RULE: Keep it conversational and friendly. Use 'you' and 'your'. Avoid legal/policy jargon."
        ]

    needs_empathy = tone_context.get("needs_empathy", False)
    urgency = tone_context.get("urgency_level", "routine")
    emotional = tone_context.get("emotional_state", "neutral")
    sensitivity = tone_context.get("topic_sensitivity", "low")
    action_oriented = tone_context.get("action_oriented", False)
    category = tone_context.get("intent_category", "policy_info")

    tone_rules = []

    # Empathy tier
    if needs_empathy or emotional in ("anxious", "distressed", "grieving"):
        tone_rules.append(
            "CRITICAL TONE RULE: The user is expressing distress or urgency about a personal matter. "
            "Lead with genuine empathy and reassurance BEFORE any policy details. "
            "Example: I am sorry you are dealing with this. Let me help you figure out the next steps. "
            "Never start with formal policy headers when the user is distressed."
        )
    elif urgency == "urgent":
        tone_rules.append(
            "CRITICAL TONE RULE: The user has an urgent need. Be direct and efficient, but warm. "
            "Lead with the most important actionable information. "
            "Acknowledge the urgency: I understand this is time-sensitive. Here is what you need to know right now."
        )
    elif emotional == "frustrated":
        tone_rules.append(
            "CRITICAL TONE RULE: The user seems frustrated. Acknowledge their frustration briefly, "
            "then offer concrete next steps. Avoid being defensive."
        )
    elif emotional == "confused":
        tone_rules.append(
            "CRITICAL TONE RULE: The user seems confused. Be extra clear and step-by-step. "
            "Break down complex information into simple, numbered steps."
        )
    elif emotional == "celebratory":
        tone_rules.append(
            "CRITICAL TONE RULE: The user is sharing good news. Match their positive energy. "
            "Be warm and congratulatory while still being accurate."
        )
    else:
        tone_rules.append(
            "CRITICAL TONE RULE: Keep it conversational and friendly. Use 'you' and 'your'. "
            "Avoid legal/policy jargon unless necessary."
        )

    # Sensitivity tier
    if sensitivity == "high":
        tone_rules.append(
            "SENSITIVITY RULE: This is a high-sensitivity topic (medical, family crisis, personal wellbeing). "
            "Never diagnose. Never give medical advice. Present options, not directives. "
            "Suggest (do not instruct) seeing a doctor or contacting HR. Be extra careful with tone."
        )
    elif sensitivity == "medium":
        tone_rules.append(
            "SENSITIVITY RULE: This is a medium-sensitivity topic (leave disputes, salary, performance). "
            "Be factual and fair. Present the user's data clearly. Offer escalation paths if they disagree."
        )

    # FEEDBACK CLOSING tier (Stage 3)
    if category == "emergency" or needs_empathy or emotional in ("anxious", "distressed", "grieving"):
        tone_rules.append(
            "FEEDBACK CLOSING RULE: End by explicitly asking if the information looks correct "
            "and offering to connect them with HR. Use: 'Does this match what you expected? "
            "If anything feels wrong, I can flag it for HR immediately.'"
        )
    elif urgency == "urgent":
        tone_rules.append(
            "FEEDBACK CLOSING RULE: End by confirming this is what they needed and offering "
            "to escalate. Use: 'Is this the right info? I can loop in your HR partner now if needed.'"
        )
    elif action_oriented or category in ACTION_CATEGORIES:
        tone_rules.append(
            "FEEDBACK CLOSING RULE: The user wants to take action. End by offering to help them "
            "complete the next step. Use: 'Ready to move forward? I can help you with the next step whenever you are.'"
        )
    else:
        tone_rules.append(
            "FEEDBACK CLOSING RULE: End with an open invitation. Use: 'Let me know if you need "
            "anything else, or if something looks off.'"
        )

    return tone_rules
