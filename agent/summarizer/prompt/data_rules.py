from typing import Dict, List, Optional

from agent.summarizer.prompt.registry import PERSONAL_DATA_CATEGORIES


def build_data_protocol(rag_context: str, conflict_report: str, export_url: str,
                        total_row_count: int, large_result_threshold: int,
                        tone_context: Optional[Dict] = None, wants_export: bool = False) -> List[str]:
    tone_context = tone_context or {}
    category = tone_context.get("intent_category", "policy_info")
    action_oriented = tone_context.get("action_oriented", False)

    rules = [
        "Answer using ONLY the provided facts. NEVER cite document names, section numbers, or page numbers.",
        "NEVER start with 'According to...' or 'As per Section...'",
        "Prioritize practical steps and what the user should DO, not just listing rules verbatim.",
        "Use bold headers for key sections, not formal bullet-point dumps.",
    ]

    if rag_context:
        rules.append(
            "HR POLICY DOCUMENTS retrieved. Paraphrase in your own words. "
            "Do NOT mention handbook name, section numbers, or page numbers."
        )

    # Category-aware source-of-truth rule
    if rag_context and category in PERSONAL_DATA_CATEGORIES:
        rules.append(
            "CRITICAL SOURCE OF TRUTH RULE: For personal employee queries, the database record is the SOURCE OF TRUTH. "
            "HR policy documents are for REFERENCE ONLY. "
            "Never substitute policy text for the employee's actual personal data."
        )

    if conflict_report != "NO_CONFLICTS":
        rules.append(
            "POLICY-DATA CONFLICTS: " + conflict_report + "\n"
            "CRITICAL: The employee's database record is the SOURCE OF TRUTH. "
            "When policy conflicts with actual record, use ONLY the database value. "
            "Do NOT mention the conflicting policy number. Simply state their actual entitlement."
        )

    if export_url and not wants_export:
        rules.append(
            "An Excel export was generated. Mention the link naturally: "
            "I also put the full data in an Excel file for you."
        )

    if total_row_count > large_result_threshold:
        rules.append(
            f"CRITICAL: Many records returned. Display the first {large_result_threshold} rows as a markdown table "
            "with ALL columns present. Do NOT summarize as a single sentence."
        )

    # Assist rule for data-present cases
    if total_row_count > 0 and (category in PERSONAL_DATA_CATEGORIES or action_oriented):
        rules.append(
            "ASSIST RULE: After presenting data, suggest 1-2 relevant next actions. "
            "Keep suggestions brief and relevant to the category. "
            "This is Stage 2 of your response - do not skip it."
        )

    return rules
