from typing import Dict, Optional


def build_export_aware_guidance(wants_export: bool, export_url: str, metadata: Optional[Dict] = None, tone_context: Optional[Dict] = None) -> str:
    """Generate system prompt guidance when user explicitly requested Excel export.

    When wants_export=True and export_url is present, suppress full data walkthrough
    and chart generation in the LLM response. The file is the primary deliverable.
    """
    if wants_export and export_url:
        return (
            "EXCEL EXPORT GENERATED: An Excel workbook has been created for the user.\n"
            "CRITICAL LINK RULE: Use the placeholder {{EXPORT_URL}} exactly once in your response "
            "where you want the download link to appear. Example: 'You can download it here: {{EXPORT_URL}}'.\n"
            "Do NOT output any other markdown links, bracketed text (like [Download...]), or filenames. "
            "Do NOT describe the workbook contents (sheets, metadata, etc.). "
            "Keep your response concise and focused on the data. "
            "The system will replace {{EXPORT_URL}} with the actual clickable link."
        )
    return ""
