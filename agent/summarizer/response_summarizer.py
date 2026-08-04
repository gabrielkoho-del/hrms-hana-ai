import json
import re
import logging
from typing import Dict, Optional, List

from agent.output.excel_exporter import export_to_excel
from agent.integrations.llm_client import call_llm
from agent.dab.dab_response import format_dab_items_context

from agent.summarizer.prompt.registry import PERSONAL_DATA_CATEGORIES, ACTION_CATEGORIES, get_assist_suggestions
from agent.summarizer.prompt.tone import build_tone_block
from agent.summarizer.prompt.structure import build_response_structure
from agent.summarizer.prompt.data_rules import build_data_protocol
from agent.summarizer.prompt.formatting import build_formatting_protocol, build_zero_row_guidance
from agent.summarizer.prompt.export_guidance import build_export_aware_guidance
from agent.summarizer.validators.conflict import detect_conflicts
from agent.dab.dab_response import extract_items_with_meta

logger = logging.getLogger("hr_agent")

LARGE_RESULT_THRESHOLD = 100


def build_assist_block(category: str, has_data: bool, has_empty_result: bool,
                        action_context: str, tone_context: Dict) -> str:
    """
    Generate assist-stage guidance using the config-driven registry.
    Robust: new intents automatically map to existing categories.
    """
    if not has_data and not has_empty_result:
        return ""

    if has_empty_result:
        suggestions = get_assist_suggestions(category, False, True, action_context)
        if not suggestions:
            suggestions = [
                "Verify your information with HR",
                "Check if the data exists under a different filter",
                "Contact the relevant department for assistance",
            ]
        return (
            "ASSIST GUIDANCE: The database returned no records. "
            "Your job in Stage 2 is to explain what was checked and offer 2-3 specific next steps. "
            "Number them. Make them actionable. "
            f"Suggested next steps: {', '.join(suggestions)}"
        )

    # Has data
    suggestions = get_assist_suggestions(category, True, False, action_context)
    if not suggestions:
        suggestions = [
            "Would you like to explore related information?",
            "Need help with any follow-up action?",
            "Shall I export this data for your records?",
        ]

    if action_context:
        return (
            f"ASSIST GUIDANCE: The user likely wants to: {action_context}. "
            "In Stage 2, suggest 1-2 concrete next steps related to this action. "
            f"Suggested: {', '.join(suggestions[:2])}"
        )

    return (
        f"ASSIST GUIDANCE: In Stage 2, suggest 1-2 relevant next steps based on the data shown. "
        f"Suggested: {', '.join(suggestions[:2])}"
    )


def summarize_results(user_query, tool_results, conversation_history="", chart_config=None,
                        call_llm_fn=None, large_result_threshold=20,
                        tone_context=None, action_context="", export_url="",
                        wants_export=False, metadata=None, code_context=""):
    tone_context = tone_context or {}

    # Extract pending offers for ambiguous clarification
    pending_offers = tone_context.get("pending_offers", ())
    if tone_context.get("is_ambiguous") and pending_offers:
        logger.info("AMBIGUOUS_RESPONSE: injecting %d pending offers into prompt", len(pending_offers))

    context_parts = []
    total_row_count = 0
    rag_context = ""
    export_data = None
    has_empty_personal_data = False

    # Inject code context FIRST (before other context) so LLM sees mappings
    # when interpreting values like "A", "SL", "ENG" in tool results
    if code_context:
        context_parts.append(code_context)

    # ROBUST: Use category, not hardcoded intent names
    category = tone_context.get("intent_category", "policy_info")
    intent = tone_context.get("intent", "")
    action_oriented = tone_context.get("action_oriented", False)

    llm = call_llm_fn or call_llm

    # ═══════════════════════════════════════════════════════════════════════
    # EXTRACT RAW CHART ARTIFACT from tool_results (injected by executor)
    # Industry standard: LLM sees the actual markdown and embeds it naturally
    # ═══════════════════════════════════════════════════════════════════════
    # Make a shallow copy to avoid mutating the caller's dictionary
    tool_results = dict(tool_results)

    raw_chart_md = ""
    chart_format = ""
    if "__chart" in tool_results:
        raw_chart_md = tool_results["__chart"].get("result", "")
        chart_cfg = tool_results["__chart"].get("args", {}) or {}
        # Detect format from content so the LLM knows exactly what to expect
        if raw_chart_md.startswith("```mermaid"):
            chart_format = "MERMAID"
        elif "![Chart](" in raw_chart_md or "data:image/png;base64" in raw_chart_md:
            chart_format = "PNG_IMAGE"
        else:
            chart_format = "UNKNOWN"
        # Remove from tool_results so it doesn't get formatted as generic data
        del tool_results["__chart"]
        logger.info("CHART_DEBUG_SUMMARIZER: extracted raw_chart_md len=%d format=%s", 
                    len(raw_chart_md), chart_format)

    # Extract RAG context if present
    if "__rag_context" in tool_results:
        rag_context = tool_results["__rag_context"].get("result", "")
        del tool_results["__rag_context"]

    for tool_name, tool_output in tool_results.items():
        if tool_name.startswith("__"):
            continue

        if isinstance(tool_output, dict) and "error" in tool_output:
            error_msg = tool_output["error"]
            if not isinstance(error_msg, str):
                error_msg = str(error_msg)
            context_parts.append(f"[SYSTEM ERROR] {tool_name} failed: {error_msg}")
            logger.warning("Tool error in summarizer: %s -> %s", tool_name, error_msg[:200])
            continue

        if isinstance(tool_output, dict) and "result" in tool_output:
            raw_data = tool_output["result"]
            tool_args = tool_output.get("args", {})
        else:
            raw_data = tool_output
            tool_args = {}

        if isinstance(raw_data, str):
            try:
                data = json.loads(raw_data)
            except:
                context_parts.append(f"{tool_name}: {raw_data[:200]}")
                continue
        else:
            data = raw_data

        # HANA-style results arrive as list[dict] in tool_output["result"].
        # DAB-style results arrive as dict with "items"/"value"/"result".
        if isinstance(data, list):
            items = [item for item in data if isinstance(item, dict)]
            count = len(items)
            has_more = False
            end_cursor = None
            if count > 0:
                total_row_count = max(total_row_count, count)
                if count > large_result_threshold:
                    export_data = items
            item_lines = format_dab_items_context(tool_name, items, tool_args, has_more, end_cursor)
            context_parts.extend(item_lines)
            entity = tool_args.get("entity", "").lower()
            if count == 0 and any(k in entity for k in ("employee", "leave", "salary", "benefit")) and not tool_args.get("function"):
                has_empty_personal_data = True
            continue

        if not isinstance(data, dict):
            continue

        items, count, has_more, end_cursor = extract_items_with_meta(data)

        if count > 0:
            total_row_count = max(total_row_count, count)
            if count > large_result_threshold and isinstance(items, list) and all(isinstance(i, dict) for i in items):
                export_data = items
            item_lines = format_dab_items_context(tool_name, items, tool_args, has_more, end_cursor)
            context_parts.extend(item_lines)
        else:
            item_lines = format_dab_items_context(tool_name, items, tool_args, has_more, end_cursor)
            context_parts.extend(item_lines)
            entity = tool_args.get("entity", "").lower()
            if any(k in entity for k in ("employee", "leave", "salary", "benefit")):
                has_empty_personal_data = True

    context = "\n\n".join(context_parts)
    logger.info("Context length: %d chars, %d rows", len(context), total_row_count)

    # Generate Excel if large result (only if no chart-export already produced by executor)
    if not export_url and export_data and len(export_data) > 0:
        prefix = re.sub(r'[^\w]', '_', user_query[:30]).lower()
        export_url = export_to_excel(export_data, prefix=prefix or "export")
        logger.info("Excel export URL (large-result fallback): %s", export_url)

    # Large result notice
    large_result_prefix = ""
    if total_row_count > large_result_threshold:
        large_result_prefix = (
            "📊 **That's a lot of results!**\n\n"
            + f"I found **{total_row_count:,} matching records**, so I'm showing you the first {large_result_threshold} below. "
            + 'Need the full set? Just say **"export to Excel"** and I will send you the complete file.\n\n'
            + "Or you can narrow it down — for example:\n"
            + '• "Show only active employees"\n'
            + '• "Filter by Sales department"\n'
            + '• "Employees who joined in 2025"\n\n'
            + "---\n\n"
        )

    # Export mode: suppress chart and large-result notice (file is primary deliverable)
    if wants_export and export_url:
        raw_chart_md = ""
        chart_config = None
        large_result_prefix = ""

    # Conflict detection
    conflict_report = detect_conflicts(rag_context, context, llm)
    if conflict_report != "NO_CONFLICTS":
        logger.warning("Policy-data conflicts detected: %s", conflict_report[:200])

    # Compute flags for three-stage structure
    has_data = total_row_count > 0
    has_empty_result = has_empty_personal_data or (total_row_count == 0 and category in PERSONAL_DATA_CATEGORIES)

    # ═════════════════════════════════════════════════════════════════════════════
    # BUILD COMPACT SYSTEM PROMPT — Modular, prioritized, with 3-stage structure
    # ═════════════════════════════════════════════════════════════════════════════
    system_parts = []

    # 1. IDENTITY
    system_parts.append("You are a helpful HR colleague. Speak naturally, like you're explaining to a coworker over chat.")

    # 2. RESPONSE STRUCTURE (3-stage principle — uses category, not hardcoded intent)
    system_parts.append(build_response_structure(
        category, has_data, has_empty_result, action_oriented, tone_context
    ))

    # 3. TONE BLOCK (includes feedback closing)
    tone_rules = build_tone_block(tone_context)
    system_parts.extend(tone_rules)

    # 4. DATA PROTOCOL (uses category for source-of-truth rules)
    data_rules = build_data_protocol(rag_context, conflict_report, export_url, total_row_count, large_result_threshold, tone_context, wants_export=wants_export)
    system_parts.extend(data_rules)

    # 5. ZERO-ROW GUIDANCE
    system_parts.append(build_zero_row_guidance())

    # 6. FORMATTING PROTOCOL
    system_parts.append(build_formatting_protocol())

    # 7. EXPORT-AWARE GUIDANCE
    export_guidance = build_export_aware_guidance(wants_export, export_url, metadata=metadata, tone_context=tone_context)
    if export_guidance:
        system_parts.append(export_guidance)

    # 8. CHART ARTIFACT — inject raw markdown directly into system prompt
    # The LLM sees the actual chart block and must embed it verbatim
    if raw_chart_md:
        # Determine human-readable format description for the LLM
        if chart_format == "MERMAID":
            format_desc = "a Mermaid diagram (```mermaid block)"
            format_instructions = (
                "This is a Mermaid diagram. Copy the ```mermaid fence and all contents EXACTLY. "
                "Do NOT modify the Mermaid syntax."
            )
        elif chart_format == "PNG_IMAGE":
            format_desc = "a PNG image (rendered via markdown image link)"
            format_instructions = (
                "This is a PNG image link like ![Chart](URL). Copy the markdown image syntax EXACTLY. "
                "Do NOT convert it to Mermaid or any other format. "
                "The image will render automatically in the chat interface."
            )
        else:
            format_desc = "a chart"
            format_instructions = "Copy the chart block EXACTLY as provided."

        system_parts.append(
            "CHART ARTIFACT (already generated — you MUST embed this EXACTLY in your response):\n"
            "\n"
            + raw_chart_md + "\n\n"
            "CRITICAL INSTRUCTIONS — READ CAREFULLY:\n"
            "1. The chart above is " + format_desc + ". It is ALREADY generated and ready to display.\n"
            "2. You MUST copy it VERBATIM into your response. Do NOT modify a single character.\n"
            "3. " + format_instructions + "\n"
            "4. Place it AFTER your Stage 1 (Inform) text and BEFORE Stage 3 (Offer feedback).\n"
            "5. NEVER create your own chart, diagram, or Mermaid block. Use ONLY the one provided above.\n"
            "6. NEVER say 'chart shown above' or '[Insert chart here]' or 'here is a diagram'. "
            "The actual artifact must be present verbatim.\n"
            "7. After the chart, add 2-3 bullet points describing key patterns from the data."
        )
    elif chart_config and isinstance(chart_config, dict):
        # Fallback: planner provided config but no raw markdown was generated
        # (e.g., chart generation was blocked by privacy or failed)
        chart_type = chart_config.get("type", "bar")
        x_col = chart_config.get("x_column", "")
        y_col = chart_config.get("y_column", "")
        chart_title = chart_config.get("title", "")
        system_parts.append(
            f"CHART CONFIG: A {chart_type} chart was planned but not generated. "
            f"Title: '{chart_title}'. X-axis: '{x_col}'. Y-axis: '{y_col}'. "
            "If you reference a chart, note that it could not be generated and present the data table instead."
        )
    elif tone_context.get("chart_eligible") is False:
        system_parts.append(
            "NO CHART RULE: This query is not eligible for charts (personal data, single value, small dataset, or action-oriented). "
            "Use text, bold numbers, or markdown tables only. Do not reference any chart."
        )

    # Build user prompt
    user_prompt_parts = []
    if conversation_history:
        user_prompt_parts.append("Previous conversation:\n" + conversation_history + "\n")
    if rag_context:
        user_prompt_parts.append(rag_context)
    user_prompt_parts.append('Question: "' + user_query + '"\n\nFacts:\n' + context + '\n\n')
    if conflict_report != "NO_CONFLICTS":
        user_prompt_parts.append(
            "NOTE: When Facts show a personal entitlement or balance that conflicts with any policy document above, "
            "use ONLY the value from Facts. Do not mention the conflicting policy number.\n\n"
        )
    if has_empty_personal_data and category in PERSONAL_DATA_CATEGORIES:
        user_prompt_parts.append(
            "CRITICAL INSTRUCTION: The database returned NO personal records for this query. "
            "Do NOT use the HR policy documents above to infer, fabricate, or substitute the missing personal data. "
            "Policy documents are for REFERENCE ONLY. "
            "State clearly that the requested personal information was not found in the database, and suggest next steps.\n\n"
        )

    # Add assist guidance using registry (not hardcoded if/elif)
    assist_guidance = build_assist_block(category, has_data, has_empty_result, action_context, tone_context)
    if assist_guidance:
        user_prompt_parts.append(assist_guidance + "\n\n")

    user_prompt_parts.append("Your answer:")
    user_prompt = "\n".join(user_prompt_parts)

    system = "\n".join(system_parts)

    logger.info("System prompt: %d chars, User prompt: %d chars", len(system), len(user_prompt))

    choice = llm(system, user_prompt, temperature=0.3, max_tokens=8192)
    llm_text = choice["message"]["content"] if choice else context
    logger.info("CHART_DEBUG_SUMMARIZER: llm_text_len=%d has_mermaid=%s",
                len(llm_text), "```mermaid" in llm_text)

    if large_result_prefix:
        llm_text = large_result_prefix + llm_text

    # ═══════════════════════════════════════════════════════════════════════
    # EXPORT URL INJECTION — placeholder replacement
    # ═══════════════════════════════════════════════════════════════════════
    if export_url:
        export_link_md = f"[Download Excel Export]({export_url})"
        llm_text = llm_text.replace("{{EXPORT_URL}}", export_link_md)

    return llm_text
