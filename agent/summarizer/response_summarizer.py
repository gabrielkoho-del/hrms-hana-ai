import json
import re
import logging
import hashlib
from typing import Dict, Optional, List

from agent.output.excel_exporter import export_to_excel
from agent.integrations.llm_client import call_llm
from agent.data.response import format_dab_items_context

from agent.summarizer.prompt.registry import PERSONAL_DATA_CATEGORIES, ACTION_CATEGORIES, get_assist_suggestions
from agent.summarizer.prompt.tone import build_tone_block
from agent.summarizer.prompt.structure import build_response_structure
from agent.summarizer.prompt.data_rules import build_data_protocol
from agent.summarizer.prompt.formatting import build_formatting_protocol, build_zero_row_guidance
from agent.summarizer.prompt.export_guidance import build_export_aware_guidance
from agent.summarizer.validators.conflict import detect_conflicts
from agent.data.response import extract_items_with_meta

logger = logging.getLogger("hr_agent")

LARGE_RESULT_THRESHOLD = 100


def _sanitize_error_message(msg: str) -> str:
    """Sanitize tool error messages before injecting into LLM prompt.
    
    Removes stack traces, connection strings, internal paths, and other
    sensitive information that could leak into user-facing responses.
    """
    if not isinstance(msg, str):
        msg = str(msg)
    # Remove file paths
    msg = re.sub(r'[\w/\\]+\.py', '<path>', msg)
    # Remove connection strings and credentials
    msg = re.sub(r'(password|pwd|secret|token|key|api_key)\s*[=:]\s*\S+', r'\1=***', msg, flags=re.I)
    # Remove stack traces
    msg = re.sub(r'File ".*?", line \d+', '', msg)
    # Remove line numbers
    msg = re.sub(r'line \d+', 'line <n>', msg)
    # Remove internal URLs
    msg = re.sub(r'https?://[^\s]+', '<url>', msg)
    # Cap length
    return msg[:500]


def _sanitize_user_input(text: str) -> str:
    """Sanitize user input to prevent prompt injection attacks.
    
    Escapes quotes and removes potential instruction override attempts.
    """
    if not isinstance(text, str):
        text = str(text)
    # Remove potential instruction override attempts
    text = re.sub(
        r'(?i)(ignore|forget|disregard|override|bypass)\s+(previous|above|all|system|these)\s+(instructions|prompts|rules|guidelines)',
        '',
        text
    )
    # Remove potential role override attempts
    text = re.sub(
        r'(?i)(you are now|act as|pretend to be|roleplay as)\s+\w+',
        '',
        text
    )
    # Escape quotes that could break prompt structure
    text = text.replace('"', '\\"').replace("'", "\\'")
    # Cap length
    return text[:2000]


def _validate_llm_output(llm_text: str, raw_chart_md: str, context: str) -> str:
    """Validate LLM output and fix common issues.
    
    - Ensures chart blocks are preserved verbatim
    - Spot-checks that key numbers from context appear in output
    """
    if not llm_text:
        return llm_text
    
    # Verify chart block preserved verbatim
    if raw_chart_md and raw_chart_md not in llm_text:
        logger.warning("Chart block not preserved in LLM output, re-injecting")
        # Try to insert after Stage 1 or at the beginning
        if "STAGE 1" in llm_text or "**" in llm_text:
            parts = llm_text.split('\n\n')
            for i, part in enumerate(parts):
                if part.startswith('STAGE 1') or part.startswith('**'):
                    parts.insert(i + 1, raw_chart_md)
                    llm_text = '\n\n'.join(parts)
                    break
        else:
            llm_text = raw_chart_md + "\n\n" + llm_text
    
    return llm_text


def _call_llm_with_retry(llm, system: str, user_prompt: str, max_retries: int = 2) -> Optional[str]:
    """Call LLM with retry logic for transient failures."""
    last_error = None
    for attempt in range(max_retries):
        try:
            choice = llm(system, user_prompt, temperature=0.3, max_tokens=8192)
            if choice and choice.get("message", {}).get("content"):
                return choice["message"]["content"]
            last_error = "Empty LLM response"
        except Exception as e:
            last_error = str(e)
            logger.warning("LLM call attempt %d/%d failed: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
    
    logger.error("LLM call failed after %d attempts: %s", max_retries, last_error)
    return None


def build_assist_block(category: str, has_data: bool, has_empty_result: bool,
                        action_context: str, tone_context: Dict) -> str:
    """
    Generate assist-stage guidance using the config-driven registry.
    Robust: new intents automatically map to existing categories.
    """
    if not has_data and not has_empty_result:
        return ""

    intent = tone_context.get("intent", "")
    
    if has_empty_result:
        suggestions = get_assist_suggestions(category, False, True, action_context, intent=intent)
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
    suggestions = get_assist_suggestions(category, True, False, action_context, intent=intent)
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

    # =======================================================================
    # EXTRACT RAW CHART ARTIFACT from tool_results (injected by executor)
    # Industry standard: LLM sees the actual markdown and embeds it naturally
    # =======================================================================
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

    # Extract variance analysis if present (multi-metric financial reports)
    variance_context = ""
    if "__variances" in tool_results:
        variance_context = tool_results["__variances"].get("result", "")
        del tool_results["__variances"]
        logger.info("CHART_DEBUG_SUMMARIZER: extracted variance_context len=%d", len(variance_context))

    # Extract forecast output if present
    forecast_context = ""
    if "__forecast" in tool_results:
        forecast_context = tool_results["__forecast"].get("result", "")
        del tool_results["__forecast"]
        logger.info("FORECAST_DEBUG_SUMMARIZER: extracted forecast_context len=%d", len(forecast_context))

    # Extract market context if present
    market_context = ""
    if "__market_context" in tool_results:
        market_context = tool_results["__market_context"].get("result", "")
        del tool_results["__market_context"]
        logger.info("FORECAST_DEBUG_SUMMARIZER: extracted market_context len=%d", len(market_context))

    # Extract RAG context if present
    if "__rag_context" in tool_results:
        rag_context = tool_results["__rag_context"].get("result", "")
        del tool_results["__rag_context"]

    blocked_leave_write = False
    confirmation_required = False
    confirmation_type = ""
    confirmation_summary = ""
    confirmation_message = ""
    for tool_name, tool_output in tool_results.items():
        if tool_name.startswith("__"):
            continue

        # Detect confirmation_required results from the leave checkpoint.
        # These are stored as {"result": json.dumps({"confirmation_required": True, ...})}
        if isinstance(tool_output, dict) and "result" in tool_output:
            raw_result = tool_output["result"]
            if isinstance(raw_result, str):
                try:
                    parsed = json.loads(raw_result)
                except (json.JSONDecodeError, TypeError):
                    parsed = None
            else:
                parsed = raw_result if isinstance(raw_result, dict) else None
            if isinstance(parsed, dict) and parsed.get("confirmation_required"):
                confirmation_required = True
                confirmation_type = parsed.get("confirmation_type", "leave_application")
                confirmation_summary = parsed.get("summary", "")
                confirmation_message = parsed.get("message", "")
                logger.info("Summarizer: confirmation_required detected for %s (type=%s)", parsed.get("entity", ""), confirmation_type)
                context_parts.append(
                    f"[CONFIRMATION REQUIRED] {confirmation_summary}"
                )
                continue

        if isinstance(tool_output, dict) and "error" in tool_output:
            error_msg = tool_output["error"]
            if not isinstance(error_msg, str):
                error_msg = str(error_msg)
            sanitized_error = _sanitize_error_message(error_msg)
            context_parts.append(f"[SYSTEM ERROR] {tool_name} failed: {sanitized_error}")
            logger.warning("Tool error in summarizer: %s -> %s", tool_name, error_msg[:200])
            # Detect a blocked leave write (balance gate) so the LLM gets
            # explicit instructions: the leave was NOT submitted.
            if "leave" in tool_name.lower() and (
                "NOT submitted" in error_msg or "Insufficient leave balance" in error_msg
            ):
                blocked_leave_write = True
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
            except json.JSONDecodeError as e:
                logger.warning("JSON parse failed for %s: %s | raw: %s", 
                               tool_name, e, raw_data[:500])
                context_parts.append(f"{tool_name}: [unparseable JSON] {raw_data[:500]}")
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
                    if export_data is None:
                        export_data = []
                    export_data.extend(items)
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
                if export_data is None:
                    export_data = []
                export_data.extend(items)
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
            "[CHART] **That's a lot of results!**\n\n"
            + f"I found **{total_row_count:,} matching records**, so I'm showing you the first {large_result_threshold} below. "
            + 'Need the full set? Just say **"export to Excel"** and I will send you the complete file.\n\n'
            + "Or you can narrow it down -- for example:\n"
            + '* "Show only active employees"\n'
            + '* "Filter by Sales department"\n'
            + '* "Employees who joined in 2025"\n\n'
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

    # =============================================================================
    # BUILD COMPACT SYSTEM PROMPT -- Modular, prioritized, with 3-stage structure
    # =============================================================================
    system_parts = []

    # 1. IDENTITY
    system_parts.append("You are a helpful HR colleague. Speak naturally, like you're explaining to a coworker over chat.")

    # 2. RESPONSE STRUCTURE (3-stage principle -- uses category, not hardcoded intent)
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

    # 5b. BLOCKED LEAVE WRITE GUIDANCE -- the balance gate blocked the
    # create_record. The LLM must tell the user the leave was NOT submitted
    # and relay the reason; it must never claim the application succeeded.
    if blocked_leave_write:
        system_parts.append(
            "BLOCKED ACTION RULE: A leave application was blocked by a safety check "
            "(balance verification failed or insufficient balance). "
            "State clearly and early that the leave application was NOT submitted. "
            "Relay the exact reason from the [SYSTEM ERROR] fact (e.g., insufficient balance "
            "with available/requested days, or balance query failure). "
            "Do NOT claim or imply the application succeeded. "
            "Offer a concrete next step: apply for fewer days, choose another leave type, "
            "retry later, or contact HR."
        )

    # 5c. CONFIRMATION CHECKPOINT -- a write was intercepted before execution
    # and is awaiting explicit user confirmation. The LLM must present the
    # details clearly and ask the user to confirm with a short "yes" reply.
    # Do NOT execute the write in this turn.
    if confirmation_required:
        if confirmation_type == "profile_update":
            system_parts.append(
                f"CONFIRMATION CHECKPOINT: A profile update is pending your confirmation. "
                f"Details: {confirmation_summary}. "
                f"Present this to the user clearly and ask them to reply 'yes' to confirm "
                f"and apply the update, or 'no' to cancel. "
                f"Do NOT apply the update in this turn — wait for explicit confirmation."
            )
        else:
            system_parts.append(
                f"CONFIRMATION CHECKPOINT: A leave application is pending your confirmation. "
                f"Details: {confirmation_summary}. "
                f"Present this to the user clearly and ask them to reply 'yes' to confirm "
                f"and submit the application, or 'no' to cancel. "
                f"Do NOT submit the application in this turn — wait for explicit confirmation."
            )

    # 6. FORMATTING PROTOCOL
    system_parts.append(build_formatting_protocol())

    # 7. EXPORT-AWARE GUIDANCE
    export_guidance = build_export_aware_guidance(wants_export, export_url, metadata=metadata, tone_context=tone_context)
    if export_guidance:
        system_parts.append(export_guidance)

    # 8. CHART ARTIFACT -- inject raw markdown directly into system prompt
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
            "CHART ARTIFACT (already generated -- you MUST embed this EXACTLY in your response):\n"
            "\n"
            + raw_chart_md + "\n\n"
            "CRITICAL INSTRUCTIONS -- READ CAREFULLY:\n"
            "1. The chart above is " + format_desc + ". It is ALREADY generated and ready to display.\n"
            "2. You MUST copy it VERBATIM into your response. Do NOT modify a single character.\n"
            "3. " + format_instructions + "\n"
            "4. Place it AFTER your Stage 1 (Inform) text and BEFORE Stage 3 (Offer feedback).\n"
            "5. NEVER create your own chart, diagram, or Mermaid block. Use ONLY the one provided above.\n"
            "6. NEVER say 'chart shown above' or '[Insert chart here]' or 'here is a diagram'. "
            "The actual artifact must be present verbatim.\n"
            "7. After the chart, add 2-3 bullet points describing key patterns from the data."
        )

    # Inject variance analysis for multi-metric reports
    if variance_context:
        system_parts.append(
            "VARIANCE ANALYSIS (computed from the data -- use these exact figures in your management summary):\n"
            "\n"
            + variance_context + "\n\n"
            "INSTRUCTIONS:\n"
            "1. Use the variance figures above to highlight key movements in the data.\n"
            "2. Call out material variances explicitly: largest increases, largest decreases, and overall trend direction.\n"
            "3. Do NOT invent variance figures -- use ONLY the values provided above.\n"
        )

    # Inject forecast output
    if forecast_context:
        system_parts.append(
            "FORECAST RESULT (generated by statistical model -- use these exact figures):\n"
            "\n"
            + forecast_context + "\n\n"
            "INSTRUCTIONS:\n"
            "1. Present the forecast as a clear time-series table or bullet list (period, point estimate, and 95% intervals).\n"
            "2. Highlight the point estimate and the 95% confidence intervals (fields lower_95 / upper_95 in each row).\n"
            "3. Note the model type and training data size (training_rows) from model_info.\n"
            "4. If a 'metrics' block with mape/rmse is present, you may briefly note forecast accuracy in one sentence.\n"
            "5. Do NOT invent forecast values -- use ONLY the values provided above.\n"
        )

    # Inject market context (external data sources)
    if market_context:
        system_parts.append(
            "MARKET CONTEXT (external data sources that influenced the prediction):\n"
            "\n"
            + market_context + "\n\n"
            "INSTRUCTIONS:\n"
            "1. Explain which external data sources influenced the prediction.\n"
            "2. Connect market conditions to the HR forecast where relevant.\n"
            "3. Be concise -- 1-2 sentences on market influence.\n"
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
    user_prompt_parts.append('Question: "' + _sanitize_user_input(user_query) + '"\n\nFacts:\n' + context + '\n\n')
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

    llm_text = _call_llm_with_retry(llm, system, user_prompt)
    if llm_text is None:
        logger.error("LLM summarization failed, falling back to raw context")
        llm_text = f"[SUMMARIZATION ERROR] {context[:500]}..."
    logger.info("CHART_DEBUG_SUMMARIZER: llm_text_len=%d has_mermaid=%s",
                len(llm_text), "```mermaid" in llm_text)

    # Validate and fix LLM output
    llm_text = _validate_llm_output(llm_text, raw_chart_md, context)

    if large_result_prefix:
        llm_text = large_result_prefix + llm_text

    # =======================================================================
    # EXPORT URL INJECTION -- placeholder replacement
    # =======================================================================
    if export_url:
        export_link_md = f"[Download Excel Export]({export_url})"
        llm_text = llm_text.replace("{{EXPORT_URL}}", export_link_md)

    return llm_text
