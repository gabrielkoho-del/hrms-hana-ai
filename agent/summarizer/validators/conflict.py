import logging

logger = logging.getLogger("hr_agent")


def detect_conflicts(rag_context: str, sql_facts: str, llm_fn) -> str:
    if not rag_context or not sql_facts:
        return "NO_CONFLICTS"

    rag_lower = rag_context.lower()
    if any(phrase in rag_lower for phrase in ["no relevant", "no matching", "not found", "no documents"]):
        return "NO_CONFLICTS"

    system = """You are a policy-data validator. Compare HR policy text against employee SQL records.
Detect ONLY real contradictions, not missing data or unrelated topics.

Examples of conflicts:
- Policy says "eligible for 90 days maternity leave" but SQL shows leave_balance = 0
- Policy requires "doctor's note for MC > 2 days" but SQL shows no_medical_cert = true
- Policy says "probation period is 3 months" but SQL shows hire_date = 2 months ago
- Policy says "all employees get 14 days annual leave" but SQL shows annual_leave = 0

If no conflicts: respond with exactly NO_CONFLICTS
If conflicts found: list each as a bullet point:
- CONFLICT: [policy claim] vs [data fact]

Be conservative. Only flag genuine contradictions."""

    prompt = "Policy:\n" + rag_context[:2000] + "\n\nEmployee Data:\n" + sql_facts[:2000]

    try:
        choice = llm_fn(system, prompt, temperature=0.1, max_tokens=400)
        if choice and choice.get("message", {}).get("content"):
            result = choice["message"]["content"].strip()
            if "NO_CONFLICTS" in result.upper():
                return "NO_CONFLICTS"
            return result
    except Exception as e:
        logger.warning("Conflict detection failed: %s", e)

    return "NO_CONFLICTS"
