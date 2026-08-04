"""
Lightweight intent guards for the agentic executor.

L1: Greeting / smalltalk / punctuation-only input.
L2: Vague data-availability discovery questions ("what data do you have?").

These guards operate before planning/intent-classification to avoid
unnecessary LLM calls for trivial or informational inputs.
"""
import re
from typing import Dict, Tuple


# L1: Greeting / smalltalk / punctuation-only patterns.
_GREETING_PATTERNS = [
    re.compile(r'^\s*(hi|hello|hey|greetings|howdy|hola|bonjour|sup|yo)\b(\s+there)?[\s,!?.…]*$', re.I),
    re.compile(r'^\s*good\s+(morning|afternoon|evening|day)\b(\s+(everyone|all|there|team|folks))?[\s,!?.…]*$', re.I),
    re.compile(r'^\s*(thanks|thank\s+you|thx|ty)\b(\s+(very\s+much|a\s+lot|so\s+much))?[\s,!?.…]*$', re.I),
    re.compile(r'^\s*(bye|goodbye|see\s+ya|later)\b[\s,!?.…]*$', re.I),
    re.compile(r"^\s*what\'s\s+up\b[\s,!?.…]*$", re.I),
    re.compile(r"^\s*what\s+is\s+up\b[\s,!?.…]*$", re.I),
    re.compile(r'^\s*[!?.…]{1,3}\s*$'),
]

_SMALLTALK_PATTERNS = [
    re.compile(r'^\s*(who\s+are\s+you|what\s+can\s+you\s+do|'
               r'what\s+is\s+your\s+name|tell\s+me\s+about\s+yourself)\b[\s,!?.…]*$', re.I),
    re.compile(r'^\s*(are\s+you\s+(an?\s+)?(ai|bot|assistant|human))\b[\s,!?.…]*$', re.I),
]


def is_greeting_or_smalltalk(query: str) -> Tuple[bool, str]:
    """Return (True, label) if the query is a greeting, smalltalk, or punctuation-noise.

    Labels:
      - "greeting"  : hello / thanks / bye / punctuation-only
      - "smalltalk" : who are you / what can you do / are you an AI?
      - "task_or_followup" : anything requiring actual work
    """
    q = query.strip()
    if len(q) <= 3:
        return True, "greeting"
    for pattern in _GREETING_PATTERNS:
        if pattern.match(q):
            return True, "greeting"
    for pattern in _SMALLTALK_PATTERNS:
        if pattern.match(q):
            return True, "smalltalk"
    return False, "task_or_followup"


def greeting_response(label: str) -> str:
    if label == "greeting":
        return "Hello! How can I help you with HR-related questions today?"
    return (
        "I'm your HR AI Agent. I can help you look up employee information, "
        "leave balances, org hierarchy, and HR policies. What would you like to know?"
    )


# L2: Vague data-availability discovery patterns.
_VAGUE_DATA_PATTERNS = [
    re.compile(r"\b(any\s+(financial|hr|employee|leave|salary)\s+(data|records?|files?))\b", re.I),
    re.compile(r"\b(do\s+i\s+have\s+(any|some)\s+(\w+\s+)?(data|info(rmation)?|records?|files?))\b", re.I),
    re.compile(r"\b(do\s+you\s+have\s+(any|some)\s+(\w+\s+)?(data|info(rmation)?|records?|files?))\b", re.I),
    re.compile(r"\b(what\s+data|what\s+info(rmation)?|what\s+records?|what\s+files?|"
               r"what\s+do\s+you\s+have|what\s+is\s+available|available\s+data|"
               r"available\s+info(rmation)?|available\s+records?)\b", re.I),
    re.compile(r"\b(show\s+me\s+what\s+(data|info(rmation)?|records?|systems?|sources?))\b", re.I),
    re.compile(r"\b(tell\s+me\s+what\s+(data|info(rmation)?|records?))\b", re.I),
    re.compile(r"\b(is\s+there\s+any\s+(data|info(rmation)?|records?|files?))\b", re.I),
    re.compile(r"\b(what\s+can\s+you\s+(access|show|pull|fetch|provide))\b", re.I),
]


def is_vague_data_question(query: str) -> bool:
    """Return True if the query is a vague 'what data is available?' style question."""
    q = query.strip()
    if len(q) <= 5:
        return False
    for pattern in _VAGUE_DATA_PATTERNS:
        if pattern.search(q):
            return True
    return False


def data_summary_response(cached_schema: Dict) -> str:
    """Build a human-readable summary of available data sources from the cached schema."""
    lines = ["I can access the following data sources:"]

    if cached_schema and isinstance(cached_schema, dict):
        entities = list(cached_schema.keys())
        if entities:
            lines.append("")
            lines.append("SQL Server / DAB entities:")
            preview = ", ".join(entities[:10]) + (" ..." if len(entities) > 10 else "")
            lines.append(f"  {preview}")

    lines.append("")
    lines.append("Which data source or specific table would you like to explore?")
    return "\n".join(lines)
