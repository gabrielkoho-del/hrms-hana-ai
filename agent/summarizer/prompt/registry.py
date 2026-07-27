import yaml
from pathlib import Path
from typing import Dict, List

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "suggestions.yaml"

PERSONAL_DATA_CATEGORIES = {"personal_data", "emergency", "grievance", "action_request"}
ACTION_CATEGORIES = {"action_request", "emergency", "grievance"}


def _load_assist_registry() -> Dict[str, Dict[str, List[str]]]:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {
        "policy_info": {
            "has_data": [
                "Would you like the policy in a specific format?",
                "Need help interpreting any specific section?",
                "Shall I check related policies (e.g., benefits, leave, conduct)?",
            ],
            "empty_result": [
                "Check the employee handbook or intranet directly",
                "Contact HR policy team for the latest version",
                "Search the company knowledge base",
            ],
        }
    }


ASSIST_REGISTRY = _load_assist_registry()


def get_assist_suggestions(category: str, has_data: bool, has_empty: bool,
                            action_context: str = "") -> List[str]:
    """
    Get assist suggestions from registry.
    Falls back to generic suggestions if category not found.
    """
    state = "has_data" if has_data else "empty_result"
    registry = ASSIST_REGISTRY.get(category, ASSIST_REGISTRY.get("policy_info", {}))
    suggestions = registry.get(state, [])

    if action_context and has_data:
        suggestions = [f"Ready to proceed with: {action_context}"] + suggestions

    return suggestions[:3]
