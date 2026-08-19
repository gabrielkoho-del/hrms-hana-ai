"""
intent_config.py
Single source of truth loader for config/intents.yaml.
Loaded by intent_classifier.py at startup.

Industry standard: cache-once, fail-open with warnings.
"""
import os
import yaml
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("hr_agent")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_INTENTS_CONFIG_PATH = os.path.join(_BASE_DIR, "config", "intents.yaml")

# Module-level cache
_INTENTS_CONFIG: Optional[Dict[str, Any]] = None
_INTENTS_MTIME: float = 0.0


def _load_intents_config() -> Dict[str, Any]:
    """Load intents configuration from YAML file. Returns empty dict on failure."""
    if not os.path.isfile(_INTENTS_CONFIG_PATH):
        logger.warning("Intents config not found at %s, using empty defaults", _INTENTS_CONFIG_PATH)
        return {}
    try:
        with open(_INTENTS_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config if isinstance(config, dict) else {}
    except Exception as e:
        logger.warning("Failed to load intents config from %s: %s", _INTENTS_CONFIG_PATH, e)
        return {}


def get_intents_config(force_reload: bool = False) -> Dict[str, Any]:
    """Return cached intents config, reloading if file changed or forced."""
    global _INTENTS_CONFIG, _INTENTS_MTIME
    if force_reload or _INTENTS_CONFIG is None:
        _INTENTS_CONFIG = _load_intents_config()
        _INTENTS_MTIME = os.path.getmtime(_INTENTS_CONFIG_PATH) if os.path.isfile(_INTENTS_CONFIG_PATH) else 0.0
        return _INTENTS_CONFIG
    # Check if file was modified since last load
    try:
        current_mtime = os.path.getmtime(_INTENTS_CONFIG_PATH)
        if current_mtime != _INTENTS_MTIME:
            logger.info("Intents config changed on disk, reloading")
            _INTENTS_CONFIG = _load_intents_config()
            _INTENTS_MTIME = current_mtime
    except OSError:
        pass
    return _INTENTS_CONFIG


def get_categories() -> Dict[str, Any]:
    """Return category definitions from intents config."""
    config = get_intents_config()
    return config.get("categories", {})


def get_intents() -> Dict[str, Any]:
    """Return intent definitions from intents config."""
    config = get_intents_config()
    return config.get("intents", {})


def get_keywords(group: Optional[str] = None) -> Any:
    """Return keyword lists from intents config.

    If group is provided, return that keyword group (e.g., 'finance', 'aggregate_indicators').
    If group is None, return the full keywords dict.
    """
    config = get_intents_config()
    keywords = config.get("keywords", {})
    if group is not None:
        return keywords.get(group, [])
    return keywords


def get_keyword_groups() -> Dict[str, List[str]]:
    """Return all keyword groups as a dict of group_name -> list of keywords."""
    config = get_intents_config()
    return config.get("keywords", {})


def get_full_info_patterns() -> List[str]:
    """Return full-info regex pattern strings from intents config."""
    config = get_intents_config()
    return config.get("full_info_patterns", [])


def get_intent_category(intent_name: str) -> Optional[str]:
    """Return the category for a given intent name."""
    intents = get_intents()
    intent_def = intents.get(intent_name, {})
    return intent_def.get("category")


def get_category_config(category_name: str) -> Dict[str, Any]:
    """Return config for a given category, with defaults."""
    categories = get_categories()
    return categories.get(category_name, {})


def get_exemplars(intent_name: Optional[str] = None, category_name: Optional[str] = None) -> Dict[str, List[str]]:
    """Return exemplar queries.

    If intent_name is provided, return exemplars for that intent only.
    If category_name is provided, return exemplars for all intents in that category.
    """
    intents = get_intents()
    if intent_name is not None:
        intent_def = intents.get(intent_name, {})
        return {intent_name: intent_def.get("exemplars", [])}
    if category_name is not None:
        result = {}
        for name, defn in intents.items():
            if defn.get("category") == category_name:
                result[name] = defn.get("exemplars", [])
        return result
    # Return all exemplars
    return {name: defn.get("exemplars", []) for name, defn in intents.items()}


def get_all_intent_names() -> List[str]:
    """Return list of all valid intent names."""
    return list(get_intents().keys())


def get_all_category_names() -> List[str]:
    """Return list of all valid category names."""
    return list(get_categories().keys())


def get_finance_keywords() -> Set[str]:
    """Return finance keyword set (single source of truth from intents.yaml)."""
    return set(get_keywords("finance"))


def get_aggregate_keywords() -> Set[str]:
    """Return aggregate indicator keyword set."""
    return set(get_keywords("aggregate_indicators"))


def get_individual_keywords() -> Set[str]:
    """Return individual indicator keyword set."""
    return set(get_keywords("individual_indicators"))


def get_forecast_keywords() -> Set[str]:
    """Return forecast keyword set."""
    return set(get_keywords("forecast"))


def get_export_keywords() -> Set[str]:
    """Return export keyword set."""
    return set(get_keywords("export"))


def get_export_offer_keywords() -> Set[str]:
    """Return export offer keyword set."""
    return set(get_keywords("export_offer"))


def get_multi_option_keywords() -> Set[str]:
    """Return multi-option offer keyword set."""
    return set(get_keywords("multi_option_offer"))


def get_affirmative_keywords() -> Set[str]:
    """Return affirmative keyword set."""
    return set(get_keywords("affirmative"))
