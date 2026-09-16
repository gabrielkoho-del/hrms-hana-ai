# agent/output/chart_config.py
#
# Chart configuration loader — single source of truth for column-name patterns,
# preferences, and human-readable label overrides used by chart_generator.py.
#
# Loaded from config/chart_config.yaml at import time.
# NO fail-open to built-in defaults: missing or malformed config raises an error
# so the deployment fixes the config file rather than silently using hardcoded values.
"""Chart configuration loader — single source of truth for column patterns and labels.

Loaded from ``config/chart_config.yaml`` at import time. No fail-open to built-in
defaults: missing or malformed config raises so the deployment fixes the config
file rather than silently using hardcoded values.
"""
import logging
import os
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import yaml

logger = logging.getLogger("hr_agent")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHART_CONFIG_YAML = os.path.join(_BASE_DIR, "..", "config", "chart_config.yaml")


class ChartConfig:
    """Chart configuration loaded from config/chart_config.yaml.

    No fail-open to defaults: if the file is missing or malformed, an error
    is raised so the deployment fixes the config rather than silently falling
    back to hardcoded values.
    """

    def __init__(self, yaml_path: str = _CHART_CONFIG_YAML):
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(
                f"chart_config.yaml not found at {yaml_path}. "
                "Create the file or fix the path."
            )

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"chart_config.yaml must be a YAML mapping, got {type(data).__name__}"
            )

        # ── Privacy enforcement ──────────────────────────────────────────────
        pi = data.get("personal_identifiers")
        if not isinstance(pi, list) or not pi:
            raise ValueError("personal_identifiers must be a non-empty list")
        self.personal_identifiers: FrozenSet[str] = frozenset(
            str(x).lower().strip() for x in pi
        )

        # ── Aggregate value detection ────────────────────────────────────────
        avc = data.get("aggregate_value_cols")
        if not isinstance(avc, list) or not avc:
            raise ValueError("aggregate_value_cols must be a non-empty list")
        self.aggregate_value_cols: FrozenSet[str] = frozenset(
            str(x).lower().strip() for x in avc
        )

        # ── Pre-aggregated category detection ────────────────────────────────
        pacp = data.get("pre_aggregated_category_patterns")
        if not isinstance(pacp, list) or not pacp:
            raise ValueError(
                "pre_aggregated_category_patterns must be a non-empty list"
            )
        self.pre_aggregated_category_patterns: FrozenSet[str] = frozenset(
            str(x).lower().strip() for x in pacp
        )

        # ── Non-numeric column protection ────────────────────────────────────
        nncp = data.get("non_numeric_column_patterns")
        if not isinstance(nncp, list) or not nncp:
            raise ValueError(
                "non_numeric_column_patterns must be a non-empty list"
            )
        self.non_numeric_column_patterns: FrozenSet[str] = frozenset(
            str(x).lower().strip() for x in nncp
        )

        # ── Column preference order ──────────────────────────────────────────
        pcc = data.get("preferred_category_cols")
        if not isinstance(pcc, list):
            raise ValueError("preferred_category_cols must be a list")
        self.preferred_category_cols: List[str] = [
            str(x).lower().strip() for x in pcc
        ]

        pnc = data.get("preferred_numeric_cols")
        if not isinstance(pnc, list):
            raise ValueError("preferred_numeric_cols must be a list")
        self.preferred_numeric_cols: List[str] = [
            str(x).lower().strip() for x in pnc
        ]

        # ── Wide-format metric detection ─────────────────────────────────────
        wfec = data.get("wide_format_exclude_cols")
        if not isinstance(wfec, list):
            raise ValueError("wide_format_exclude_cols must be a list")
        self.wide_format_exclude_cols: FrozenSet[str] = frozenset(
            str(x).lower().strip() for x in wfec
        )

        # ── Multi-series x-axis preference ───────────────────────────────────
        pxc = data.get("preferred_x_cols")
        if not isinstance(pxc, list):
            raise ValueError("preferred_x_cols must be a list")
        self.preferred_x_cols: List[str] = [str(x).lower().strip() for x in pxc]

        # ── Human-readable y-axis label overrides ────────────────────────────
        hryl = data.get("human_readable_y_labels")
        if not isinstance(hryl, dict):
            raise ValueError("human_readable_y_labels must be a mapping")
        self.human_readable_y_labels: Dict[str, str] = {
            str(k).lower().strip(): str(v) for k, v in hryl.items()
        }

        # ── Chart type constants ─────────────────────────────────────────────
        mst = data.get("mermaid_supported_types")
        if not isinstance(mst, list) or not mst:
            raise ValueError("mermaid_supported_types must be a non-empty list")
        self.mermaid_supported_types: Tuple[str, ...] = tuple(
            str(x).lower().strip() for x in mst
        )

        mot = data.get("matplotlib_override_types")
        if not isinstance(mot, list):
            raise ValueError("matplotlib_override_types must be a list")
        self.matplotlib_override_types: Tuple[str, ...] = tuple(
            str(x).lower().strip() for x in mot
        )

        logger.info("CHART_CONFIG: loaded from %s", yaml_path)

    def humanize_y_label(self, y_col: Optional[str]) -> str:
        """Map a raw aggregate column name to a human-readable y-axis label.

        Checks explicit overrides first, then applies standard prefix/suffix
        patterns, then falls back to title-cased column name.
        """
        if not y_col:
            return "Count"
        c = y_col.lower().strip()

        # Explicit config override
        if c in self.human_readable_y_labels:
            return self.human_readable_y_labels[c]

        # Standard aggregate prefixes
        if c == "count" or c.startswith("count_") or c in ("employees", "employee_count"):
            return "Number of Employees"
        if c.startswith("total_"):
            field = c[6:].replace("_", " ").title()
            return f"Total {field}"
        if c.startswith("sum_"):
            field = c[4:].replace("_", " ").title()
            return f"Total {field}"
        if c.startswith("avg_") or c.startswith("average_"):
            field = c[4:].replace("_", " ").title() if c.startswith("avg_") else c[8:].replace("_", " ").title()
            return f"Average {field}"
        if c.startswith("min_"):
            field = c[4:].replace("_", " ").title()
            return f"Minimum {field}"
        if c.startswith("max_"):
            field = c[4:].replace("_", " ").title()
            return f"Maximum {field}"

        # Generic fallback
        return y_col.replace("_", " ").title()


# Singleton — imported by chart_generator.py
chart_config = ChartConfig()
