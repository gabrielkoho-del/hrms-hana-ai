# agent/chart_generator.py
"""Chart generation module — supports matplotlib PNG (local/online) and Mermaid (LibreChat native).

Environment-driven output modes:
  CHART_MODE=matplotlib  → PNG files served via FastAPI /charts endpoint
  CHART_MODE=mermaid     → Mermaid syntax rendered natively by LibreChat v0.8.2+
  CHART_MODE=auto        → Mermaid for simple charts, matplotlib for complex ones
  CHART_MODE=base64      → Inline base64 PNG (air-gapped/offline)

Adheres to HR AI Agent Chart Guidelines:
  • Privacy First: blocks charts with individual employee identifiers
  • Context-Aware: role-based chart access (admin/manager/employee)
  • Dual Format: raw data table always accompanies chart
  • Accessibility: color-blind friendly palette + alt-text

PATCH 2026-06-19: Fixed aggregated data charting + robust DAB response format handling.
PATCH 2026-06-22: Fixed ylabel for pre-aggregated hist data; improved pre-aggregation detection.
PATCH 2026-06-23: Fixed Mermaid quoting (x-axis, title, y-axis), pie slice limits, case-sensitivity in column picking, line chart auto-mode, and fallback table preservation.
PATCH 2026-06-26: Added y_label parameter to generate_chart() and sub-generators for executor post-binning sync compatibility.
"""
import json
import os
import time
import base64
import logging
from typing import List, Dict, Optional, Literal, Tuple

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent.dab.dab_response import extract_items

logger = logging.getLogger("hr_agent")

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════
CHART_MODE = os.getenv("CHART_MODE", "auto").lower()
CHART_OUTPUT_DIR = os.getenv("CHART_OUTPUT_DIR", "/mnt/agents/output")
CHART_BASE_URL = os.getenv("CHART_BASE_URL", os.getenv("AGENT_BASE_URL", "http://localhost:8000"))
CHART_MAX_CATEGORIES = int(os.getenv("CHART_MAX_CATEGORIES", "20"))
CHART_DPI = int(os.getenv("CHART_DPI", "150"))
CHART_PIE_MAX_SLICES = int(os.getenv("CHART_PIE_MAX_SLICES", "8"))

os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)

COLORBLIND_PALETTE = [
    "#648FFF", "#FE6100", "#DC267F", "#785EF0", "#FFB000", "#3DDC97",
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",
    "#CC79A7", "#999999",
]

PERSONAL_IDENTIFIERS = {"emp_id", "employee_id", "email", "name", "first_name", "last_name", "national_id"}
AGGREGATE_VALUE_COLS = {"count", "total", "sum", "avg", "average", "min", "max", "value", "amount", "frequency", "employees", "employee_count"}

# Pre-aggregated categorical patterns: if x_col matches these, data is pre-binned
PRE_AGGREGATED_CATEGORY_PATTERNS = {
    "age_group", "salary_range", "salary_band", "tenure_group", "tenure_band",
    "experience_range", "grade", "level", "job_level", "performance_rating",
    "department", "job_title", "status", "gender", "location", "leave_type",
    "hire_year", "hire_month", "hire_quarter", "year", "month", "quarter"
}


# ═════════════════════════════════════════════════════════════════════════════
# PRIVACY ENFORCEMENT
# ═════════════════════════════════════════════════════════════════════════════

def check_privacy_safe(data: List[Dict], auth_role: Optional[str] = None) -> Tuple[bool, str]:
    if not data or not isinstance(data, list):
        return False, "No data to visualize."
    if data and isinstance(data[0], dict):
        keys = {k.lower() for k in data[0].keys()}
        personal_keys = keys & PERSONAL_IDENTIFIERS
        if personal_keys:
            return False, (
                f"Privacy block: Data contains individual identifiers ({', '.join(personal_keys)}). "
                "Charts must show aggregated data only."
            )
    return True, ""


# ═════════════════════════════════════════════════════════════════════════════
# DATA EXTRACTION — ROBUST: handles all DAB response formats
# ═════════════════════════════════════════════════════════════════════════════

def extract_chartable_data(tool_results: Dict) -> List[Dict]:
    """
    Extract a list of dicts from tool results that can be charted.
    ROBUST: handles all DAB response wrapper formats.
    """
    for tool_name, output in tool_results.items():
        if tool_name.startswith("__"):
            continue

        # Handle {"result": raw_data} wrapper from executor
        raw = output.get("result") if isinstance(output, dict) else output

        logger.info("CHART_DEBUG_EXTRACT: tool=%s raw_type=%s", tool_name, type(raw).__name__)

        # Try deep extraction
        extracted = extract_items(raw)
        if extracted:
            # PATCH: Skip DAB error wrappers (type+text columns only)
            if len(extracted) == 1 and set(extracted[0].keys()) == {"type", "text"}:
                logger.warning("CHART_DEBUG_EXTRACT: skipping error wrapper: %s", 
                              extracted[0].get("text", "")[:200])
                continue
            if extracted and all(set(row.keys()) == {"type", "text"} for row in extracted):
                logger.warning("CHART_DEBUG_EXTRACT: skipping text-only result (likely error)")
                continue
            logger.info("CHART_DEBUG_EXTRACT: found %d rows, cols=%s", 
                       len(extracted), list(extracted[0].keys()) if extracted else [])
            return extracted

        # Legacy fallback paths
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

        if not isinstance(raw, dict):
            continue

        # DAB read_records returns {"entity": "...", "result": [...]}
        if "result" in raw and isinstance(raw["result"], list):
            items = raw["result"]
            if items and isinstance(items[0], dict):
                logger.info("CHART_DEBUG_EXTRACT: legacy path 'result' -> %d rows", len(items))
                return items

        if "employees" in raw and isinstance(raw["employees"], list):
            return raw["employees"]

        if "columns" in raw and "rows" in raw:
            cols = raw["columns"]
            rows = raw["rows"]
            if cols and rows:
                return [dict(zip(cols, row)) for row in rows]

        for key, val in raw.items():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                logger.info("CHART_DEBUG_EXTRACT: fallback path key='%s' -> %d rows", key, len(val))
                return val

    logger.warning("CHART_DEBUG_EXTRACT: no chartable data found in any tool result")
    return []


def _is_aggregate_value_col(col_name: str) -> bool:
    c = col_name.lower().strip()
    if c in AGGREGATE_VALUE_COLS:
        return True
    # Defensive prefix/suffix only: "employee_count", "total_salary", "avg_age"
    # Avoid substring false positives: "accounts" (contains "count"), "subtotal" (contains "total")
    return any(
        c.startswith(kw + "_") or c.endswith("_" + kw)
        for kw in AGGREGATE_VALUE_COLS
    )


def _is_pre_aggregated_category_col(col_name: str) -> bool:
    """Check if a column name indicates pre-binned/pre-aggregated categorical data."""
    c = col_name.lower().strip()
    if c in PRE_AGGREGATED_CATEGORY_PATTERNS:
        return True
    # Defensive suffix check: "age_group", "salary_range" — avoid substring false positives
    return any(
        c.endswith(suffix) for suffix in ["_range", "_band", "_group", "_bucket", "_bin"]
    )


def _pick_columns(data: List[Dict], chart_type: str) -> tuple[Optional[str], Optional[str], bool, Optional[str]]:
    """
    Auto-detect x (categorical), y (numeric), and optional series columns.
    Handles pre-aggregated data: when one column is count/total/sum/avg,
    the OTHER column is treated as categorical even if numeric.
    If multiple categorical columns are present, creates a composite label.
    Returns (x_col, y_col, is_pre_aggregated, series_col).
    """
    if not data or not isinstance(data[0], dict):
        logger.warning("CHART_DEBUG_PICK: no data or not dict list")
        return None, None, False, None

    df = pd.DataFrame(data)
    all_cols = list(df.columns)
    logger.info("CHART_DEBUG_PICK: columns=%s", all_cols)

    # Check for pre-aggregated data pattern
    agg_cols = [c for c in all_cols if _is_aggregate_value_col(c)]
    non_agg_cols = [c for c in all_cols if not _is_aggregate_value_col(c)]

    # Check for pre-binned categorical columns (age_group, salary_range, etc.)
    pre_agg_cat_cols = [c for c in non_agg_cols if _is_pre_aggregated_category_col(c)]

    if len(agg_cols) >= 1 and len(non_agg_cols) >= 1:
        # Pre-aggregated: prioritize pre-binned category columns
        # If multiple categorical columns, composite them for X-axis
        if len(non_agg_cols) >= 2:
            x_col = non_agg_cols[0]  # Use first as primary for composite
            series_col = non_agg_cols[1] if len(non_agg_cols) > 1 else None
            logger.info("CHART_DEBUG_PICK: pre-aggregated multi-series x=%s series=%s y=%s",
                        x_col, series_col, agg_cols[0])
            return x_col, agg_cols[0], True, series_col
        elif pre_agg_cat_cols:
            x_col = pre_agg_cat_cols[-1]  # Most granular pre-binned
        else:
            x_col = non_agg_cols[-1]  # Most granular
        y_col = agg_cols[0]
        series_col = None
        logger.info("CHART_DEBUG_PICK: pre-aggregated x=%s y=%s", x_col, y_col)
        return x_col, y_col, True, series_col

    # Heuristic for 2-column data where one name suggests aggregation
    if len(all_cols) == 2:
        c0, c1 = all_cols[0], all_cols[1]
        if any(kw in c0.lower() for kw in ("count", "total", "sum", "avg", "average")):
            return c1, c0, True, None
        if any(kw in c1.lower() for kw in ("count", "total", "sum", "avg", "average")):
            return c0, c1, True, None

    # Standard classification
    numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in all_cols if c not in numeric_cols]

    preferred_cat = ["department", "job_title", "status", "gender", "location",
                     "hire_year", "hire_month", "hire_quarter", "leave_type", "job_level",
                     "age_group", "salary_range", "salary_band", "tenure_group",
                     "age", "year", "month", "quarter"]
    x_col = next((cc for cc in categorical_cols if cc.lower() in preferred_cat), None)
    if not x_col and categorical_cols:
        x_col = categorical_cols[0]

    preferred_num = ["count", "total", "sum", "avg", "salary", "age", "leave_balance", "turnover_rate"]
    y_col = next((nc for nc in numeric_cols if nc.lower() in preferred_num), None)
    if not y_col and numeric_cols:
        y_col = numeric_cols[0]

    logger.info("CHART_DEBUG_PICK: standard x=%s y=%s", x_col, y_col)
    return x_col, y_col, False, None


# ═════════════════════════════════════════════════════════════════════════════
# DUAL FORMAT: Markdown table companion
# ═════════════════════════════════════════════════════════════════════════════

def generate_data_table(data: List[Dict], max_rows: int = 20) -> str:
    if not data or not isinstance(data[0], dict):
        return ""
    df = pd.DataFrame(data).head(max_rows)
    if df.empty:
        return ""
    lines = ["**Raw Data:**"]
    header = " | ".join(df.columns)
    lines.append(f"| {header} |")
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for _, row in df.iterrows():
        vals = " | ".join(str(v) if v is not None else "" for v in row.values)
        lines.append(f"| {vals} |")
    if len(data) > max_rows:
        lines.append(f"\n*... and {len(data) - max_rows} more rows. Export to Excel for full data.*")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# ALT-TEXT GENERATION (Accessibility)
# ═════════════════════════════════════════════════════════════════════════════

def _generate_alt_text(data: List[Dict], chart_type: str, x_col: Optional[str], y_col: Optional[str], title: Optional[str]) -> str:
    if not data:
        return "Chart: no data available."
    n = len(data)
    t = title or "Chart"
    # PATCH 2026-06-24: humanize snake_case column names for accessibility
    x = (x_col or "category").replace("_", " ").strip()
    y = (y_col or "value").replace("_", " ").strip()
    if chart_type == "pie":
        return f"{t}: Pie chart showing proportional breakdown across {n} {x} categories."
    elif chart_type in ("bar", "barh"):
        return f"{t}: Bar chart comparing {n} {x} categories by {y}."
    elif chart_type == "line":
        return f"{t}: Line chart showing {y} trend across {n} {x} data points."
    elif chart_type == "hist":
        return f"{t}: Histogram showing distribution of {x} across {n} records."
    else:
        return f"{t}: Chart displaying {n} data points."


# ═════════════════════════════════════════════════════════════════════════════
# MERMAID GENERATION (LibreChat native, no file serving needed)
# ═════════════════════════════════════════════════════════════════════════════

def _escape_mermaid(text: str) -> str:
    """Escape characters that break Mermaid xychart-beta syntax.

    Mermaid xychart-beta does NOT support backslash-escaped quotes.
    We replace internal double quotes with single quotes and strip
    surrounding quotes to prevent double-wrapping.
    Do NOT replace [ ] ( ) , — they are valid inside quoted strings.
    """
    text = text.strip()
    # Defensive: strip surrounding quotes to prevent double-wrapping
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        text = text[1:-1]
    return (
        text.replace('"', "'")
           .replace("\\", "/")
           .replace("\n", " ")
    )
def _humanize_y_label(y_col: Optional[str]) -> str:
    """Map raw aggregate column names to human-readable y-axis labels."""
    if not y_col:
        return "Count"
    c = y_col.lower().strip()
    # Count / total of employees → Number of Employees
    if c == "count" or c.startswith("count_") or c == "total" or c == "employees" or c == "employee_count":
        return "Number of Employees"
    # Total → Total <Field> (e.g., total_salary → Total Salary)
    if c.startswith("total_"):
        field = c[6:].replace("_", " ").title()
        return f"Total {field}"
    # Sum → Total <Field>
    if c.startswith("sum_"):
        field = c[4:].replace("_", " ").title()
        return f"Total {field}"
    # Average → Average <Field>
    if c.startswith("avg_") or c.startswith("average_"):
        field = c[4:].replace("_", " ").title() if c.startswith("avg_") else c[8:].replace("_", " ").title()
        return f"Average {field}"
    # Min / Max
    if c.startswith("min_"):
        field = c[4:].replace("_", " ").title()
        return f"Minimum {field}"
    if c.startswith("max_"):
        field = c[4:].replace("_", " ").title()
        return f"Maximum {field}"
    # Generic fallback: title-case
    return y_col.replace("_", " ").title()


def generate_mermaid_chart(
    data: List[Dict],
    chart_type: str,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    include_table: bool = True
) -> str:
    """Generate Mermaid xychart-beta or pie syntax."""
    if not data or not isinstance(data[0], dict):
        logger.warning("CHART_DEBUG_MERMAID: no data")
        return ""

    df = pd.DataFrame(data)
    picked_x, picked_y, _, series_col = _pick_columns(data, chart_type)
    # If explicit x_column/y_column doesn't exist in binned data, fall back to auto-detected
    x_col = x_column if x_column and x_column in df.columns else picked_x
    y_col = y_column if y_column and y_column in df.columns else picked_y

    if not x_col or x_col not in df.columns:
        logger.warning("CHART_DEBUG_MERMAID: x_column '%s' not found in %s", x_column, list(df.columns))
        return ""

    chart_title = title or f"{x_col.replace('_', ' ').title()} Distribution"
    # PATCH 2026-06-23: escape title once for reuse
    safe_title = _escape_mermaid(chart_title)
    alt_text = _generate_alt_text(data, chart_type, x_col, y_col, chart_title)

    lines = ["```mermaid"]

    if chart_type == "pie":
        if series_col:
            # Multi-series pie: aggregate to top-level categories
            df = df.groupby(x_col)[y_col].sum().reset_index()
        df = df.head(CHART_PIE_MAX_SLICES)
        if len(df) > CHART_PIE_MAX_SLICES:
            df = df.nlargest(CHART_PIE_MAX_SLICES - 1, y_col)
            other_val = df.iloc[CHART_PIE_MAX_SLICES - 1:][y_col].sum() if len(df) > CHART_PIE_MAX_SLICES else 0
            if other_val > 0:
                other_row = pd.DataFrame([{x_col: "Other", y_col: other_val}])
                df = pd.concat([df, other_row], ignore_index=True)
        lines.append("pie showData")
        lines.append(f'    title "{safe_title}"')
        for _, row in df.iterrows():
            label = _escape_mermaid(str(row.get(x_col, "Unknown")))
            val = row.get(y_col, 0) if y_col and y_col in df.columns else 0
            if val <= 0:
                continue
            lines.append(f'    "{label}" : {val}')
        if len(lines) <= 3:
            logger.warning("CHART_DEBUG_MERMAID: pie chart has no valid slices, falling back to bar")
            return generate_mermaid_chart(data, "bar", x_column, y_column, title, y_label=y_label, include_table=include_table)

    elif chart_type in ("bar", "barh", "line"):
        lines.append("xychart-beta")
        lines.append(f'    title "{safe_title}"')
        # Handle multi-series with composite labels
        if series_col:
            df = _composite_x_labels(df, x_col, series_col)
            x_axis_col = "__x_label__"
        else:
            x_axis_col = x_col
        categories = [f'"{_escape_mermaid(str(v))}"' for v in df[x_axis_col].tolist()]
        lines.append(f'    x-axis [{", ".join(categories)}]')
        if y_col and y_col in df.columns:
            values = [str(v) for v in df[y_col].tolist()]
            if pd.api.types.is_numeric_dtype(df[y_col]):
                y_max = df[y_col].max()
                max_val = max(1, int(y_max)) if pd.notna(y_max) else 1
            else:
                max_val = max(1, len(df))
            y_label_text = _escape_mermaid(y_label) if y_label else _escape_mermaid(_humanize_y_label(y_col))
            lines.append(f'    y-axis "{y_label_text}" 0 --> {max_val}')
            if chart_type == "line":
                lines.append(f'        line [{", ".join(values)}]')
            else:
                lines.append(f'        bar [{", ".join(values)}]')
        else:
            counts = ["1"] * len(df)
            y_label_text = _escape_mermaid(y_label) if y_label else "Count"
            lines.append(f'    y-axis "{y_label_text}" 0 --> 1')
            lines.append(f'        bar [{", ".join(counts)}]')

    else:
        logger.warning("Mermaid does not support chart_type='%s', switching to matplotlib", chart_type)
        return generate_mermaid_chart(data, "bar", x_column, y_column, title, y_label=y_label, include_table=include_table)

    logger.info("CHART_DEBUG_MERMAID: generated syntax:\n%s", "\n".join(lines))
    lines.append("```")
    lines.append(f"<!-- Alt-text: {alt_text} -->")
    if include_table:
        table_md = generate_data_table(data)
        if table_md:
            lines.append("")
            lines.append(table_md)
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB GENERATION (PNG — requires file serving)
# ═════════════════════════════════════════════════════════════════════════════

def _apply_colorblind_palette(ax, chart_type: str, num_colors: int = 1):
    colors = COLORBLIND_PALETTE[:max(num_colors, 1)]
    if chart_type == "pie":
        return colors
    return colors


def _composite_x_labels(df: pd.DataFrame, x_col: str, series_col: Optional[str]) -> pd.DataFrame:
    """
    Create composite X-axis labels when both x_col and series_col are present.
    E.g., department + gender -> "Sales - Female", "Finance - Male"
    Returns a copy of df with a new '__x_label__' column.
    """
    if not series_col or series_col not in df.columns:
        return df
    result = df.copy()
    result["__x_label__"] = result[x_col].astype(str) + " - " + result[series_col].astype(str)
    return result


def generate_matplotlib_chart(
    data: List[Dict],
    chart_type: str,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    return_mode: Literal["url", "base64", "path"] = "url",
    include_table: bool = True
) -> str:
    """
    Generate matplotlib chart and return URL, base64 data URI, or file path.
    Handles pre-aggregated data correctly via df.plot(x=cat, y=val).
    Handles multi-series (group by) data by creating composite X-axis labels.
    """
    if not data or not isinstance(data, list):
        logger.warning("CHART_DEBUG_MATPLOTLIB: no data")
        return ""

    df = pd.DataFrame(data)
    picked_x, picked_y, is_pre_aggregated, series_col = _pick_columns(data, chart_type)

    # If explicit x_column/y_column doesn't exist in binned data, fall back to auto-detected
    x_col = x_column if x_column and x_column in df.columns else picked_x
    y_col = y_column if y_column and y_column in df.columns else picked_y

    # If y was explicitly provided and is aggregate, trust the pre-aggregated flag
    if y_column and y_col and _is_aggregate_value_col(y_col):
        is_pre_aggregated = True

    if not x_col or x_col not in df.columns:
        logger.warning("CHART_DEBUG_MATPLOTLIB: x_col '%s' not found in %s", x_col, list(df.columns))
        return ""

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        chart_title = title or f"{x_col.replace('_', ' ').title()} Distribution"
        alt_text = _generate_alt_text(data, chart_type, x_col, y_col, chart_title)

        logger.info("CHART_DEBUG_MATPLOTLIB: x=%s y=%s is_agg=%s type=%s rows=%d series=%s",
                    x_col, y_col, is_pre_aggregated, chart_type, len(df), series_col)

        if chart_type == "bar":
            if series_col:
                # Multi-series grouped bar charts
                df_labeled = _composite_x_labels(df, x_col, series_col)
                df_labeled = df_labeled.sort_values(by=y_col, ascending=False).head(CHART_MAX_CATEGORIES)
                colors = _apply_colorblind_palette(ax, "bar", len(df_labeled))
                df_labeled.plot(x="__x_label__", y=y_col, kind="bar", ax=ax, color=colors[0], legend=False)
                ax.set_xlabel("Category")
                ax.set_ylabel(y_label if y_label else _humanize_y_label(y_col))
            elif is_pre_aggregated:
                plot_df = df.sort_values(by=y_col, ascending=False).head(CHART_MAX_CATEGORIES)
                colors = _apply_colorblind_palette(ax, "bar", len(plot_df))
                plot_df.plot(x=x_col, y=y_col, kind="bar", ax=ax, color=colors[0], legend=False)
                ax.set_xlabel(x_col.replace("_", " ").title())
                ax.set_ylabel(y_label if y_label else _humanize_y_label(y_col))
            else:
                value_counts = df[x_col].value_counts().sort_values(ascending=False)
                value_counts = value_counts.head(CHART_MAX_CATEGORIES)
                colors = _apply_colorblind_palette(ax, "bar", len(value_counts))
                value_counts.plot(kind="bar", ax=ax, color=colors[0])
                ax.set_xlabel(x_col.replace("_", " ").title())
                ax.set_ylabel(y_label if y_label else "Count")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        elif chart_type == "barh":
            if series_col:
                # Multi-series grouped bar charts (horizontal)
                df_labeled = _composite_x_labels(df, x_col, series_col)
                df_labeled = df_labeled.sort_values(by=y_col, ascending=True).head(CHART_MAX_CATEGORIES)
                colors = _apply_colorblind_palette(ax, "barh", len(df_labeled))
                df_labeled.plot(x="__x_label__", y=y_col, kind="barh", ax=ax, color=colors[0], legend=False)
                ax.set_xlabel("Category")
                ax.set_ylabel(y_col.replace("_", " ").title() if y_col else "Count")
            elif is_pre_aggregated:
                plot_df = df.sort_values(by=y_col, ascending=True).head(CHART_MAX_CATEGORIES)
                colors = _apply_colorblind_palette(ax, "barh", len(plot_df))
                plot_df.plot(x=x_col, y=y_col, kind="barh", ax=ax, color=colors[0], legend=False)
                ax.set_xlabel(y_label if y_label else _humanize_y_label(y_col))
                ax.set_ylabel(x_col.replace("_", " ").title())
            else:
                value_counts = df[x_col].value_counts().sort_values(ascending=True)
                value_counts = value_counts.head(CHART_MAX_CATEGORIES)
                colors = _apply_colorblind_palette(ax, "barh", len(value_counts))
                value_counts.plot(kind="barh", ax=ax, color=colors[0])
                ax.set_xlabel(y_label if y_label else "Count")
                ax.set_ylabel(x_col.replace("_", " ").title())

        elif chart_type == "pie":
            if series_col:
                # Multi-series pie: aggregate to top-level categories
                plot_df = df.groupby(x_col)[y_col].sum().reset_index()
                plot_df = plot_df.sort_values(by=y_col, ascending=False).head(CHART_PIE_MAX_SLICES)
            elif is_pre_aggregated:
                plot_df = df.head(CHART_PIE_MAX_SLICES)
                if len(df) > CHART_PIE_MAX_SLICES:
                    top = df.nlargest(CHART_PIE_MAX_SLICES - 1, y_col)
                    other_val = df.iloc[CHART_PIE_MAX_SLICES - 1:][y_col].sum()
                    other_row = pd.DataFrame([{x_col: "Other", y_col: other_val}])
                    plot_df = pd.concat([top, other_row], ignore_index=True)
                colors = _apply_colorblind_palette(ax, "pie", len(plot_df))
                plot_df.set_index(x_col)[y_col].plot(kind="pie", ax=ax, autopct="%1.1f%%", startangle=90, colors=colors)
            else:
                value_counts = df[x_col].value_counts()
                if len(value_counts) > CHART_PIE_MAX_SLICES:
                    top = value_counts.nlargest(CHART_PIE_MAX_SLICES - 1)
                    other = value_counts.iloc[CHART_PIE_MAX_SLICES - 1:].sum()
                    if other > 0:
                        top["Other"] = other
                    value_counts = top
                colors = _apply_colorblind_palette(ax, "pie", len(value_counts))
                value_counts.plot(kind="pie", ax=ax, autopct="%1.1f%%", startangle=90, colors=colors)
            ax.set_ylabel("")

        elif chart_type == "hist":
            if series_col:
                # Multi-series hist: composite labels
                df_labeled = _composite_x_labels(df, x_col, series_col)
                df_labeled = df_labeled.sort_values(by=y_col).head(CHART_MAX_CATEGORIES)
                colors = _apply_colorblind_palette(ax, "bar", len(df_labeled))
                df_labeled.plot(x="__x_label__", y=y_col, kind="bar", ax=ax, color=colors[0], legend=False)
                ax.set_xlabel("Category")
                ax.set_ylabel(y_label if y_label else _humanize_y_label(y_col))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
            elif is_pre_aggregated:
                logger.info("CHART_DEBUG: hist with pre-aggregated data — rendering as bar chart with proper labels")
                plot_df = df.sort_values(by=x_col).head(CHART_MAX_CATEGORIES)
                colors = _apply_colorblind_palette(ax, "bar", len(plot_df))
                plot_df.plot(x=x_col, y=y_col, kind="bar", ax=ax, color=colors[0], legend=False)
                ax.set_xlabel(x_col.replace("_", " ").title())
                ax.set_ylabel(y_label if y_label else _humanize_y_label(y_col))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
            else:
                # Raw numeric distribution — genuine histogram
                # For HR user-facing charts, ylabel should be "Count" not "Frequency"
                # "Frequency" is technically correct but semantically confusing for non-technical users
                if y_col and y_col in df.columns:
                    df[y_col].plot(kind="hist", ax=ax, bins=20, color=COLORBLIND_PALETTE[0], edgecolor="black")
                    ax.set_xlabel(y_col.replace("_", " ").title())
                else:
                    df[x_col].plot(kind="hist", ax=ax, bins=20, color=COLORBLIND_PALETTE[0], edgecolor="black")
                    ax.set_xlabel(x_col.replace("_", " ").title())
                # Use explicit y_label if provided, otherwise default to Count
                display_label = y_label if y_label else "Count"
                ax.set_ylabel(display_label)

        elif chart_type == "line":
            if series_col:
                # Multi-series line: aggregate to top-level categories
                df = _composite_x_labels(df, x_col, series_col)
                plot_df = df.sort_values(by=x_col)
                plot_df.plot(x="__x_label__", y=y_col, kind="line", ax=ax, marker="o", color=COLORBLIND_PALETTE[0], legend=False)
                ax.set_ylabel(y_label if y_label else _humanize_y_label(y_col))
            elif is_pre_aggregated:
                plot_df = df.sort_values(by=x_col)
                plot_df.plot(x=x_col, y=y_col, kind="line", ax=ax, marker="o", color=COLORBLIND_PALETTE[0], legend=False)
                ax.set_ylabel(y_label if y_label else _humanize_y_label(y_col))
            else:
                if y_col and y_col in df.columns:
                    plot_df = df.sort_values(by=x_col)
                    plot_df.plot(x=x_col, y=y_col, kind="line", ax=ax, marker="o", color=COLORBLIND_PALETTE[0], legend=False)
                    ax.set_ylabel(y_label if y_label else _humanize_y_label(y_col))
                else:
                    counts = df.groupby(x_col).size().reset_index(name="count")
                    counts = counts.sort_values(by=x_col)
                    counts.plot(x=x_col, y="count", kind="line", ax=ax, marker="o", color=COLORBLIND_PALETTE[0], legend=False)
                    ax.set_ylabel(y_label if y_label else "Count")
            ax.set_xlabel(x_col.replace("_", " ").title())

        elif chart_type == "box":
            if y_col and y_col in df.columns:
                # Determine x-axis grouping
                if series_col:
                    # Create composite labels for each combination of x_col and series_col
                    df = _composite_x_labels(df, x_col, series_col)
                    x_col_for_box = "__x_label__"
                elif x_col and x_col in df.columns and x_col != y_col:
                    x_col_for_box = x_col
                else:
                    x_col_for_box = None

                if x_col_for_box and x_col_for_box in df.columns:
                    # Grouped box plot, limit categories for readability
                    groups = df.groupby(x_col_for_box)[y_col].apply(list)
                    if len(groups) > CHART_MAX_CATEGORIES:
                        medians = df.groupby(x_col_for_box)[y_col].median().sort_values(ascending=False)
                        top_groups = medians.head(CHART_MAX_CATEGORIES).index.tolist()
                        groups = groups[top_groups]
                    ax.boxplot(groups.values, labels=groups.index)
                    ax.set_xlabel(x_col.replace("_", " ").title() if not series_col else "Category")
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
                else:
                    # Single box plot
                    ax.boxplot(df[y_col].dropna())
                ax.set_ylabel(y_label if y_label else _humanize_y_label(y_col))
            else:
                logger.warning("Box plot requires a numeric y_column")
                plt.close(fig)
                return ""

        else:
            logger.warning("Unknown chart type: %s", chart_type)
            plt.close(fig)
            return ""

        ax.set_title(chart_title)
        plt.tight_layout()

        filename = f"chart_{int(time.time())}_{chart_type}.png"
        out_path = os.path.join(CHART_OUTPUT_DIR, filename)
        plt.savefig(out_path, dpi=CHART_DPI, bbox_inches="tight")
        plt.close(fig)

        md_parts = [f"<!-- Alt-text: {alt_text} -->"]
        if return_mode == "path":
            md_parts.append(f"![Chart]({out_path})")
        elif return_mode == "base64":
            with open(out_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            md_parts.append(f"![Chart](data:image/png;base64,{b64})")
        else:
            base = CHART_BASE_URL.rstrip("/")
            md_parts.append(f"![Chart]({base}/charts/{filename})")

        if include_table:
            table_md = generate_data_table(data)
            if table_md:
                md_parts.append("")
                md_parts.append(table_md)

        result = "\n".join(md_parts)
        logger.info("CHART_DEBUG_MATPLOTLIB: success, markdown_len=%d", len(result))
        return result

    except Exception as e:
        logger.error("CHART_DEBUG_MATPLOTLIB: generation failed: %s", e, exc_info=True)
        if fig is not None:
            plt.close(fig)
        return ""


# ═════════════════════════════════════════════════════════════════════════════
# UNIFIED ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def generate_chart(
    data: List[Dict],
    chart_type: str,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    mode: Optional[str] = None,
    auth_role: Optional[str] = None,
    include_table: bool = True
) -> str:
    """
    Generate a chart in the configured mode.
    """
    logger.info("CHART_DEBUG_GENERATE: called type=%s data_rows=%d x=%s y=%s y_label=%s", 
                chart_type, len(data) if data else 0, x_column, y_column, y_label)

    is_safe, reason = check_privacy_safe(data, auth_role)
    if not is_safe:
        logger.warning("Chart blocked: %s", reason)
        table_md = generate_data_table(data) if data else ""
        return f"<!-- Chart blocked: {reason} -->\n\n{table_md}"

    if auth_role == "employee":
        logger.info("Employee role: suppressing chart, returning table only.")
        return generate_data_table(data) if include_table else ""

    mode = (mode or CHART_MODE).lower()
    logger.info("CHART_DEBUG_GENERATE: mode=%s type=%s data_rows=%d", mode, chart_type, len(data))

    # Auto-correct chart type for pre-aggregated data
    # If data has pre-binned categories (age_group, salary_range, etc.), force bar chart instead of hist to avoid "Frequency" ylabel
    df = pd.DataFrame(data) if data else pd.DataFrame()
    if not df.empty and chart_type == "hist":
        picked_x, picked_y, is_pre_agg, _ = _pick_columns(data, chart_type)
        if is_pre_agg and picked_x:
            # Use picked_x (the actual column in data)            
            if _is_pre_aggregated_category_col(picked_x):
                logger.info("CHART_DEBUG_GENERATE: auto-correcting hist to bar for pre-aggregated category '%s'", picked_x)
                chart_type = "bar"

    if mode == "auto":
        if chart_type in ("bar", "barh", "pie", "line") and len(data) <= CHART_MAX_CATEGORIES:
            mode = "mermaid"
        else:
            mode = "matplotlib"

    # Defensive guard: Mermaid only supports bar, barh, pie, line
    if mode == "mermaid" and chart_type not in ("bar", "barh", "pie", "line"):
        logger.warning("Chart type '%s' not supported in mermaid mode, switching to matplotlib", chart_type)
        mode = "matplotlib"

    if mode == "mermaid":
        result = generate_mermaid_chart(data, chart_type, x_column, y_column, title, y_label=y_label, include_table=include_table)
    elif mode == "base64":
        result = generate_matplotlib_chart(data, chart_type, x_column, y_column, title, y_label=y_label, return_mode="base64", include_table=include_table)
    else:
        result = generate_matplotlib_chart(data, chart_type, x_column, y_column, title, y_label=y_label, return_mode="url", include_table=include_table)

    logger.info("CHART_DEBUG_GENERATE: result_len=%d empty=%s", len(result), result == "")
    return result