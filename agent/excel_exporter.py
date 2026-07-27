import os
import glob
import time
import math
import re
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
import xlsxwriter

logger = logging.getLogger("hr_agent")

# ==============================
# CONFIG: EXCEL EXPORT
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.getenv("EXPORT_DIR", os.path.join(BASE_DIR, "exports"))
AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", "http://localhost:8000")
EXPORT_MAX_AGE_HOURS = int(os.getenv("EXPORT_MAX_AGE_HOURS", "24"))
os.makedirs(EXPORT_DIR, exist_ok=True)

# ==============================
# COLORBLIND PALETTE
# ==============================
COLORBLIND_PALETTE = [
    "#648FFF", "#FE6100", "#DC267F", "#785EF0", "#FFB000", "#3DDC97",
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",
    "#CC79A7", "#999999",
]

PERSONAL_IDENTIFIERS = {"emp_id", "employee_id", "email", "name", "first_name", "last_name", "national_id"}
AGGREGATE_VALUE_COLS = {"count", "total", "sum", "avg", "average", "value", "amount", "frequency", "employees", "employee_count"}

CHART_MAX_CATEGORIES = 20
MAX_CELL_LENGTH = 32_767

# ==============================
# SANITIZER FUNCTIONS
# ==============================

def _sanitize_value(v):
    """Return a value safe for XlsxWriter cells."""
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
    if isinstance(v, (np.floating,)):
        if np.isnan(v) or np.isinf(v):
            return ""
    return v


def _sanitize_stat(v):
    """Sanitize a statistical value for Excel writing."""
    v = _sanitize_value(v)
    if v == "":
        return 0
    return v


_SHEET_BAD_CHARS = set("\\/?*[]:")

def _safe_sheet_name(name: str, max_len: int = 31) -> str:
    cleaned = "".join(c if c not in _SHEET_BAD_CHARS else "_" for c in name)
    cleaned = cleaned.strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip()
    if not cleaned:
        cleaned = "Sheet"
    return cleaned


def _to_excel_datetime(dt) -> Optional[datetime]:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    return None


def _truncate_string(value: str) -> str:
    if isinstance(value, str) and len(value) > MAX_CELL_LENGTH:
        return value[:MAX_CELL_LENGTH - 3] + "..."
    return value


def _ensure_writable(filepath: str) -> str:
    if os.path.exists(filepath):
        try:
            with open(filepath, 'a'):
                pass
        except (IOError, PermissionError):
            base, ext = os.path.splitext(filepath)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filepath = f"{base}_{timestamp}{ext}"
            logger.warning("File locked, using alternate path: %s", filepath)
    return filepath


# ==============================
# COLUMN AUTO-DETECTION
# ==============================

def _pick_columns(data: List[Dict], chart_type: str = "bar"):
    """Auto-detect x (categorical) and y (numeric) columns."""
    if not data or not isinstance(data[0], dict):
        return None, None
    df = pd.DataFrame(data)
    all_cols = list(df.columns)

    # Pre-aggregated detection
    agg_cols = [
        c for c in all_cols
        if c.lower() in AGGREGATE_VALUE_COLS
        or any(c.lower().startswith(kw + "_") or c.lower().endswith("_" + kw) for kw in AGGREGATE_VALUE_COLS)
    ]
    non_agg_cols = [c for c in all_cols if c not in agg_cols]

    if len(agg_cols) >= 1 and len(non_agg_cols) >= 1:
        x_col = non_agg_cols[-1]
        y_col = agg_cols[0]
        return x_col, y_col

    if len(all_cols) == 2:
        c0, c1 = all_cols[0], all_cols[1]
        if any(kw in c0.lower() for kw in ("count", "total", "sum", "avg", "average")):
            return c1, c0
        if any(kw in c1.lower() for kw in ("count", "total", "sum", "avg", "average")):
            return c0, c1

    numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in all_cols if c not in numeric_cols]

    preferred_cat = [
        "department", "job_title", "status", "gender", "location",
        "hire_year", "hire_month", "hire_quarter", "leave_type", "job_level",
        "age_group", "salary_range", "salary_band", "tenure_group",
        "age", "year", "month", "quarter"
    ]
    x_col = next((cc for cc in categorical_cols if cc.lower() in preferred_cat), None)
    if not x_col and categorical_cols:
        x_col = categorical_cols[0]

    preferred_num = ["count", "total", "sum", "avg", "salary", "age", "leave_balance", "turnover_rate"]
    y_col = next((nc for nc in numeric_cols if nc.lower() in preferred_num), None)
    if not y_col and numeric_cols:
        y_col = numeric_cols[0]

    return x_col, y_col


# ==============================
# PRIVACY GUARD
# ==============================

def _check_privacy_safe(data: List[Dict]) -> tuple[bool, str]:
    """Return (is_safe, reason) — blocks if personal identifiers are present."""
    if not data or not isinstance(data, list):
        return False, "No data to export."
    if data and isinstance(data[0], dict):
        keys = {k.lower() for k in data[0].keys()}
        personal_keys = keys & PERSONAL_IDENTIFIERS
        if personal_keys:
            return False, f"Privacy block: Data contains individual identifiers ({', '.join(personal_keys)})."
    return True, ""


# ==============================
# LEGACY DATA-ONLY EXPORT
# ==============================

def export_to_excel(data: List[Dict], prefix: str = "export", strict_privacy: bool = True) -> str:
    """Generate Excel file and return public download URL. Uses XlsxWriter."""
    if not data:
        logger.warning("export_to_excel: empty data")
        return ""

    if strict_privacy:
        is_safe, reason = _check_privacy_safe(data)
        if not is_safe:
            logger.warning("export_to_excel blocked by privacy guard: %s", reason)
            return ""

    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    filepath = os.path.join(EXPORT_DIR, filename)
    filepath = _ensure_writable(filepath)

    df = pd.DataFrame(data)
    headers = list(df.columns)

    try:
        with xlsxwriter.Workbook(filepath, {'nan_inf_to_errors': True}) as workbook:
            worksheet = workbook.add_worksheet("Data")

            # Header format
            header_fmt = workbook.add_format({
                'bold': True,
                'font_color': 'white',
                'bg_color': '#366092',
                'align': 'center',
                'valign': 'vcenter',
                'border': 1,
            })

            # Data format
            data_fmt = workbook.add_format({
                'valign': 'vcenter',
                'border': 1,
            })
            num_fmt = workbook.add_format({
                'valign': 'vcenter',
                'border': 1,
                'num_format': '#,##0',
            })
            date_fmt = workbook.add_format({
                'valign': 'vcenter',
                'border': 1,
                'num_format': 'YYYY-MM-DD HH:MM',
            })

            # Detect column types
            is_numeric = [pd.api.types.is_numeric_dtype(df[c]) for c in headers]
            is_datetime = [pd.api.types.is_datetime64_any_dtype(df[c]) for c in headers]

            # Write headers
            for col_idx, header in enumerate(headers):
                worksheet.write(0, col_idx, header, header_fmt)

            # Write data with sanitization
            for row_idx, row in enumerate(df.itertuples(index=False), 1):
                for col_idx, value in enumerate(row):
                    safe_val = _sanitize_value(value)
                    safe_val = _truncate_string(safe_val) if isinstance(safe_val, str) else safe_val

                    if is_datetime[col_idx]:
                        safe_val = _to_excel_datetime(safe_val)
                        worksheet.write(row_idx, col_idx, safe_val, date_fmt)
                    elif is_numeric[col_idx]:
                        worksheet.write(row_idx, col_idx, safe_val, num_fmt)
                    else:
                        worksheet.write(row_idx, col_idx, safe_val, data_fmt)

            # Auto-adjust column widths (calibrated for proportional font)
            for col_idx, header in enumerate(headers):
                max_len = len(str(header))
                for row in df.itertuples(index=False):
                    val = row[col_idx]
                    if val is not None:
                        cell_len = len(str(val))
                        max_len = max(max_len, cell_len)
                # Calibrated width for Arial/Calibri proportional rendering
                adjusted_width = min(max_len * 0.6 + 2, 50)
                worksheet.set_column(col_idx, col_idx, adjusted_width)

            # Freeze panes at A2 (after all data written)
            worksheet.freeze_panes(1, 0)

        logger.info("Excel export generated: %s (%d rows)", filepath, len(data))
        return f"{AGENT_BASE_URL}/exports/{filename}"

    except Exception as e:
        logger.error("Excel export failed: %s", e, exc_info=True)
        return ""


# ==============================
# ENTERPRISE CHART-ENABLED EXPORT
# ==============================

def export_to_excel_with_chart(
    data: List[Dict],
    chart_type: str = "bar",
    x_column: str = None,
    y_column: str = None,
    title: str = None,
    prefix: str = "export",
    auth_role: str = "employee",
    metadata: dict = None,
    strict_privacy: bool = True,
) -> str:
    """
    Enterprise-quality BI report export using XlsxWriter.

    Generates a workbook with 3–5 worksheets:
      1. Executive Summary: report title, generation metadata, key stats,
         insights/observations, and native Excel chart.
      2. Distribution Table: formatted table with % of Total, auto-filter,
         freeze panes, bold totals row.
      3. Chart: dedicated chart sheet with native Excel chart (editable).
      4. Raw Data: all underlying data with proper types, auto-width, freeze panes.
      5. Metadata (optional): audit & provenance information.

    Returns a public download URL on success, or "" when the export cannot
    be produced (privacy block, missing columns, unsupported type, or any error).

    Role-based behavior:
      - Admin/Manager: Full workbook with all sheets, charts, and metadata.
      - Employee: Chart sheets show placeholder; data sheets remain functional.
    """
    if not data:
        logger.warning("export_to_excel_with_chart: empty data")
        return ""

    # ── Privacy guard ──
    is_safe, reason = _check_privacy_safe(data)
    if not is_safe and strict_privacy:
        logger.warning("Chart export blocked: %s. Returning empty; caller should fall back to data-only.", reason)
        return ""

    # ── Role guard ──
    include_chart = (auth_role != "employee")
    if not include_chart:
        logger.info("Employee role: chart suppressed; generating data-rich workbook without chart.")

    # ── Auto-detect columns ──
    if not x_column or not y_column:
        detected_x, detected_y = _pick_columns(data, chart_type)
        x_column = x_column or detected_x
        y_column = y_column or detected_y
        logger.info("Auto-detected columns: x=%s, y=%s", x_column, y_column)

    if not x_column or not y_column:
        logger.warning("Could not determine x/y columns for chart. Returning empty.")
        return ""

    df = pd.DataFrame(data)
    if x_column not in df.columns or y_column not in df.columns:
        logger.warning(
            "Specified columns not found in data. x=%s y=%s cols=%s",
            x_column, y_column, list(df.columns)
        )
        return ""

    # Ensure y_column is numeric
    df[y_column] = pd.to_numeric(df[y_column], errors="coerce")
    df = df.dropna(subset=[x_column, y_column])
    if df.empty:
        logger.warning("No valid data after numeric coercion. Returning empty.")
        return ""

    # ── Build distribution table with % of Total ──
    total_y = df[y_column].sum()
    df_sorted = df.sort_values(by=y_column, ascending=False).reset_index(drop=True)
    df_sorted["pct_of_total"] = df_sorted[y_column].apply(
        lambda v: round((v / total_y) * 100, 1) if total_y else 0.0
    )

    # ── Cap categories for chart readability ──
    original_row_count = len(df_sorted)
    if len(df_sorted) > CHART_MAX_CATEGORIES:
        logger.info("Capping chart to %d categories (from %d)", CHART_MAX_CATEGORIES, len(df_sorted))
        if chart_type == "line":
            df_chart = df_sorted.head(CHART_MAX_CATEGORIES).copy()
        else:
            df_chart = df_sorted.nlargest(CHART_MAX_CATEGORIES, y_column).copy()
    else:
        df_chart = df_sorted.copy()

    # ── Derive insights ──
    insights = _derive_insights(df_sorted, x_column, y_column, total_y)

    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    filepath = os.path.join(EXPORT_DIR, filename)
    filepath = _ensure_writable(filepath)

    # Chart type mapping
    xlsx_chart_type = {
        "bar": "column",
        "barh": "bar",
        "pie": "pie",
        "line": "line",
        "area": "area",
    }.get(chart_type)

    if xlsx_chart_type is None:
        logger.warning("Unsupported chart type '%s'. Returning empty.", chart_type)
        return ""

    try:
        with xlsxwriter.Workbook(filepath, {'nan_inf_to_errors': True}) as workbook:
            workbook.set_calc_mode('auto')

            # ── Define formats ──
            title_fmt = workbook.add_format({
                'bold': True, 'font_size': 18, 'font_color': '#1F4E79',
                'align': 'left', 'valign': 'vcenter',
            })
            subtitle_fmt = workbook.add_format({
                'bold': True, 'font_size': 12, 'font_color': '#366092',
                'align': 'left', 'valign': 'vcenter',
            })
            label_fmt = workbook.add_format({
                'bold': True, 'font_size': 10, 'font_color': '#366092',
            })
            value_fmt = workbook.add_format({
                'font_size': 10,
            })
            header_fmt = workbook.add_format({
                'bold': True, 'font_color': 'white', 'bg_color': '#366092',
                'align': 'center', 'valign': 'vcenter', 'border': 1,
            })
            data_fmt = workbook.add_format({
                'valign': 'vcenter', 'border': 1,
            })
            num_fmt = workbook.add_format({
                'valign': 'vcenter', 'border': 1, 'num_format': '#,##0',
            })
            pct_fmt = workbook.add_format({
                'valign': 'vcenter', 'border': 1, 'num_format': '0.0%',
            })
            bold_num_fmt = workbook.add_format({
                'bold': True, 'valign': 'vcenter', 'border': 1, 'num_format': '#,##0',
                'bg_color': '#E7E6E6',
            })
            bold_pct_fmt = workbook.add_format({
                'bold': True, 'valign': 'vcenter', 'border': 1, 'num_format': '0.0%',
                'bg_color': '#E7E6E6',
            })
            date_fmt = workbook.add_format({
                'valign': 'vcenter', 'border': 1, 'num_format': 'YYYY-MM-DD HH:MM',
            })
            meta_title_fmt = workbook.add_format({
                'bold': True, 'font_size': 14, 'font_color': '#1F4E79',
                'align': 'left', 'valign': 'vcenter',
            })
            insight_fmt = workbook.add_format({
                'font_size': 10, 'text_wrap': True, 'valign': 'top',
            })
            placeholder_fmt = workbook.add_format({
                'font_size': 11, 'italic': True, 'font_color': '#999999',
                'align': 'center', 'valign': 'vcenter', 'text_wrap': True,
            })

            chart_title = title or f"{x_column.replace('_', ' ').title()} Distribution"
            y_label = y_column.replace("_", " ").title()
            x_label = x_column.replace("_", " ").title()
            generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ═══════════════════════════════════════════════════════════════
            # SHEET 1: EXECUTIVE SUMMARY
            # ═══════════════════════════════════════════════════════════════
            ws_summary_name = _safe_sheet_name("Executive Summary")
            ws_summary = workbook.add_worksheet(ws_summary_name)
            ws_summary.hide_gridlines(2)

            # Title block (rows 1-2, 0-based: 0-1)
            ws_summary.merge_range(0, 1, 0, 6, chart_title + " Analysis", title_fmt)
            ws_summary.set_row(0, 32)

            # Subtitle / filter line
            filter_text = metadata.get("filter", "Active Employees") if metadata else "Active Employees"
            ws_summary.merge_range(1, 1, 1, 6, f"Generated: {generated_at}    |    Filter: {filter_text}", value_fmt)
            ws_summary.set_row(1, 18)

            # Key Statistics block (rows 3-8, 0-based)
            stats_start = 3
            ws_summary.write(stats_start, 1, "Summary", subtitle_fmt)
            ws_summary.set_row(stats_start, 22)

            stats = [
                ("Total Employees", _sanitize_stat(total_y)),
                ("Distinct Categories", len(df_sorted)),
            ]
            # Largest / Smallest
            if not df_sorted.empty:
                largest_row = df_sorted.iloc[0]
                smallest_row = df_sorted.iloc[-1]
                stats.append((
                    f"Largest {x_label}",
                    f"{largest_row[x_column]} ({largest_row[y_column]:,.0f})"
                ))
                stats.append((
                    f"Smallest {x_label}",
                    f"{smallest_row[x_column]} ({smallest_row[y_column]:,.0f})"
                ))

            for i, (stat_name, stat_val) in enumerate(stats, 1):
                r = stats_start + i
                ws_summary.write(r, 1, stat_name + ":", label_fmt)
                ws_summary.write(r, 2, stat_val, value_fmt)

            # Insights / Observations block
            insight_start = stats_start + len(stats) + 2
            ws_summary.write(insight_start, 1, "Observations", subtitle_fmt)
            ws_summary.set_row(insight_start, 22)
            for i, insight in enumerate(insights, 1):
                r = insight_start + i
                ws_summary.merge_range(r, 1, r, 6, f"• {insight}", insight_fmt)
                ws_summary.set_row(r, 18)

            # Column widths for Summary
            ws_summary.set_column(0, 0, 3)   # A
            ws_summary.set_column(1, 1, 22)  # B
            ws_summary.set_column(2, 2, 28)  # C
            ws_summary.set_column(3, 6, 14)  # D-G

            # ═══════════════════════════════════════════════════════════════
            # SHEET 2: DISTRIBUTION TABLE
            # ═══════════════════════════════════════════════════════════════
            ws_dist_name = _safe_sheet_name("Distribution Table")
            ws_dist = workbook.add_worksheet(ws_dist_name)

            # Table header
            dist_headers = [x_label, y_label, "% of Total"]
            for col_idx, h in enumerate(dist_headers):
                ws_dist.write(0, col_idx, h, header_fmt)

            # Data rows
            for row_idx, row in df_sorted.iterrows():
                r = row_idx + 1
                ws_dist.write(r, 0, row[x_column], data_fmt)
                ws_dist.write(r, 1, row[y_column], num_fmt)
                ws_dist.write(r, 2, row["pct_of_total"] / 100.0, pct_fmt)

            # Totals row
            total_row = len(df_sorted) + 1
            ws_dist.write(total_row, 0, "Total", bold_num_fmt)
            ws_dist.write(total_row, 1, total_y, bold_num_fmt)
            ws_dist.write(total_row, 2, 1.0, bold_pct_fmt)

            # Auto-width + freeze + filter
            ws_dist.set_column(0, 0, 22)
            ws_dist.set_column(1, 1, 14)
            ws_dist.set_column(2, 2, 14)
            ws_dist.freeze_panes(1, 0)
            ws_dist.autofilter(0, 0, total_row, 2)

            # ═══════════════════════════════════════════════════════════════
            # SHEET 3: CHART (native Excel chart, editable)
            # ═══════════════════════════════════════════════════════════════
            ws_chart_name = _safe_sheet_name("Chart")
            ws_chart = workbook.add_worksheet(ws_chart_name)

            if include_chart:
                # Write chart data in hidden columns (A:B) on Chart sheet
                ws_chart.write(0, 0, x_column, header_fmt)
                ws_chart.write(0, 1, y_column, header_fmt)

                for i, row in df_chart.iterrows():
                    r = i + 1
                    cat_val = row[x_column]
                    val = row[y_column]
                    safe_cat = _sanitize_value(cat_val)
                    safe_val = _sanitize_value(val)
                    # Write category as string to force text-based category axis
                    ws_chart.write_string(r, 0, str(safe_cat) if safe_cat is not None else "", data_fmt)
                    ws_chart.write(r, 1, safe_val, num_fmt)

                num_cats = len(df_chart)
                cat_start_row = 1
                cat_end_row = num_cats

                chart = workbook.add_chart({'type': xlsx_chart_type})
                chart.set_title({'name': chart_title})
                chart.set_size({'width': 720, 'height': 480})

                if chart_type == "barh":
                    chart.set_y_axis({'name': x_label})
                    chart.set_x_axis({'name': y_label})
                else:
                    chart.set_y_axis({'name': y_label})
                    chart.set_x_axis({'name': x_label})

                if chart_type == "pie":
                    points = [
                        {'fill': {'color': COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]}}
                        for i in range(num_cats)
                    ]
                    chart.add_series({
                        'name': y_column,
                        'categories': [ws_chart_name, cat_start_row, 0, cat_end_row, 0],
                        'values': [ws_chart_name, cat_start_row, 1, cat_end_row, 1],
                        'points': points,
                        'data_labels': {'value': True, 'category': True},
                    })
                else:
                    chart.add_series({
                        'name': y_column,
                        'categories': [ws_chart_name, cat_start_row, 0, cat_end_row, 0],
                        'values': [ws_chart_name, cat_start_row, 1, cat_end_row, 1],
                        'fill': {'color': COLORBLIND_PALETTE[0]},
                        'data_labels': {'value': True},
                    })

                ws_chart.insert_chart(3, 3, chart)
            else:
                # Employee role placeholder
                ws_chart.merge_range(3, 1, 6, 5,
                    "Chart visualization is not available for employee role.\n\n"
                    "Please contact your HR manager or admin for chart access.",
                    placeholder_fmt)

            # ═══════════════════════════════════════════════════════════════
            # SHEET 4: RAW DATA
            # ═══════════════════════════════════════════════════════════════
            ws_data_name = _safe_sheet_name("Raw Data")
            ws_data = workbook.add_worksheet(ws_data_name)

            raw_headers = list(df.columns)
            # Exclude our computed pct_of_total from raw data display
            if "pct_of_total" in raw_headers:
                raw_headers.remove("pct_of_total")

            is_numeric = [pd.api.types.is_numeric_dtype(df[c]) for c in raw_headers]
            is_datetime = [pd.api.types.is_datetime64_any_dtype(df[c]) for c in raw_headers]

            # Write headers
            for col_idx, header in enumerate(raw_headers):
                ws_data.write(0, col_idx, header, header_fmt)

            # Write data with sanitization
            for row_idx, row in enumerate(df.itertuples(index=False), 1):
                for col_idx, header in enumerate(raw_headers):
                    # Get value by column name from the row
                    val = getattr(row, header)
                    safe_val = _sanitize_value(val)
                    safe_val = _truncate_string(safe_val) if isinstance(safe_val, str) else safe_val

                    if is_datetime[col_idx]:
                        safe_val = _to_excel_datetime(safe_val)
                        ws_data.write(row_idx, col_idx, safe_val, date_fmt)
                    elif is_numeric[col_idx]:
                        ws_data.write(row_idx, col_idx, safe_val, num_fmt)
                    else:
                        ws_data.write(row_idx, col_idx, safe_val, data_fmt)

            # Auto-adjust column widths
            for col_idx, header in enumerate(raw_headers):
                max_len = len(str(header))
                for row in df.itertuples(index=False):
                    val = getattr(row, header)
                    if val is not None:
                        cell_len = len(str(val))
                        max_len = max(max_len, cell_len)
                adjusted_width = min(max_len * 0.6 + 2, 50)
                ws_data.set_column(col_idx, col_idx, adjusted_width)

            ws_data.freeze_panes(1, 0)

            # ═══════════════════════════════════════════════════════════════
            # SHEET 5: METADATA (optional)
            # ═══════════════════════════════════════════════════════════════
            if metadata and isinstance(metadata, dict):
                ws_meta_name = _safe_sheet_name("Metadata")
                ws_meta = workbook.add_worksheet(ws_meta_name)
                ws_meta.hide_gridlines(2)

                ws_meta.merge_range(0, 1, 0, 3, "Audit & Provenance", meta_title_fmt)
                ws_meta.set_row(0, 28)

                ws_meta.write(2, 1, "Property", header_fmt)
                ws_meta.write(2, 2, "Value", header_fmt)

                row = 3
                for k, v in metadata.items():
                    safe_k = _truncate_string(str(k))
                    safe_v = _truncate_string(str(v))
                    ws_meta.write(row, 1, safe_k, data_fmt)
                    ws_meta.write(row, 2, safe_v, data_fmt)
                    row += 1

                ws_meta.set_column(0, 0, 3)
                ws_meta.set_column(1, 1, 25)
                ws_meta.set_column(2, 2, 60)

        logger.info(
            "Excel export generated: %s (%d rows, type=%s, x=%s, y=%s, chart=%s)",
            filepath, len(df), chart_type, x_column, y_column, include_chart
        )
        return f"{AGENT_BASE_URL}/exports/{filename}"

    except Exception as e:
        logger.error("Excel chart export failed: %s", e, exc_info=True)
        return ""


# ==============================
# INSIGHT DERIVATION
# ==============================

def _derive_insights(df: pd.DataFrame, x_col: str, y_col: str, total_y: float) -> List[str]:
    """Generate human-readable observations from distribution data."""
    insights = []
    if df.empty or total_y == 0:
        return ["No data available for analysis."]

    n = len(df)
    # Top category insight
    top = df.iloc[0]
    top_pct = round((top[y_col] / total_y) * 100, 1)
    insights.append(
        f"{top[x_col]} represents {top_pct}% of the total ({top[y_col]:,.0f} out of {total_y:,.0f})."
    )

    # Top-2 combined insight (if >= 2 categories)
    if n >= 2:
        top2_sum = df.iloc[:2][y_col].sum()
        top2_pct = round((top2_sum / total_y) * 100, 1)
        insights.append(
            f"{df.iloc[0][x_col]} and {df.iloc[1][x_col]} together account for {top2_pct}% of the total."
        )

    # Bottom category insight (if >= 3 categories and bottom is notably small)
    if n >= 3:
        bottom = df.iloc[-1]
        bottom_pct = round((bottom[y_col] / total_y) * 100, 1)
        if bottom_pct < 5.0:
            insights.append(
                f"{bottom[x_col]} is relatively small, representing only {bottom_pct}% of the total."
            )

    # Diversity insight
    if n >= 4:
        mid_pct = round((df.iloc[n // 2][y_col] / total_y) * 100, 1)
        insights.append(
            f"The distribution spans {n} categories with the median category at {mid_pct}%."
        )

    return insights


# ==============================
# CLEANUP
# ==============================

def cleanup_old_exports():
    """Delete exports older than EXPORT_MAX_AGE_HOURS."""
    cutoff = time.time() - (EXPORT_MAX_AGE_HOURS * 3600)
    count = 0
    for filepath in glob.glob(os.path.join(EXPORT_DIR, "*.xlsx")):
        if os.path.getmtime(filepath) < cutoff:
            try:
                os.remove(filepath)
                count += 1
            except Exception as e:
                logger.warning("Failed to delete old export %s: %s", filepath, e)
    if count > 0:
        logger.info("Cleaned up %d old export(s)", count)