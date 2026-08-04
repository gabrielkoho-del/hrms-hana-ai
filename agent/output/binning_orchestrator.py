"""
Client-side dynamic binning service.

Applies pandas-based binning to DAB result sets after retrieval. Supports
two flows:
  - post_aggregate: grouped data already has a count column (e.g., age=25, count=5)
  - pre_aggregate: raw records are binned then counted (e.g., date_of_birth -> age -> count)

Detection of which flow to use is handled by the caller via explicit `method`
in the bin_config; the executor performs dynamic detection before invoking
this service.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from agent.output.chart_generator import _is_aggregate_value_col


logger = logging.getLogger("hr_agent")


def apply_client_side_binning(
    items: List[Dict],
    bin_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Apply dynamic binning to raw data after fetching from DAB.

    Args:
        items: Raw records or grouped aggregates from DAB.
        bin_config: {
            "method": "post_aggregate" | "pre_aggregate",
            "column": str,
            "bins": List[float],
            "labels": List[str],
            "output_column": Optional[str],
            "calculate_age": Optional[bool],
        }

    Returns:
        Binned records ready for charting or summarization.
        On total failure, returns the original items unchanged.
    """
    if not items or not isinstance(items, list):
        logger.warning("CLIENT_SIDE_BINNING: no items to bin")
        return []

    method = bin_config.get("method", "post_aggregate")
    column = bin_config.get("column")
    bins = bin_config.get("bins", [0, 25, 35, 45, 55, 100])
    labels = bin_config.get("labels", ["18-25", "26-35", "36-45", "46-55", "56+"])
    output_column = bin_config.get(
        "output_column", f"{column}_group" if column else "binned_group"
    )

    logger.info(
        "CLIENT_SIDE_BINNING: method=%s column=%s output_column=%s bins=%s labels=%s items=%d",
        method, column, output_column, bins, labels, len(items),
    )

    df = pd.DataFrame(items)
    if df.empty:
        logger.warning("CLIENT_SIDE_BINNING: empty dataframe")
        return []

    if method == "post_aggregate":
        return _bin_post_aggregate(df, column, bins, labels, output_column, items)
    if method == "pre_aggregate":
        return _bin_pre_aggregate(df, column, bins, labels, output_column, bin_config, items)

    logger.warning("CLIENT_SIDE_BINNING: unknown method '%s'", method)
    return items


def _bin_post_aggregate(
    df: pd.DataFrame,
    column: str,
    bins: List[float],
    labels: List[str],
    output_column: str,
    original_items: List[Dict],
) -> List[Dict[str, Any]]:
    if column not in df.columns:
        logger.error(
            "CLIENT_SIDE_BINNING: column '%s' not found in %s", column, list(df.columns)
        )
        return original_items

    df[column] = pd.to_numeric(df[column], errors="coerce")
    initial_rows = len(df)
    df = df.dropna(subset=[column])
    if len(df) < initial_rows:
        logger.info("CLIENT_SIDE_BINNING: dropped %d non-numeric rows", initial_rows - len(df))

    df[output_column] = pd.cut(df[column], bins=bins, labels=labels, right=True, include_lowest=True)

    count_cols = [c for c in df.columns if _is_aggregate_value_col(c)]
    count_col = count_cols[0] if count_cols else None
    if count_col:
        result = df.groupby(output_column, observed=False)[count_col].sum().reset_index()
        logger.info("CLIENT_SIDE_BINNING: using aggregate column '%s' for summing", count_col)
    else:
        result = df.groupby(output_column, observed=False).size().reset_index(name="count")
        logger.warning(
            "CLIENT_SIDE_BINNING: no aggregate column found; using .size() (counts rows, not employees)"
        )

    result[output_column] = result[output_column].astype(str)
    binned = result.to_dict("records")
    total_count = sum(
        r.get(count_col, r.get("count", 0)) for r in binned
    ) if count_col else sum(r.get("count", 0) for r in binned)
    logger.info(
        "CLIENT_SIDE_BINNING: post_aggregate binned %d raw groups into %d bins, total_count=%s",
        len(original_items), len(binned), total_count,
    )
    return binned


def _bin_pre_aggregate(
    df: pd.DataFrame,
    column: str,
    bins: List[float],
    labels: List[str],
    output_column: str,
    bin_config: Dict[str, Any],
    original_items: List[Dict],
) -> List[Dict[str, Any]]:
    if bin_config.get("calculate_age"):
        dob_col = column
        if dob_col not in df.columns:
            logger.error(
                "CLIENT_SIDE_BINNING: DOB column '%s' not found in %s",
                dob_col, list(df.columns),
            )
            return original_items

        today = datetime.today()
        df[dob_col] = pd.to_datetime(df[dob_col], errors="coerce")
        initial_rows = len(df)
        df = df.dropna(subset=[dob_col])
        if len(df) < initial_rows:
            logger.info("CLIENT_SIDE_BINNING: dropped %d invalid DOB rows", initial_rows - len(df))

        df["age"] = df[dob_col].apply(
            lambda x: (
                today.year - x.year - ((today.month, today.day) < (x.month, x.day))
                if pd.notna(x) else None
            )
        )
        column = "age"

    if column not in df.columns:
        logger.error("CLIENT_SIDE_BINNING: column '%s' not found after preprocessing", column)
        return original_items

    df[column] = pd.to_numeric(df[column], errors="coerce")
    initial_rows = len(df)
    df = df.dropna(subset=[column])
    if len(df) < initial_rows:
        logger.info("CLIENT_SIDE_BINNING: dropped %d non-numeric rows", initial_rows - len(df))

    df[output_column] = pd.cut(df[column], bins=bins, labels=labels, right=True, include_lowest=True)
    result = df.groupby(output_column, observed=False).size().reset_index(name="count")
    result[output_column] = result[output_column].astype(str)
    binned = result.to_dict("records")
    logger.info(
        "CLIENT_SIDE_BINNING: pre_aggregate binned %d raw records into %d bins",
        len(original_items), len(binned),
    )
    return binned
