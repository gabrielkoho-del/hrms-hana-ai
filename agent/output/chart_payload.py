"""
Chart.js payload builder for the custom HR chat UI side-channel.

Converts the agent's internal chart_config + chartable_data into a
Chart.js-shaped dict {type, title, labels, datasets} that the custom
React UI renders natively via Chart.js (no PNG/mermaid dependency).

This is the structured-data counterpart to the markdown chart artifact
(![Chart](url) / ```mermaid```) that LibreChat consumes. The streaming
endpoint emits the payload as an hr_chart side-channel chunk when the
client sends X-HR-Client: custom-ui.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hr_agent")

# Map internal chart types to Chart.js types.
# Chart.js supports: bar, barh (horizontalBar), line, pie, doughnut, radar,
#                    polarArea, bubble, scatter, mixed.
# Internal types not directly supported by Chart.js fall back to "bar".
_CHARTJS_TYPE_MAP = {
    "bar": "bar",
    "barh": "horizontalBar",
    "line": "line",
    "pie": "pie",
    "doughnut": "doughnut",
    "area": "line",  # Chart.js area = line with fill:true (set in options)
    "hist": "bar",
    "box": "bar",
    "scatter": "scatter",
    "heatmap": "bar",  # Chart.js has no native heatmap; bar is the closest
    "gauge": "doughnut",  # gauge rendered as doughnut with needle (options)
    "radar": "radar",
}


def build_chartjs_payload(
    chart_config: Optional[Dict],
    chartable_data: List[Dict],
    title: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build a Chart.js config dict from the agent's chart metadata.

    Returns None if there is no chartable data or no chart config.
    The returned dict has the shape:
        {
            "type": "bar" | "line" | "pie" | ...,
            "title": "Human Title",
            "labels": ["label1", "label2", ...],
            "datasets": [{"label": "Series 1", "data": [1, 2, ...]}],
            "options": {...}  # optional Chart.js options (legend, scales, etc.)
        }
    """
    if not chartable_data or not isinstance(chartable_data[0], dict):
        return None

    if not chart_config or not isinstance(chart_config, dict):
        return None

    chart_type = chart_config.get("type", "bar")
    js_type = _CHARTJS_TYPE_MAP.get(chart_type, "bar")

    x_col = chart_config.get("x_column")
    y_col = chart_config.get("y_column")
    chart_title = title or chart_config.get("title") or ""

    # Determine x (labels) and y (values) columns from data.
    first_row = chartable_data[0]
    all_cols = list(first_row.keys())

    if not x_col and all_cols:
        x_col = all_cols[0]
    if not y_col:
        # Pick the first numeric-looking column that isn't the x column.
        import pandas as pd
        df = pd.DataFrame(chartable_data)
        numeric_cols = [
            c for c in df.columns
            if c != x_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        y_col = numeric_cols[0] if numeric_cols else (all_cols[1] if len(all_cols) > 1 else all_cols[0])

    # Build labels and data arrays.
    labels = [str(row.get(x_col, "")) for row in chartable_data]

    # Handle multi-series: if chart_config has metrics, create one dataset per metric.
    metrics = chart_config.get("metrics", [])
    multi_series = chart_config.get("multi_series", False)

    datasets = []
    if multi_series and metrics:
        for metric in metrics:
            m_field = metric.get("field", y_col)
            m_label = metric.get("label") or m_field.replace("_", " ").title()
            data = [float(row.get(m_field, 0) or 0) for row in chartable_data]
            datasets.append({"label": m_label, "data": data})
    else:
        data = []
        for row in chartable_data:
            val = row.get(y_col, 0)
            try:
                data.append(float(val) if val is not None else 0.0)
            except (TypeError, ValueError):
                data.append(0.0)
        datasets.append({"label": y_col.replace("_", " ").title() if y_col else "Value", "data": data})

    # Build options based on chart type.
    options = {}
    if js_type in ("bar", "horizontalBar"):
        options["indexAxis"] = "y" if js_type == "horizontalBar" else "x"
        options["scales"] = {"y": {"beginAtZero": True}}
    elif js_type == "line":
        options["scales"] = {"y": {"beginAtZero": True}}
        if chart_config.get("trend_line"):
            options["plugins"] = {"trendLine": True}
    elif js_type == "pie" or js_type == "doughnut":
        options["plugins"] = {"legend": {"position": "right"}}
    elif js_type == "radar":
        options["scales"] = {"r": {"angleLines": {"display": True}, "suggestedMin": 0}}

    payload = {
        "type": js_type,
        "title": chart_title,
        "labels": labels,
        "datasets": datasets,
    }
    if options:
        payload["options"] = options

    logger.info("chart_payload: built type=%s labels=%d datasets=%d",
                js_type, len(labels), len(datasets))
    return payload
