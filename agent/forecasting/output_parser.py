"""
agent/forecasting/output_parser.py
Validates and normalizes sandbox forecast output.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hr_agent.forecasting")


@dataclass
class ForecastOutput:
    forecasts: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    metrics: Dict[str, Any]
    data_sources_used: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


class OutputParseError(Exception):
    pass


def validate_output(data: Dict[str, Any]) -> ForecastOutput:
    """Validate sandbox output against expected schema.

    Expected schema:
      {
        "forecasts": [
          {
            "period": "2026-10-01",
            "point_estimate": 1315,
            "lower_80": 1280,
            "upper_80": 1350,
            "lower_95": 1260,
            "upper_95": 1370
          }
        ],
        "model_info": {
          "model_type": "statsforecast",
          "training_rows": 8,
          "features_used": ["trend", "yearly_seasonality"],
          "business_overrides_applied": ["+45 per new office"]
        },
        "metrics": {
          "mape": 0.025,
          "rmse": 12.3
        }
      }
    """
    if not isinstance(data, dict):
        raise OutputParseError(f"Output must be a JSON object, got {type(data).__name__}")

    raw = dict(data)
    forecasts = data.get("forecasts", [])
    if not isinstance(forecasts, list) or len(forecasts) == 0:
        raise OutputParseError("'forecasts' must be a non-empty list")

    model_info = data.get("model_info", {})
    if not isinstance(model_info, dict):
        raise OutputParseError("'model_info' must be an object")

    metrics = data.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    # Validate individual forecast rows
    validated_forecasts = []
    for i, row in enumerate(forecasts):
        if not isinstance(row, dict):
            raise OutputParseError(f"forecast[{i}] must be an object")
        period = row.get("period", "")
        point = _safe_float(row.get("point_estimate"))
        if point is None:
            raise OutputParseError(f"forecast[{i}] missing or invalid 'point_estimate'")
        validated_forecasts.append({
            "period": str(period),
            "point_estimate": point,
            "lower_80": _safe_float(row.get("lower_80")),
            "upper_80": _safe_float(row.get("upper_80")),
            "lower_95": _safe_float(row.get("lower_95")),
            "upper_95": _safe_float(row.get("upper_95")),
        })

    # Fill missing confidence intervals from historical std dev if possible
    if any(f["lower_80"] is None for f in validated_forecasts):
        _fill_missing_cis(validated_forecasts, metrics)

    return ForecastOutput(
        forecasts=validated_forecasts,
        model_info=model_info,
        metrics=metrics,
        data_sources_used=data.get("data_sources_used", []),
        raw=raw,
    )


def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _fill_missing_cis(forecasts: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    """Generate missing CIs from historical RMSE if available."""
    rmse = _safe_float(metrics.get("rmse"))
    if rmse is None or rmse <= 0:
        rmse = abs(forecasts[0]["point_estimate"]) * 0.05 if forecasts else 1.0

    for f in forecasts:
        if f["lower_80"] is None:
            f["lower_80"] = f["point_estimate"] - 1.28 * rmse
        if f["upper_80"] is None:
            f["upper_80"] = f["point_estimate"] + 1.28 * rmse
        if f["lower_95"] is None:
            f["lower_95"] = f["point_estimate"] - 1.96 * rmse
        if f["upper_95"] is None:
            f["upper_95"] = f["point_estimate"] + 1.96 * rmse


def normalize_dates(output: ForecastOutput) -> ForecastOutput:
    """Normalize all period strings to ISO 8601 (YYYY-MM-DD)."""
    for f in output.forecasts:
        p = str(f.get("period", ""))
        if len(p) == 7:  # YYYY-MM
            p = f"{p}-01"
        f["period"] = p
    return output


def check_quality(output: ForecastOutput) -> List[str]:
    """Run data quality checks and return list of warnings."""
    warnings: List[str] = []
    forecasts = output.forecasts
    if len(forecasts) < 2:
        warnings.append("Forecast has fewer than 2 periods; variance checks skipped")
        return warnings

    # Monotonicity of point estimates (not strictly required, but suspicious if wildly oscillating)
    points = [f["point_estimate"] for f in forecasts]
    if all(p >= 0 for p in points):
        # Check for negative values in HR metrics
        pass
    if any(p < 0 for p in points):
        warnings.append("Negative point estimates detected; verify data bounds")

    # CI bounds sanity
    for f in forecasts:
        if f["lower_80"] is not None and f["upper_80"] is not None:
            if f["lower_80"] > f["upper_80"]:
                warnings.append(f"Lower CI > Upper CI for period {f['period']}")
        if f["lower_95"] is not None and f["upper_95"] is not None:
            if f["lower_95"] > f["upper_95"]:
                warnings.append(f"Lower 95% CI > Upper 95% CI for period {f['period']}")

    # Reasonable bounds: headcount should not swing >50% in one period
    for i in range(1, len(points)):
        if points[i - 1] > 0:
            pct_change = abs(points[i] - points[i - 1]) / points[i - 1]
            if pct_change > 0.5:
                warnings.append(
                    f"Large swing in period {forecasts[i]['period']}: "
                    f"{pct_change:.1%} change from previous period"
                )

    return warnings


def parse_and_validate(data: Dict[str, Any]) -> ForecastOutput:
    """Full validation pipeline."""
    output = validate_output(data)
    normalize_dates(output)
    warnings = check_quality(output)
    if warnings:
        for w in warnings:
            logger.warning("Forecast quality warning: %s", w)
    return output
