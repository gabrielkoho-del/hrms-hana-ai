"""Binning configuration and metadata inference for multi-tenant schemas."""
import logging
from typing import Dict, List, Optional, Any

from agent.config import LARGE_RESULT_THRESHOLD

logger = logging.getLogger("hr_agent")

BINNING_MAP = {
    "age": {
        "aliases": {
            "age", "date_of_birth", "dob", "birth_date", "birthdate",
            "dateofbirth", "birthday", "date_birth", "d_o_b"
        },
        "config": {
            "method": "post_aggregate",
            "column": "age",
            "bins": [18, 25, 35, 45, 55, 100],
            "labels": ["18-25", "26-35", "36-45", "46-55", "56+"]
        }
    },
    "salary": {
        "aliases": {
            "salary", "basic_salary", "gross_salary", "net_salary",
            "monthly_salary", "annual_salary", "pay", "wage",
            "compensation", "remuneration", "basic_pay", "total_salary"
        },
        "config": {
            "method": "post_aggregate",
            "column": "salary",
            "bins": [0, 3000, 5000, 8000, 12000, 999999],
            "labels": ["<3K", "3-5K", "5-8K", "8-12K", "12K+"]
        }
    },
    "tenure": {
        "aliases": {
            "tenure", "years_of_service", "service_years", "length_of_service",
            "employment_duration", "years_with_company", "company_tenure",
            "service_length", "service_duration"
        },
        "config": {
            "method": "post_aggregate",
            "column": "tenure",
            "bins": [0, 1, 3, 5, 10, 100],
            "labels": ["<1yr", "1-3yr", "3-5yr", "5-10yr", "10yr+"]
        }
    }
}


def _resolve_binning_column(raw_col: str) -> Optional[str]:
    col_lower = raw_col.lower().strip().replace(" ", "_")
    for canonical, meta in BINNING_MAP.items():
        if col_lower in meta["aliases"]:
            return canonical
    for canonical, meta in BINNING_MAP.items():
        if col_lower.endswith(f"_{canonical}") or col_lower.startswith(f"{canonical}_"):
            return canonical
    return None


def _derive_y_label(func: str, entity: str, field: str) -> str:
    entity_singular = entity.rstrip("s") if entity else "Employee"
    if func == "count":
        return f"Number of {entity_singular.title()}s"
    if func == "sum":
        return f"Total {field.replace('_', ' ').title()}"
    if func == "avg":
        return f"Average {field.replace('_', ' ').title()}"
    if func == "min":
        return f"Minimum {field.replace('_', ' ').title()}"
    if func == "max":
        return f"Maximum {field.replace('_', ' ').title()}"
    return field.replace("_", " ").title() if field else "Value"


