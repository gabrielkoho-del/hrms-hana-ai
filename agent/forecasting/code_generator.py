"""
agent/forecasting/code_generator.py
LLM-generated Python code for forecasting models.
Uses Gemini with a strict system prompt to generate nixtla scripts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import py_compile
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agent.integrations.llm_client import call_llm

logger = logging.getLogger("hr_agent.forecasting")

# Prompts live at the project root: <project>/prompts/forecast_codegen.md
_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "prompts",
    "forecast_codegen.md",
)

# Cache: key = (data_schema_hash, model_type), value = code string
_code_cache: Dict[str, str] = {}
_CACHE_MAX = 128


@dataclass
class GenerationResult:
    success: bool
    code: str = ""
    model_type: str = "statsforecast"
    attempts: int = 0
    error: Optional[str] = None
    cached: bool = False


class CodeGenerator:
    """Generates forecasting Python code via LLM.

    Args:
        prompt_path: Path to the system prompt markdown file.
    """

    def __init__(self, prompt_path: Optional[str] = None):
        self.prompt_path = prompt_path or _PROMPT_PATH
        self._system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        if not os.path.isfile(self.prompt_path):
            logger.warning("Codegen prompt not found at %s", self.prompt_path)
            return self._default_prompt()
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _default_prompt(self) -> str:
        return (
            "You are a Python forecasting code generator. "
            "The variable `input_data` is pre-loaded by the sandbox with "
            "`merged_series` and `metadata` keys. Build a time-series forecast "
            "using statsforecast/mlforecast and set `_forecast_result` at the end "
            "with the schema: forecasts[], model_info, metrics, data_sources_used. "
            "Blocked modules: os, subprocess, socket, urllib, http, requests, shutil. "
            "Allowed: pandas, numpy, statsforecast, mlforecast, sklearn, "
            "json, math, datetime, statistics, collections."
        )

    def _cache_key(self, data_schema: Dict[str, Any], model_type: str) -> str:
        schema_hash = hashlib.sha256(
            json.dumps(data_schema, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        return f"{schema_hash}:{model_type}"

    def _check_cache(self, cache_key: str) -> Optional[str]:
        return _code_cache.get(cache_key)

    def _set_cache(self, cache_key: str, code: str) -> None:
        if len(_code_cache) >= _CACHE_MAX:
            # Evict oldest
            _code_cache.pop(next(iter(_code_cache)))
        _code_cache[cache_key] = code

    def generate(
        self,
        question: str,
        enriched_data: Dict[str, Any],
        model_type: str = "statsforecast",
        max_attempts: int = 3,
    ) -> GenerationResult:
        """Generate forecasting code for the given question and data.

        Args:
            question: User's natural-language forecasting question.
            enriched_data: The merged dataset from ExternalDataFetcher.
            model_type: "statsforecast".
            max_attempts: Max retry attempts on syntax/execution failure.

        Returns:
            GenerationResult with generated code or error.
        """
        cache_key = self._cache_key(enriched_data.get("metadata", {}), model_type)
        cached_code = self._check_cache(cache_key)
        if cached_code:
            logger.info("CodeGenerator cache hit for model=%s", model_type)
            return GenerationResult(success=True, code=cached_code, model_type=model_type, cached=True)

        # Build context from enriched data (limit size for prompt)
        merged_preview = json.dumps(
            enriched_data.get("merged_series", [])[:12], default=str, indent=2
        )
        metadata = enriched_data.get("metadata", {})

        user_prompt = (
            f"Forecasting question: {question}\n\n"
            f"Target model: {model_type}\n\n"
            f"Data sources: {', '.join(metadata.get('data_sources', []))}\n\n"
            f"Merged data preview (first 12 rows):\n{merged_preview}\n\n"
            f"Columns available in merged_series: "
            f"{list((enriched_data.get('merged_series') or [{}])[0].keys()) if enriched_data.get('merged_series') else 'N/A'}\n\n"
            "Generate the complete Python script now."
        )

        last_error = None
        for attempt in range(1, max_attempts + 1):
            logger.info("CodeGenerator attempt %d/%d for model=%s", attempt, max_attempts, model_type)
            try:
                choice = call_llm(
                    self._system_prompt,
                    user_prompt + (f"\n\nPrevious error (fix it): {last_error}" if last_error else ""),
                    temperature=0.1,
                    max_tokens=8192,
                )
                code = choice.get("message", {}).get("content", "").strip()
                if code.startswith("```python"):
                    code = code[len("```python"):]
                if code.startswith("```"):
                    code = code[3:]
                if code.endswith("```"):
                    code = code[:-3]
                code = code.strip()

                # Syntax validation
                try:
                    py_compile.compile(code, "<sandbox>", "exec")
                except py_compile.PyCompileError as exc:
                    last_error = f"Syntax error: {exc}"
                    logger.warning("CodeGenerator syntax error: %s", exc)
                    continue

                # Cache and return
                self._set_cache(cache_key, code)
                return GenerationResult(
                    success=True,
                    code=code,
                    model_type=model_type,
                    attempts=attempt,
                    cached=False,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning("CodeGenerator LLM call failed: %s", exc)

        return GenerationResult(
            success=False,
            model_type=model_type,
            attempts=max_attempts,
            error=f"Failed after {max_attempts} attempts. Last error: {last_error}",
        )

    def fix(self, code: str, error: str, enriched_data: Dict[str, Any], model_type: str = "statsforecast") -> GenerationResult:
        """Retry code generation with execution error feedback."""
        cache_key = self._cache_key(enriched_data.get("metadata", {}), model_type)
        # Invalidate cache since we're fixing
        _code_cache.pop(cache_key, None)

        fix_prompt = (
            "The previous generated script failed with this error:\n"
            f"```\n{error}\n```\n\n"
            "Fix the script. Requirements remain the same: "
            "use the pre-loaded `input_data`, build forecast, set `_forecast_result`. "
            "Return ONLY the corrected Python code."
        )
        try:
            choice = call_llm(
                self._system_prompt,
                fix_prompt,
                temperature=0.1,
                max_tokens=8192,
            )
            code = choice.get("message", {}).get("content", "").strip()
            if code.startswith("```python"):
                code = code[len("```python"):]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()

            try:
                py_compile.compile(code, "<sandbox>", "exec")
            except py_compile.PyCompileError as exc:
                return GenerationResult(success=False, code=code, model_type=model_type, error=f"Fixed code still has syntax error: {exc}")

            self._set_cache(cache_key, code)
            return GenerationResult(success=True, code=code, model_type=model_type, cached=False)
        except Exception as exc:
            return GenerationResult(success=False, model_type=model_type, error=f"Fix retry failed: {exc}")
