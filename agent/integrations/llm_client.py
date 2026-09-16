import os
import json
import time
import logging
import requests
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(r"C:\Users\USER\Documents\hrms-api\config\.env")
if _env_path.exists():
    load_dotenv(dotenv_path=str(_env_path), override=True)
else:
    load_dotenv()  # fallback

logger = logging.getLogger("hr_agent")

# ═════════════════════════════════════════════════════════════════════════════
# PROVIDER CONFIG
# ═════════════════════════════════════════════════════════════════════════════
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# PATCH G1: Switch to Gemini OpenAI-compatible endpoint (industry standard)
# Native API is unreliable for tool calling per Google AI dev forum reports.
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Groq kept as optional fallback
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ═════════════════════════════════════════════════════════════════════════════
# MODEL CONFIG — Gemini 3.5 Flash (current stable, free tier)
# PATCH G2: Updated from retired gemini-3.1-flash-lite to gemini-3.5-flash
# ═════════════════════════════════════════════════════════════════════════════
MODEL_TIERS = {
    "planner": {
        "model": "gemini-3.5-flash",
        "temperature": 0.1,
        "max_tokens": 8192,
    },
    "executor": {
        "model": "gemini-3.5-flash",
        "temperature": 0.1,
        "max_tokens": 8192,
    },
    "responder": {
        "model": "gemini-3.5-flash",
        "temperature": 0.3,
        "max_tokens": 8192,
    },
}

MOCK_ENABLED = os.getenv("GROQ_MOCK", "false").lower() == "true"


# ═════════════════════════════════════════════════════════════════════════════
# MOCK RESPONSES
# ═════════════════════════════════════════════════════════════════════════════
def _mock_response(tier: str, prompt: str) -> Optional[Dict]:
    if tier == "planner":
        return {
            "message": {
                "content": (
                    '{"steps":[{"tool":"read_records","args":'
                    '{"entity":"Employee","select":"emp_id,name,department"}}],'
                    '"reasoning":"mock plan","chart":null,"rag":false,"client_side_binning":null}'
                ),
                "tool_calls": None,
            }
        }
    if tier == "executor":
        return {
            "message": {
                "content": "Mock chart configuration generated.",
                "tool_calls": None,
            }
        }
    return {
        "message": {
            "content": "Mock final response.",
            "tool_calls": None,
        }
    }


# ═════════════════════════════════════════════════════════════════════════════
# LIGHTWEIGHT BUDGET — Solo dev, 250K TPM is effectively unlimited.
# PATCH G3: Track actual token usage from API response headers when available.
# ═════════════════════════════════════════════════════════════════════════════
class TokenBudget:
    def __init__(self):
        self._day_counts: Dict[str, int] = {t: 0 for t in MODEL_TIERS}
        self._day_reset = time.time() + 86400
        self._last_usage: Dict[str, Dict[str, int]] = {}

    def _reset_day_if_needed(self):
        now = time.time()
        if now >= self._day_reset:
            for t in self._day_counts:
                self._day_counts[t] = 0
            self._day_reset = now + 86400

    def consume(self, tier: str, estimated_tokens: int = 0) -> bool:
        self._reset_day_if_needed()
        total_today = sum(self._day_counts.values())
        if total_today >= 400:  # 80% of 500 RPD
            logger.warning("Gemini free tier: %d/500 RPD used today.", total_today)
        self._day_counts[tier] += 1
        return True

    def record_usage(self, tier: str, prompt_tokens: int, completion_tokens: int):
        """PATCH G3: Record actual token usage from API response for accurate tracking."""
        self._last_usage[tier] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def get_remaining(self, tier: str) -> Dict[str, Any]:
        self._reset_day_if_needed()
        total_today = sum(self._day_counts.values())
        last = self._last_usage.get(tier, {})
        return {
            "rpd_remaining": 500 - total_today,
            "tpm_limit": 250000,
            "rpm_limit": 15,
            "last_call_tokens": last.get("total_tokens", 0),
        }


_budget = TokenBudget()


# ═════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL API CALL — Unified OpenAI-compatible path for Gemini + Groq
# PATCH G4: Removed native Gemini API branch. All providers use OpenAI-compatible
# format via their respective endpoints. Eliminates conversion layer entirely.
# ═════════════════════════════════════════════════════════════════════════════
def _call_llm_single(
    model: str,
    system_prompt: str,
    user_prompt: str,
    tools: Optional[List[Dict]] = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
    json_mode: bool = False,
) -> Tuple[Optional[Dict], Any, Optional[requests.Response], Optional[Dict]]:
    """Execute one API call. Returns (choice, error, raw_response, usage_info).

    PATCH G4: Unified OpenAI-compatible request format for all providers.
    Gemini uses /v1beta/openai endpoint; Groq uses standard endpoint.
    """

    is_gemini = model.startswith("gemini")

    if is_gemini:
        api_key = GEMINI_API_KEY
        base_url = GEMINI_URL
    else:
        api_key = GROQ_API_KEY
        base_url = GROQ_URL

    if not api_key:
        return None, {
            "code": "missing_api_key",
            "message": f"No API key for provider serving model {model}",
            "retryable": False,
        }, None, None

    try:
        url = f"{base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Content-Type": "application/json",
        }
        if is_gemini:
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30,
        )

        # ── HTTP error handling ──
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            wait = int(retry_after) if retry_after else 5
            logger.warning("429 rate limit on %s. retry-after=%ss", model, wait)
            return None, {
                "code": "rate_limit_exceeded",
                "retry_after": wait,
                "retryable": True,
            }, resp, None

        if resp.status_code >= 400:
            try:
                err_body = resp.json()
            except Exception:
                err_body = {"message": resp.text[:500]}
            err_code = err_body.get("error", {}).get("code", "unknown_error")
            err_msg = err_body.get("error", {}).get("message", str(err_body))
            logger.warning("HTTP %s on %s: %s", resp.status_code, model, err_msg)
            return None, {
                "code": err_code,
                "message": err_msg,
                "retryable": resp.status_code >= 500,
            }, resp, None

        data = resp.json()

        # PATCH G3: Extract actual token usage for accurate budget tracking
        usage_info = data.get("usage", {})

        if "choices" not in data or not data.get("choices"):
            error_info = data.get("error", {})
            logger.warning("API error on %s: %s", model, error_info)
            return None, error_info, resp, usage_info

        return data["choices"][0], None, resp, usage_info

    except requests.exceptions.JSONDecodeError as e:
        logger.warning("Non-JSON response on %s: %s", model, e)
        return None, {
            "code": "json_decode_error",
            "message": str(e),
            "retryable": True,
        }, None, None
    except requests.exceptions.Timeout:
        logger.warning("Timeout on %s", model)
        return None, {
            "code": "timeout",
            "message": "Request timed out",
            "retryable": True,
        }, None, None
    except Exception as e:
        logger.warning("Request exception on %s: %s", model, e)
        return None, {
            "code": "request_exception",
            "message": str(e),
            "retryable": True,
        }, None, None


# ═════════════════════════════════════════════════════════════════════════════
# TIER-AWARE CALLER — Same signature as before. Zero changes in other files.
# PATCH G5: Provider-agnostic JSON mode fallback + model-not-found retry.
# ═════════════════════════════════════════════════════════════════════════════
def call_llm(
    system_prompt: str,
    user_prompt: str,
    tools: Optional[List[Dict]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    json_mode: bool = False,
    tier: str = "planner",
    estimated_tokens: int = 6000,
) -> Optional[Dict]:
    """Call LLM (Gemini/Groq via OpenAI-compatible endpoints) with tier-aware model selection."""

    if tier not in MODEL_TIERS:
        raise ValueError(f"Unknown tier '{tier}'. Use: planner, executor, responder")

    if MOCK_ENABLED:
        logger.info("[MOCK] Returning canned response for tier '%s'", tier)
        return _mock_response(tier, user_prompt)

    config = MODEL_TIERS[tier]

    if not _budget.consume(tier, estimated_tokens):
        logger.error("Tier '%s' daily budget exhausted.", tier)
        return None

    m = model if model else config["model"]
    temp = temperature if temperature is not None else config["temperature"]
    max_tok = max_tokens if max_tokens is not None else config["max_tokens"]

    choice, error, resp, usage = _call_llm_single(
        m, system_prompt, user_prompt, tools, temp, max_tok, json_mode
    )

    # PATCH G5a: Model-not-found retry — fallback to gemini-3.5-flash
    if isinstance(error, dict) and error.get("code") in ("model_not_found", "invalid_model", "not_found"):
        fallback_model = "gemini-3.5-flash"
        if m != fallback_model:
            logger.warning("Model '%s' not found — retrying with '%s'", m, fallback_model)
            choice, error, resp, usage = _call_llm_single(
                fallback_model, system_prompt, user_prompt, tools, temp, max_tok, json_mode
            )

    if choice is not None:
        if usage:
            _budget.record_usage(
                tier,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0)
            )
        return choice

    # Retryable errors (rate limit, timeout, 5xx)
    if isinstance(error, dict) and error.get("retryable"):
        retry_after = error.get("retry_after", 0)
        if retry_after:
            logger.warning("Retryable error on %s — sleeping %ss...", m, retry_after)
            time.sleep(retry_after)
        choice, error, resp, usage = _call_llm_single(
            m, system_prompt, user_prompt, tools, temp, max_tok, json_mode
        )
        if choice is not None:
            if usage:
                _budget.record_usage(
                    tier,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0)
                )
            return choice

    # PATCH G5b: Provider-agnostic JSON mode fallback.
    # Detects ANY JSON-related error, not just Groq-specific strings.
    if json_mode and isinstance(error, dict):
        err_msg = str(error.get("message", "")).lower()
        json_error_keywords = [
            "json", "response_format", "response_mime_type", "schema",
            "json_object", "invalid_request_error", "unsupported"
        ]
        if any(kw in err_msg for kw in json_error_keywords):
            logger.warning("JSON mode error on %s — retrying without... Error: %s", m, err_msg[:100])
            choice, error, resp, usage = _call_llm_single(
                m, system_prompt, user_prompt, tools, temp, max_tok, json_mode=False
            )
            if choice is not None:
                if usage:
                    _budget.record_usage(
                        tier,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0)
                    )
                return choice

    logger.error("LLM call failed for tier '%s'. Last error: %s", tier, error)
    return None


def get_budget_status() -> Dict[str, Dict[str, Any]]:
    """Return remaining budget for all tiers."""
    return {t: _budget.get_remaining(t) for t in MODEL_TIERS}


# ═════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY — Keep old name for existing imports
# ═════════════════════════════════════════════════════════════════════════════
call_groq = call_llm