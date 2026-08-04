# agent/config.py
"""Centralized configuration constants for the HR AI Agent."""
import os

# ==============================
# QUERY / EXPORT LIMITS
# ==============================
DEFAULT_MAX_ROWS = int(os.getenv("DEFAULT_MAX_ROWS", "100"))
MAX_SQL_LENGTH = int(os.getenv("MAX_SQL_LENGTH", "5000"))
LARGE_RESULT_THRESHOLD = int(os.getenv("LARGE_RESULT_THRESHOLD", "20"))
UNLIMITED_ROWS = int(os.getenv("UNLIMITED_ROWS", "9999"))

# ==============================
# API / RATE LIMITING
# ==============================
AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", "http://localhost:8000")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", "8000"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))

# ==============================
# AUTH
# ==============================
JWT_SECRET = os.getenv("JWT_SECRET", "local-dev-secret-change-me")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "hrms-api")
JWT_ISSUER_ALLOWLIST = os.getenv("JWT_ISSUER_ALLOWLIST", "")
JWT_ISSUERS = [i.strip() for i in JWT_ISSUER_ALLOWLIST.split(",") if i.strip()] or ["local-dev-issuer"]
DEV_TEST_TOKEN = os.getenv("DEV_TEST_TOKEN", "")
AUTH_MODE = os.getenv("AUTH_MODE", "test")  # "test", "production", "disabled"

# ==============================
# PLANNER / LLM
# ==============================
PLANNER_TIER = os.getenv("PLANNER_TIER", "gemini-2.5-flash")
PLANNER_ESTIMATED_TOKENS = int(os.getenv("PLANNER_ESTIMATED_TOKENS", "5500"))
CONVERSATION_HISTORY_TOKEN_BUDGET = int(os.getenv("CONVERSATION_HISTORY_TOKEN_BUDGET", "2000"))