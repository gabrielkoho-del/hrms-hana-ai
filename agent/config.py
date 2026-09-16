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

# ==============================
# FORECASTING ENGINE
# ==============================
SANDBOX_MEMORY_MB = int(os.getenv("SANDBOX_MEMORY_MB", "512"))
SANDBOX_TIMEOUT_SECONDS = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "30"))
SANDBOX_MAX_FILES = int(os.getenv("SANDBOX_MAX_FILES", "64"))
FORECAST_TEMP_DIR = os.getenv("FORECAST_TEMP_DIR", None)
FORECAST_DATA_DIR = os.getenv("FORECAST_DATA_DIR", None)
FORECAST_HORIZON_MONTHS = int(os.getenv("FORECAST_HORIZON_MONTHS", "6"))
FORECAST_EXTERNAL_MONTHS = int(os.getenv("FORECAST_EXTERNAL_MONTHS", "24"))
FORECAST_MODEL_DEFAULT = os.getenv("FORECAST_MODEL_DEFAULT", "statsforecast")
DOSM_BASE_URL = os.getenv("DOSM_BASE_URL", "https://api.data.gov.my/opendosm")
WORLD_BANK_BASE_URL = os.getenv("WORLD_BANK_BASE_URL", "https://api.worldbank.org/v2")

# ==============================
# SCHEMA REGISTRY
# ==============================
SCHEMA_REGISTRY_TTL_SECONDS = int(os.getenv("SCHEMA_REGISTRY_TTL_SECONDS", "3600"))
SCHEMA_REGISTRY_REFRESH_INTERVAL_SECONDS = int(os.getenv("SCHEMA_REGISTRY_REFRESH_INTERVAL_SECONDS", "3600"))

# ==============================
# RAG / CHROMADB
# ==============================
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "hr_policies")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.20"))

# ==============================
# AGENTIC SYSTEM USER
# ==============================
# Numeric user ID written to create_by/modify_by columns (Int64 in the HRMS
# schema) when the agentic AI creates records on behalf of an employee.
# Value 1 identifies the agentic AI system as the record creator, instead of
# the employee_no string (e.g. "A0001") which DAB rejects for Int64 columns.
AGENT_SYSTEM_USER_ID = int(os.getenv("AGENT_SYSTEM_USER_ID", "1"))