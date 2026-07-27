"""
agent/code_resolver.py
Zero-touch code resolution for multi-tenant DAB production.
Fetches codesetup from each tenant's DAB and builds reverse index for LLM context injection.
"""
import json
import logging
import time
import re
from typing import Dict, List, Optional, Tuple
from threading import Lock

logger = logging.getLogger("hr_agent")

# ─────────────────────────────────────────────────────────
# TTL CACHE ENTRY
# ─────────────────────────────────────────────────────────
class _TTLCacheEntry:
    def __init__(self, value: any, ttl_seconds: int = 300):
        self.value = value
        self.expires_at = time.time() + ttl_seconds

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

# ─────────────────────────────────────────────────────────
# HEURISTIC FILTERS
# Prevent false matches on IDs, dates, emails, salaries
# These rules ensure we don't accidentally treat numeric IDs or dates as lookup codes
# ─────────────────────────────────────────────────────────
def _is_likely_id_or_numeric(value: str) -> bool:
    """Heuristic: skip values that look like IDs, dates, or currency.
    
    Codes in codesetup are meant to be human-readable short strings like "A", "SL", "ENG".
    We must filter out values that would incorrectly match but aren't actual codes:
    - Employee IDs (10+ digits) → skip
    - Dates → skip  
    - Emails → skip
    - Currency/numbers → skip
    - UUIDs → skip
    """
    if not value or not isinstance(value, str):
        return True

    s = value.strip()
    if not s:
        return True

    # Long numeric strings → IDs
    if s.isdigit() and len(s) >= 10:
        return True

    # Negative numbers → IDs or codes, not lookup values
    if re.match(r'^-?\d+$', s):
        return len(s) <= 6  # Short numbers may be codes

    # Dates and datetimes
    date_patterns = [
        r'^\d{4}-\d{2}-\d{2}$',          # YYYY-MM-DD
        r'^\d{2}/\d{2}/\d{4}$',          # MM/DD/YYYY
        r'^\d{2}-\d{2}-\d{4}$',          # DD-MM-YYYY
        r'^\d{4}-\d{2}-\d{2}T',         # ISO datetime
        r'^\d{4}-\d{2}',                 # Starts like 2024-12
    ]
    for pat in date_patterns:
        if re.match(pat, s):
            return True

    # Email addresses
    if "@" in s and re.match(r'^[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}$', s):
        return True

    # Currency/numbers with decimals
    try:
        float(s.replace('$', '').replace(',', '').replace(' ', ''))
        return True  # Numeric values not codes
    except ValueError:
        pass

    # UUIDs
    if re.match(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$', s):
        return True

    return False

# ─────────────────────────────────────────────────────────
# CODE RESOLVER
# ─────────────────────────────────────────────────────────
class CodeResolver:
    """Per-tenant singleton that caches codesetup and provides reverse index lookups.
    
    Multi-tenant architecture:
    - Each tenant has separate DAB instance on separate port
    - Each tenant may have different codesetup values
    - Instance keyed by tenant_id ensures isolation
    
    Pagination strategy (DAB uses cursor-based, NOT offset):
    - Uses $first (page size) + $after (cursor from @odata.nextLink)
    - Handles 20K+ records across multiple pages
    - Cursor extracted from @odata.nextLink response header
    
    TTL cache: 300 seconds, auto-refresh on expiry
    """
    
    _instances: Dict[str, "CodeResolver"] = {}
    _lock = Lock()

    def __new__(cls, tenant_id: str) -> "CodeResolver":
        with cls._lock:
            if tenant_id not in cls._instances:
                instance = super().__new__(cls)
                instance.tenant_id = tenant_id
                instance._cache_entry: Optional[_TTLCacheEntry] = None
                instance._reverse_index: Dict[str, List[Tuple[str, str]]] = {}
                instance._initialized = False
                cls._instances[tenant_id] = instance
            return cls._instances[tenant_id]

    async def get_reverse_index(self) -> Dict[str, List[Tuple[str, str]]]:
        """Get reverse index for this tenant, refreshing if expired."""
        if self._cache_entry and not self._cache_entry.is_expired():
            return self._reverse_index

        # Expired or not initialized → refresh
        await self._refresh_cache()
        return self._reverse_index

    async def _refresh_cache(self):
        """Fetch codesetup via DAB cursor pagination and build reverse index."""
        from agent.dab_client import dab_manager
        import asyncio

        try:
            client = await dab_manager.get_client_async(self.tenant_id)
            
            all_codes = []
            after_cursor = None
            page_size = 1000

            while True:
                args = {
                    "entity": "codesetup",
                    "select": "type,code,description",
                    "first": str(page_size)
                }
                if after_cursor:
                    args["after"] = after_cursor

                result = await asyncio.to_thread(client.call_tool, "read_records", args)
                
                # Extract items from MCP response
                items, count, has_more, next_cursor = self._extract_items(result)
                all_codes.extend(items)

                if not has_more or not next_cursor:
                    break
                after_cursor = next_cursor

            # Build reverse index
            self._reverse_index = {}
            for item in all_codes:
                if not isinstance(item, dict):
                    continue
                code = str(item.get("code", "")).strip()
                if not code or _is_likely_id_or_numeric(code):
                    continue
                code_type = str(item.get("type", "unknown")).strip()
                description = str(item.get("description", "")).strip()
                
                if code not in self._reverse_index:
                    self._reverse_index[code] = []
                self._reverse_index[code].append((code_type, description))

            self._cache_entry = _TTLCacheEntry(self._reverse_index, ttl_seconds=300)
            self._initialized = True
            
            logger.info(
                "CodeResolver: refreshed %s cache with %d unique codes",
                self.tenant_id, len(self._reverse_index)
            )
            
        except Exception as e:
            logger.warning(
                "CodeResolver: failed to refresh cache for %s: %s. Using stale data if available.",
                self.tenant_id, e
            )
            # Keep stale data if available
            if not self._reverse_index:
                self._reverse_index = {}

    def _extract_items(self, result: dict) -> Tuple[List[Dict], int, bool, Optional[str]]:
        """Extract items list, count, has_more, and next cursor from DAB response."""
        items = []
        has_more = False
        next_cursor = None

        if isinstance(result, dict):
            # MCP wrapper
            content = result.get("content", [])
            if content and isinstance(content, list):
                first = content[0] if len(content) > 0 else {}
                if isinstance(first, dict) and "text" in first:
                    try:
                        inner = json.loads(first["text"])
                        items = inner.get("value", inner.get("items", inner.get("result", [])))
                        # Check for nextLink to determine if more pages exist (cursor-based pagination)
                        next_link = inner.get("@odata.nextLink")
                        has_more = next_link is not None and len(items) >= page_size
                        if next_link:
                            match = re.search(r'\$after=([^&]+)', next_link)
                            next_cursor = match.group(1) if match else None
                    except (json.JSONDecodeError, TypeError):
                        pass
            else:
                # Direct DAB response
                items = result.get("value", result.get("items", result.get("result", [])))
                has_more = result.get("@odata.nextLink") is not None
                if has_more:
                    next_link = result.get("@odata.nextLink", "")
                    match = re.search(r'\$after=([^&]+)', next_link)
                    next_cursor = match.group(1) if match else None

        return items, len(items), has_more, next_cursor

    def resolve_code(self, code_value: str) -> List[Tuple[str, str]]:
        """Lookup code value in reverse index. Returns list of (type, description) tuples."""
        return self._reverse_index.get(str(code_value).strip(), [])

# ─────────────────────────────────────────────────────────
# SCANNER
# Detects codes in tool results and builds LLM context block
# Ambiguity handling: LLM receives all matches and picks based on context
# ─────────────────────────────────────────────────────────
async def scan_for_codes(
    tool_results: Dict[str, dict],
    tenant_id: str,
    limit_codes: int = 20
) -> str:
    """Scan all tool results for code values and build LLM context block.
    
    Zero-touch design: only codes actually present in results are injected.
    This keeps prompt size minimal while still enabling accurate translations.
    
    Ambiguity handling (rare in practice):
    - If code "A" maps to both STATUS and GENDER types, both are shown
    - LLM uses field context to disambiguate (e.g., STATUS_CODE vs GENDER)
    """
    resolver = CodeResolver(tenant_id)
    reverse_index = await resolver.get_reverse_index()

    if not reverse_index:
        return ""

    # Collect all code values found in results
    found_codes: Dict[str, List[Tuple[str, str]]] = {}
    
    for tool_name, tool_output in tool_results.items():
        if not isinstance(tool_output, dict):
            continue
        
        # Extract data items
        raw_data = tool_output.get("result", tool_output.get("value", tool_output.get("items", [])))
        if not isinstance(raw_data, list):
            continue

        for item in raw_data:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if not isinstance(value, (str, int, float)):
                    continue
                
                str_val = str(value).strip()
                
                # Skip heuristics
                if _is_likely_id_or_numeric(str_val):
                    continue
                
                # Check if this value is a known code
                if str_val in reverse_index:
                    found_codes[str_val] = reverse_index[str_val]

    if not found_codes:
        return ""

    # Build markdown context block
    lines = ["## Code Field Mappings"]
    lines.append("")
    
    # Group by type
    codes_by_type: Dict[str, List[str]] = {}
    for code, descriptions in found_codes.items():
        for code_type, desc in descriptions:
            if code_type not in codes_by_type:
                codes_by_type[code_type] = []
            if len(codes_by_type[code_type]) < limit_codes:
                codes_by_type[code_type].append(f"- \"{code}\" → \"{desc}\"")

    for code_type, mappings in codes_by_type.items():
        lines.append(f"### {code_type}")
        lines.extend(mappings)
        lines.append("")

    lines.append("**Instructions:** Replace raw code values with their descriptions in the response.")
    lines.append("")

    return "\n".join(lines)