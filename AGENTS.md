# Project Coding Standards

## Solution & Patch Priority

When providing solutions or patches, follow this priority order:

1. **Industry practice first** � Prefer established patterns, widely-adopted libraries, and proven architectures over custom or novel approaches.
2. **Avoid hardcoding** � Use configuration, environment variables, registries, or dynamic discovery instead of literal strings, magic numbers, or fixed values.
3. **Be agentic where beneficial** � Favor self-healing, fallback chains, and LLM-driven reasoning over brittle static logic when it improves resilience without sacrificing predictability.

## Structure

- `agent/core/` � orchestration, planning, LLM clients
- `agent/integrations/` � external services (HANA MCP, SQL MCP)
- `agent/output/` � charting, Excel, binning exports
- `agent/dab/` � DAB-specific models and validation
- `agent/actions/` � unified action framework (step injection for entity operations)
  - `agent/actions/base_injector.py` � shared utilities (field parsing, validation, LLM fallback)
  - `agent/actions/strategies/` � per-entity strategy implementations
    - `simple_update_strategy.py` � self-update entities (employee_general)
    - `leave_action_strategy.py` � leave request creation (header + detail with $ref chaining)
    - `leave_entitlement_guard.py` � ensures entitlement queries hit the right entity
    - `leave_entitlement_updater.py` � entitlement balance adjustments
    - `leave_strategy.py` � dispatcher for leave strategies
- `agent/summarizer/` � response summarization
- `agent/auth/` � role-based access
- `config/` � environment templates and semantic configs

## Style

- Python: PEP 8
- JavaScript/TypeScript: Standard JS style
- Config files: Keep machine-readable and minimal
- Imports: stdlib ? third-party ? local, alphabetized within each group
- No circular imports; use `agent/core/` for shared interfaces

## Secrecy

- Never commit secrets (`.env`, keys, passwords, tokens).
- Use `.env.example` with placeholders.

## HANA MCP Integration Rules

- **HTTP-only transport**: `hana_client.py` must use `HANAMCP_HTTP_URL`; no STDIO fallback.
- **Wide result safety net**: `_format_result_for_llm()` must trim `hana_*` results with >12 columns to a 12-column Markdown preview; DAB results are excluded.
- **Tool discovery**: Validate tool names via dynamic `get_hana_tool_names()` from cached MCP schemas; never hardcode `HANA_TOOL_NAMES`.
- **Response normalization**: Extract from `structuredContent.columns/structuredContent.rows`; transform to `{"result": [dicts]}` for the summarizer.
- **Finance query discipline**: Do not inject `FINANCE` schema hallucinations; schema scope comes from the registry only.

## Error Handling

- Wrap external calls (HANA, DAB, LLM) with explicit fallback chains.
- Log failures to disk, not stdout (debug agent cannot access live terminal).
- Return structured error dicts; never raise unhandled exceptions to the user.

## Commits

- Format: `<type>(<scope>): <subject>`
- Types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`
- Scope examples: `hana`, `dab`, `summarizer`, `agent/core`

## Tool Usage

- Use `Grep`/`Glob` for file content search; avoid `Get-Content`/`Select-String`.
- Use `workdir` parameter instead of `cd` in shell commands.
- Quote paths with spaces in PowerShell: `& "path/to/script.py"`.

## Testing

- Unit tests alongside source: `agent/core/test_<module>.py`
- Mock external MCP calls; never hit live HANA/DAB in tests.
