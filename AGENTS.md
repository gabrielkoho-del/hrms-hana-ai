# Project Coding Standards

## Solution & Patch Priority

When providing solutions or patches, follow this priority order:

1. **Industry practice first** — Prefer established patterns, widely-adopted libraries, and proven architectures over custom or novel approaches.
2. **Avoid hardcoding** — Use configuration, environment variables, registries, or dynamic discovery instead of literal strings, magic numbers, or fixed values.
3. **Be agentic where beneficial** — Favor self-healing, fallback chains, and LLM-driven reasoning over brittle static logic when it improves resilience without sacrificing predictability.

## Style

- Python: PEP 8
- JavaScript/TypeScript: Standard JS style
- Config files: Keep machine-readable and minimal

## Secrecy

- Never commit secrets (`.env`, keys, passwords, tokens).
- Use `.env.example` with placeholders.
