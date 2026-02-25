# Schemas (V1)

Core schema families:

1. Evidence packet schema (agent-visible + hidden author notes)
2. Agent turn output schema (private/public + structured assessment)
3. Ground truth schema
4. Run manifest / logging schemas
5. Metrics schemas
6. Expert review CSV row schema

Pydantic models are defined in `src/cyberneg/core/schemas.py`.

JSON schema export:
- Implemented by `src/cyberneg/prompting/json_contracts.py`
- CLI commands can export schema JSON into run output folders.

