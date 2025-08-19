# Contributing

Thanks for your interest in contributing!

## Setup
- Python 3.12 recommended
- Create a virtual environment and install deps:
```
pip install -r requirements.txt
```
- Pre-run services:
```
docker compose -f docker/docker-compose.yml up -d
```

## Development
- Use feature branches: `feat/...`, `fix/...`, `docs/...`
- Keep changes focused; include tests/examples if applicable
- Run lints locally (if configured) and ensure the app runs:
```
streamlit run app/streamlit_app.py
```

## Commit style
- Use conventional messages when possible:
  - `feat: add English answer spinner`
  - `fix: robust dimension guard for Qdrant`
  - `docs: update README setup`

## Pull Requests
- Describe the change, rationale, and testing steps
- Link related issues if any
