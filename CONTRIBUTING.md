# Contributing to AgentTrust

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/assister-xyz/quality-oracle.git
cd quality-oracle
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
# Add at least one LLM key (GROQ_API_KEY or CEREBRAS_API_KEY)
```

## Running Tests

All tests are mocked — no MongoDB, Redis, or LLM keys needed:

```bash
python -m pytest tests/ -q
```

## Linting

```bash
ruff check src/
```

## Making Changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run tests and lint (`pytest tests/ -q && ruff check src/`)
4. Submit a PR with a clear description of the change

## Code Style

- Python 3.12+, type hints preferred
- Follow existing patterns in `src/`
- Ruff for linting (config in `pyproject.toml` / `ruff.toml` defaults)
- Tests go in `tests/` with `test_` prefix

## What to Contribute

- New question bank entries (see `data/questions/`)
- Additional LLM provider integrations
- Framework adapters (LangChain, CrewAI, etc.)
- Bug fixes and documentation improvements

## Reporting Issues

Open a GitHub issue with:
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
