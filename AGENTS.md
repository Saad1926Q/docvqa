# DocVQA

Stack: **uv** (package mgmt), **hatchling** (build), **ruff** (lint + format), **ty** (type checker), **pytest** (tests), **VLMEvalKit** (eval).

Fine-tune a compact VLM on DocVQA. Run ablations (rank, LR, token budget, prompt templates, train size). Evaluate zero-shot vs fine-tuned. Manual failure taxonomy on 50+ examples.

## Commands

```bash
# Install / sync
uv sync

# Lint
uv run ruff check src/
uv run ruff check src/ --fix          # auto-fix

# Format
uv run ruff format src/
uv run ruff format src/ --check       # check-only (CI)

# Type check
uv run ty check src/

# All checks before commit
uv run ruff check src/ --fix && uv run ruff format src/ && uv run ty check src/

# Tests
uv run pytest tests/

```
