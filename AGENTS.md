# SlideVQA

Stack: **uv** (package mgmt), **hatchling** (build), **ruff** (lint + format), **ty** (type checker), **pytest** (tests), **Unsloth** (SFT), **PEFT/LoRA**.

Fine-tune a compact VLM on SlideVQA. Run ablations (rank, LR, token budget, prompt templates, train size). Evaluate zero-shot vs fine-tuned. Manual failure taxonomy on 50+ examples.

## Commands

```bash
# Install / sync
uv sync

# Lint
uv run ruff check src/ scripts/eval_slidevqa.py scripts/train_slidevqa.py
uv run ruff check src/ scripts/eval_slidevqa.py scripts/train_slidevqa.py --fix          # auto-fix

# Format
uv run ruff format src/ scripts/eval_slidevqa.py scripts/train_slidevqa.py
uv run ruff format src/ scripts/eval_slidevqa.py scripts/train_slidevqa.py --check       # check-only (CI)

# Type check
uv run ty check src/ scripts/eval_slidevqa.py scripts/train_slidevqa.py

# All checks before commit
uv run ruff check src/ scripts/eval_slidevqa.py scripts/train_slidevqa.py --fix && uv run ruff format src/ scripts/eval_slidevqa.py scripts/train_slidevqa.py && uv run ty check src/ scripts/eval_slidevqa.py scripts/train_slidevqa.py

# Tests
uv run pytest tests/

```
