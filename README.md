# MLBites 🔥

PyTorch Interview Questions Practice Platform

## Quick Start

```bash
# Install dependencies
uv sync

# Run the development server
uv run uvicorn src.mlbites.main:app --reload
```

Open http://localhost:8000 in your browser.

## Project Structure

```
MLBites/
├── src/mlbites/          # FastAPI application
│   ├── main.py           # API endpoints
│   └── models.py         # Pydantic models
├── db/                   # Question database
│   └── {question_slug}/
│       ├── metadata.json # Title, tags, category
│       ├── question.md   # Question description
│       ├── starting_point.py
│       ├── tests.py      # Deterministic verification tests (required)
│       └── solution.py
├── static/               # Frontend assets
└── templates/            # Jinja2 templates
```

## Adding Questions

Create a new folder in `db/` with:
- `metadata.json`: `{"title": "...", "category": "...", "tags": [...], "difficulty": "Easy|Medium|Hard"}`
- `question.md`: Markdown description
- `starting_point.py`: Skeleton code
- `tests.py`: Deterministic verification tests (`run_tests(candidate_module)`)
- `solution.py`: Complete solution

## Verifying Solutions (CLI)

Run a question’s tests against its reference solution:

```bash
uv run python -m mlbites.verify <question_slug>
```

Or verify an arbitrary candidate file:

```bash
uv run python -m mlbites.verify <question_slug> --candidate path/to/candidate.py
```
