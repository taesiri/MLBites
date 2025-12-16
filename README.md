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
- `metadata.json`: `{"title": "...", "category": "...", "framework": "pytorch|numpy", "tags": [...], "difficulty": "Easy|Medium|Hard", "relevant_questions": [...]}`
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

## Security Notes (Running User Code)

This app lets users submit Python code which is executed (to run tests). **Treat all user code as untrusted.**

- **Not a real sandbox**: MLBites runs candidate code in a separate Python process and applies a strict “policy gate”
  (rejects dangerous imports/constructs), but **Python-level restrictions are not a complete security boundary**.
- **Do not run with secrets**: run the server with a minimal environment (no API keys), and assume user code can print anything it can read.
- **Use OS isolation in production**: run the verifier in a locked-down container/VM with:
  - **no network egress**
  - **read-only filesystem** (only mount `db/` read-only; use a temp write dir)
  - **resource limits** (CPU, memory, pids)
  - a **non-root user**, seccomp/AppArmor (Linux), or stronger isolation (gVisor/Firecracker)
- **Rate limit `/api/run`**: apply per-IP throttling and request body size limits at a reverse proxy (nginx/Caddy/Cloudflare).
