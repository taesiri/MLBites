"""FastAPI application for MLBites - PyTorch Interview Questions."""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import markdown
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .models import (
    FormatPythonRequest,
    FormatPythonResponse,
    QuestionDetail,
    QuestionListItem,
    QuestionSolution,
    RunCodeRequest,
    RunCodeResponse,
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_DIR = BASE_DIR / "db"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

logger = logging.getLogger("uvicorn.error")

# FastAPI app
app = FastAPI(
    title="MLBites",
    description="PyTorch Interview Questions Practice Platform",
    version="0.1.0",
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Markdown converter
md = markdown.Markdown(extensions=["fenced_code", "codehilite", "tables"])


def load_question_metadata(question_dir: Path) -> dict | None:
    """Load metadata.json from a question directory."""
    metadata_file = question_dir / "metadata.json"
    if not metadata_file.exists():
        return None

    with open(metadata_file, "r") as f:
        data = json.load(f)
        data["slug"] = question_dir.name
        return data


def get_all_questions() -> list[QuestionListItem]:
    """Get all questions from the database."""
    questions = []

    if not DB_DIR.exists():
        return questions

    for question_dir in sorted(DB_DIR.iterdir()):
        if question_dir.is_dir():
            metadata = load_question_metadata(question_dir)
            if metadata:
                questions.append(QuestionListItem(**metadata))

    return questions


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page."""
    questions = get_all_questions()

    # Cache-bust static assets when they change on disk (avoids stale JS/CSS in browser cache).
    # We intentionally keep this cheap: just stat two files.
    try:
        static_js_version = int((STATIC_DIR / "js" / "app.js").stat().st_mtime)
    except Exception:  # noqa: BLE001
        static_js_version = int(time.time())
    try:
        static_css_version = int((STATIC_DIR / "css" / "style.css").stat().st_mtime)
    except Exception:  # noqa: BLE001
        static_css_version = int(time.time())

    # Difficulty order for sorting
    difficulty_order = {"Easy": 0, "Medium": 1, "Hard": 2}

    # Group questions by category
    categories: dict[str, list[QuestionListItem]] = {}
    for q in questions:
        if q.category not in categories:
            categories[q.category] = []
        categories[q.category].append(q)

    # Sort questions within each category by difficulty
    for category in categories:
        categories[category].sort(key=lambda q: difficulty_order.get(q.difficulty, 1))

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "categories": categories,
            "all_questions": questions,
            "static_js_version": static_js_version,
            "static_css_version": static_css_version,
        },
    )


@app.get("/api/questions", response_model=list[QuestionListItem])
async def list_questions():
    """Get all questions."""
    return get_all_questions()


@app.get("/api/questions/{slug}", response_model=QuestionDetail)
async def get_question(slug: str):
    """Get a specific question by slug."""
    question_dir = DB_DIR / slug

    if not question_dir.exists():
        raise HTTPException(status_code=404, detail="Question not found")

    metadata = load_question_metadata(question_dir)
    if not metadata:
        raise HTTPException(status_code=404, detail="Question metadata not found")

    # Load question description
    question_md_file = question_dir / "question.md"
    if not question_md_file.exists():
        raise HTTPException(status_code=404, detail="Question description not found")

    with open(question_md_file, "r") as f:
        question_md = f.read()

    md.reset()
    description_html = md.convert(question_md)

    # Load starting code
    starting_code_file = question_dir / "starting_point.py"
    starting_code = ""
    if starting_code_file.exists():
        with open(starting_code_file, "r") as f:
            starting_code = f.read()

    return QuestionDetail(
        slug=metadata["slug"],
        title=metadata["title"],
        category=metadata["category"],
        tags=metadata.get("tags", []),
        difficulty=metadata.get("difficulty", "Medium"),
        description_html=description_html,
        starting_code=starting_code,
    )


@app.get("/api/questions/{slug}/solution", response_model=QuestionSolution)
async def get_solution(slug: str):
    """Get the solution for a specific question."""
    question_dir = DB_DIR / slug

    if not question_dir.exists():
        raise HTTPException(status_code=404, detail="Question not found")

    solution_file = question_dir / "solution.py"
    if not solution_file.exists():
        raise HTTPException(status_code=404, detail="Solution not found")

    with open(solution_file, "r") as f:
        solution_code = f.read()

    # Optional written solution (markdown)
    solution_md_file = question_dir / "solution.md"
    solution_html = None
    if solution_md_file.exists():
        with open(solution_md_file, "r") as f:
            solution_md = f.read()
        md.reset()
        solution_html = md.convert(solution_md)

    return QuestionSolution(
        slug=slug, solution_code=solution_code, solution_html=solution_html
    )


@app.get("/api/search")
async def search_questions(tags: str = ""):
    """Search questions by tags."""
    all_questions = get_all_questions()

    if not tags:
        return all_questions

    search_tags = [t.strip().lower() for t in tags.split(",") if t.strip()]

    if not search_tags:
        return all_questions

    # Filter questions that have any of the search tags
    filtered = []
    for q in all_questions:
        question_tags = [t.lower() for t in q.tags]
        if any(tag in question_tags for tag in search_tags):
            filtered.append(q)

    return filtered


def _truncate_output(text: str, *, max_chars: int = 80_000) -> str:
    if len(text) <= max_chars:
        return text
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head
    return text[:keep_head] + "\n... <output truncated> ...\n" + text[-keep_tail:]


def _run_ruff_format(
    code: str, *, stdin_filename: str
) -> tuple[str | None, str | None]:
    """
    Run `ruff format` on code passed via stdin.
    Returns (formatted_code, error).
    """
    ruff_bin = shutil.which("ruff")
    if not ruff_bin:
        return None, "ruff not found on PATH"

    # Ruff supports reading from stdin via '-' with --stdin-filename.
    candidates: list[list[str]] = [
        [ruff_bin, "format", "--stdin-filename", stdin_filename, "-"],
        [ruff_bin, "format", "--stdin-filename", stdin_filename],
    ]

    last_err = None
    for cmd in candidates:
        try:
            proc = subprocess.run(
                cmd,
                input=code,
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout, None
            # Some invocations may not emit stdout; treat that as failure.
            last_err = (proc.stderr or "").strip() or f"ruff exited {proc.returncode}"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)

    return None, last_err or "ruff format failed"


@app.post("/api/format/python", response_model=FormatPythonResponse)
async def format_python(payload: FormatPythonRequest, request: Request):
    """
    Format Python code using Ruff (no persistence).
    """
    t0 = time.perf_counter()
    client = request.client.host if request.client else "unknown"
    q = payload.question_slug or "unknown"
    logger.info(
        "format_python request client=%s question=%s filename=%s bytes=%d",
        client,
        q,
        payload.filename,
        len(payload.code.encode("utf-8")),
    )

    formatted, err = _run_ruff_format(payload.code, stdin_filename=payload.filename)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if err or formatted is None:
        logger.warning(
            "format_python failed client=%s question=%s err=%s duration_ms=%.1f",
            client,
            q,
            err,
            dt_ms,
        )
        return FormatPythonResponse(
            formatted_code=payload.code,
            changed=False,
            used_ruff=False,
            error=err or "ruff format failed",
        )

    changed = formatted != payload.code
    logger.info(
        "format_python ok client=%s question=%s changed=%s duration_ms=%.1f",
        client,
        q,
        changed,
        dt_ms,
    )
    return FormatPythonResponse(
        formatted_code=formatted,
        changed=changed,
        used_ruff=True,
        error=None,
    )


@app.post("/api/run", response_model=RunCodeResponse)
async def run_code(payload: RunCodeRequest, request: Request):
    """
    Run a question's tests against the provided candidate code in a subprocess.

    We avoid importing/running candidate code in-process to reduce the chance of
    crashing/hanging the web app. This is not a security sandbox, but it does
    add basic isolation and a hard timeout.
    """
    t0 = time.perf_counter()
    client = request.client.host if request.client else "unknown"

    slug = (payload.question_slug or "").strip()
    if not slug or not re.fullmatch(r"[a-z0-9_]+", slug):
        raise HTTPException(status_code=400, detail="Invalid question_slug")

    question_dir = DB_DIR / slug
    if not question_dir.exists():
        raise HTTPException(status_code=404, detail="Question not found")

    # Ensure tests exist (consistent error message)
    if not (question_dir / "tests.py").exists():
        raise HTTPException(status_code=404, detail="Question tests not found")

    timeout_s = 6.0
    code_bytes = len(payload.code.encode("utf-8", errors="replace"))

    with tempfile.TemporaryDirectory(prefix=f"mlbites_run_{slug}_") as td:
        candidate_path = Path(td) / "candidate.py"
        candidate_path.write_text(payload.code, encoding="utf-8")

        env = os.environ.copy()
        src_dir = str(BASE_DIR / "src")
        env["PYTHONPATH"] = src_dir + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )

        cmd = [
            sys.executable,
            "-m",
            "mlbites.verify",
            slug,
            "--candidate",
            str(candidate_path),
        ]

        logger.info(
            "run_code request client=%s question=%s bytes=%d timeout_s=%.1f",
            client,
            slug,
            code_bytes,
            timeout_s,
        )

        try:
            proc = subprocess.run(
                cmd,
                cwd=BASE_DIR,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_s,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            exit_code = proc.returncode
        except subprocess.TimeoutExpired as e:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            stdout = (e.stdout or "") if isinstance(e.stdout, str) else ""
            stderr = (e.stderr or "") if isinstance(e.stderr, str) else ""
            stderr = (stderr + "\n[timeout] Execution timed out.\n").lstrip("\n")
            logger.warning(
                "run_code timeout client=%s question=%s duration_ms=%.1f",
                client,
                slug,
                dt_ms,
            )
            return RunCodeResponse(
                status="timeout",
                exit_code=None,
                stdout=_truncate_output(stdout),
                stderr=_truncate_output(stderr),
                duration_ms=dt_ms,
            )

    dt_ms = (time.perf_counter() - t0) * 1000.0
    stdout = _truncate_output(stdout)
    stderr = _truncate_output(stderr)

    if exit_code == 0:
        status = "pass"
    elif exit_code == 1:
        status = "fail"
    else:
        status = "error"

    logger.info(
        "run_code done client=%s question=%s status=%s exit_code=%s duration_ms=%.1f",
        client,
        slug,
        status,
        exit_code,
        dt_ms,
    )

    return RunCodeResponse(
        status=status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=dt_ms,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
