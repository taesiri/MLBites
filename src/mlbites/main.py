"""FastAPI application for MLBites - PyTorch Interview Questions."""

import json
from pathlib import Path

import markdown
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .models import QuestionDetail, QuestionListItem, QuestionSolution

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_DIR = BASE_DIR / "db"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

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
    
    return QuestionSolution(slug=slug, solution_code=solution_code)


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
