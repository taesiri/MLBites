"""Pydantic models for MLBites."""

from pydantic import BaseModel


class QuestionMetadata(BaseModel):
    """Metadata for a question stored in metadata.json."""

    slug: str
    title: str
    category: str
    tags: list[str]
    difficulty: str = "Medium"


class QuestionDetail(BaseModel):
    """Full question details including content."""

    slug: str
    title: str
    category: str
    tags: list[str]
    difficulty: str
    description_html: str
    starting_code: str


class QuestionSolution(BaseModel):
    """Solution response."""

    slug: str
    solution_code: str
    solution_html: str | None = None


class QuestionListItem(BaseModel):
    """Question item for sidebar list."""

    slug: str
    title: str
    category: str
    tags: list[str]
    difficulty: str


class FormatPythonRequest(BaseModel):
    """Request to format Python code (no persistence)."""

    code: str
    filename: str = "main.py"
    question_slug: str | None = None


class FormatPythonResponse(BaseModel):
    """Response for Python formatting."""

    formatted_code: str
    changed: bool
    used_ruff: bool
    error: str | None = None
