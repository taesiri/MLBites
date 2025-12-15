"""Pydantic models for MLBites."""

from __future__ import annotations

from typing import Literal

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


class RunCodeRequest(BaseModel):
    """Request to run tests for a question against a candidate code string."""

    question_slug: str
    code: str


class RunCodeResponse(BaseModel):
    """Response for running tests."""

    status: Literal["pass", "fail", "error", "timeout"]
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: float
