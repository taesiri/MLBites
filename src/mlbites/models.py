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


class QuestionListItem(BaseModel):
    """Question item for sidebar list."""
    
    slug: str
    title: str
    category: str
    tags: list[str]
    difficulty: str
