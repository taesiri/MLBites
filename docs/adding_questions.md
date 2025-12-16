# Adding New Questions to MLBites

This guide explains how to add new PyTorch interview questions to the MLBites database.

## Directory Structure

Each question lives in its own folder under `db/`:

```
db/
├── linear_regression/
│   ├── metadata.json      # Question metadata
│   ├── question.md        # Problem description
│   ├── starting_point.py  # Starter code for users
│   ├── tests.py           # Deterministic verification tests (required)
│   └── solution.py        # Reference solution
├── logistic_regression/
│   └── ...
└── your_new_question/
    └── ...
```

## Step-by-Step Guide

### 1. Create a New Folder

Create a folder under `db/` with a descriptive, snake_case name:

```bash
mkdir db/your_question_name
```

### 2. Create `metadata.json`

This file contains the question's display information:

```json
{
    "title": "Your Question Title",
    "category": "Category Name",
    "framework": "pytorch",
    "tags": ["tag1", "tag2", "tag3"],
    "difficulty": "Easy",
    "relevant_questions": []
}
```

| Field | Description | Valid Values |
|-------|-------------|--------------|
| `title` | Display title | Any string |
| `category` | Topic grouping | "Basics", "CNN", "RNN", "Transformers", etc. |
| `framework` | Primary implementation framework | `"pytorch"` or `"numpy"` |
| `tags` | Searchable tags | Array of strings |
| `difficulty` | Skill level | "Easy", "Medium", "Hard" |
| `relevant_questions` | Related question slugs for recommendations | Array of existing `db/<slug>/` folder names |

### 3. Create `question.md`

Write the problem description in Markdown:

```markdown
# Question Title

## Problem Statement

Describe what the user needs to implement.

## Requirements

- Requirement 1
- Requirement 2

## Function Signature

```python
def your_function(arg1: Type1) -> ReturnType:
    """Docstring explaining the function."""
    pass
```

## Example

```python
# Show example usage and expected output
result = your_function(input_data)
print(result)  # Expected: ...
```

## Hints

- Hint 1
- Hint 2
```

### 4. Create `starting_point.py`

Provide skeleton code with `TODO` comments:

```python
"""
Question Title - Starting Point

Brief description of the task.
"""

import torch
import torch.nn as nn


def your_function(arg1):
    """
    Docstring explaining what to implement.
    
    Args:
        arg1: Description
        
    Returns:
        Description of return value
    """
    # TODO: Implement this function
    pass


if __name__ == "__main__":
    # Test code that runs when user clicks "Run"
    result = your_function(test_input)
    print(f"Result: {result}")
```

### 5. Create `solution.py`

Provide a complete, working reference solution:

```python
"""
Question Title - Solution

Complete working implementation.
"""

import torch
import torch.nn as nn


def your_function(arg1):
    """Full implementation."""
    # Complete working code here
    return result


if __name__ == "__main__":
    # Same test code as starting_point.py
    result = your_function(test_input)
    print(f"Result: {result}")
```

### 6. Create `tests.py` (verification)

Each question MUST include a `tests.py` so we can automatically validate a candidate solution.

`tests.py` must define:

```python
from __future__ import annotations

from types import ModuleType

def run_tests(candidate: ModuleType) -> None:
    # call candidate.your_function(...) and assert expected outputs
    # raise AssertionError with a helpful message on failure
    raise NotImplementedError
```

To run tests locally:

```bash
uv run python -m mlbites.verify your_question_name
```

## Best Practices

1. **Test your solution** - Run `solution.py` to verify it works
2. **Keep it focused** - Each question should test one concept
3. **Provide good hints** - Help users without giving away the answer
4. **Use type hints** - Makes the expected API clear
5. **Include edge cases** - In the test code when appropriate

## Example: Complete Question Structure

See `db/linear_regression/` for a complete example:

- [metadata.json](../db/linear_regression/metadata.json)
- [question.md](../db/linear_regression/question.md)
- [starting_point.py](../db/linear_regression/starting_point.py)
- [solution.py](../db/linear_regression/solution.py)
