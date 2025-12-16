from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


Framework = str  # keep lightweight; app models use Literal


TAG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SLUG_RE = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class Issue:
    slug: str
    path: Path
    message: str


def _expected_framework_from_slug(slug: str) -> Framework | None:
    if slug.endswith("_pytorch"):
        return "pytorch"
    if slug.endswith("_numpy"):
        return "numpy"
    return None


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_metadata(
    *, slug: str, path: Path, data: dict, known_slugs: set[str]
) -> list[Issue]:
    issues: list[Issue] = []

    if not SLUG_RE.fullmatch(slug):
        issues.append(
            Issue(slug, path, "Invalid slug folder name (expected lower_snake_case).")
        )

    required_keys = {
        "title",
        "category",
        "framework",
        "tags",
        "difficulty",
        "relevant_questions",
    }
    allowed_keys = set(required_keys)

    missing = sorted(required_keys - set(data.keys()))
    extra = sorted(set(data.keys()) - allowed_keys)
    for k in missing:
        issues.append(Issue(slug, path, f"Missing required key: {k!r}"))
    for k in extra:
        issues.append(Issue(slug, path, f"Unexpected key (not in schema): {k!r}"))

    title = data.get("title")
    if not isinstance(title, str) or not title.strip():
        issues.append(Issue(slug, path, "Invalid 'title' (expected non-empty string)."))

    category = data.get("category")
    if not isinstance(category, str) or not category.strip():
        issues.append(
            Issue(slug, path, "Invalid 'category' (expected non-empty string).")
        )

    framework = data.get("framework")
    if framework not in {"pytorch", "numpy"}:
        issues.append(
            Issue(slug, path, "Invalid 'framework' (expected 'pytorch' or 'numpy').")
        )
        framework = None

    tags = data.get("tags")
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        issues.append(Issue(slug, path, "Invalid 'tags' (expected list[str])."))
        tags_list: list[str] = []
    else:
        tags_list = tags

    # Normalize checks (no mutation; just report)
    lowered = [t.strip() for t in tags_list]
    if any(t != t.strip() for t in tags_list):
        issues.append(Issue(slug, path, "Tag contains leading/trailing whitespace."))
    if any(t.lower() != t for t in lowered):
        issues.append(Issue(slug, path, "Tags must be lowercase."))
    if len(set(lowered)) != len(lowered):
        issues.append(Issue(slug, path, "Tags contain duplicates."))
    for t in lowered:
        if not TAG_RE.fullmatch(t):
            issues.append(
                Issue(
                    slug,
                    path,
                    f"Tag {t!r} is not kebab-case ([a-z0-9] and '-' only).",
                )
            )

    if framework and framework not in lowered:
        issues.append(
            Issue(
                slug, path, f"'framework' is {framework!r} but tags do not include it."
            )
        )

    expected_fw = _expected_framework_from_slug(slug)
    if expected_fw and framework and expected_fw != framework:
        issues.append(
            Issue(
                slug,
                path,
                f"'framework' is {framework!r} but slug suffix implies {expected_fw!r}.",
            )
        )
    if expected_fw and expected_fw not in lowered:
        issues.append(
            Issue(
                slug,
                path,
                f"Slug suffix implies {expected_fw!r} but tags do not include it.",
            )
        )

    difficulty = data.get("difficulty")
    if difficulty not in {"Easy", "Medium", "Hard"}:
        issues.append(
            Issue(slug, path, "Invalid 'difficulty' (expected Easy|Medium|Hard).")
        )

    rel = data.get("relevant_questions")
    if not isinstance(rel, list) or not all(isinstance(x, str) for x in rel):
        issues.append(
            Issue(slug, path, "Invalid 'relevant_questions' (expected list[str]).")
        )
        rel_list: list[str] = []
    else:
        rel_list = rel
    for other in rel_list:
        if other not in known_slugs:
            issues.append(
                Issue(
                    slug, path, f"relevant_questions contains unknown slug: {other!r}"
                )
            )
        if other == slug:
            issues.append(
                Issue(slug, path, "relevant_questions must not include itself.")
            )

    return issues


def validate_db(db_dir: Path) -> tuple[list[Issue], dict[str, set[str]]]:
    issues: list[Issue] = []
    categories: set[str] = set()
    tags: set[str] = set()

    question_dirs = [p for p in db_dir.iterdir() if p.is_dir()]
    known_slugs = {p.name for p in question_dirs}

    for qdir in sorted(question_dirs, key=lambda p: p.name):
        slug = qdir.name
        meta_path = qdir / "metadata.json"
        if not meta_path.exists():
            issues.append(Issue(slug, meta_path, "Missing metadata.json"))
            continue

        try:
            data = _load_json(meta_path)
        except Exception as e:  # noqa: BLE001
            issues.append(Issue(slug, meta_path, f"Failed to parse JSON: {e}"))
            continue

        if isinstance(data.get("category"), str):
            categories.add(data["category"])
        if isinstance(data.get("tags"), list):
            tags.update([t for t in data["tags"] if isinstance(t, str)])

        issues.extend(
            _validate_metadata(
                slug=slug, path=meta_path, data=data, known_slugs=known_slugs
            )
        )

    stats = {"categories": categories, "tags": tags}
    return issues, stats


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mlbites.validate_metadata",
        description="Validate db/*/metadata.json consistency (schema, tags, categories, framework).",
    )
    p.add_argument(
        "--db-dir",
        type=Path,
        default=None,
        help="Path to db/ (defaults to <repo>/db).",
    )
    p.add_argument(
        "--show-stats",
        action="store_true",
        help="Print unique categories and tags after validation.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    base_dir = Path(__file__).resolve().parent.parent.parent
    db_dir = (args.db_dir or (base_dir / "db")).resolve()

    if not db_dir.exists():
        print(f"[ERROR] db dir not found: {db_dir}", file=sys.stderr)
        return 2

    issues, stats = validate_db(db_dir)
    if issues:
        for it in issues:
            print(f"[FAIL] {it.slug}: {it.path}: {it.message}", file=sys.stderr)
        print(f"\n{len(issues)} issue(s) found.", file=sys.stderr)
        return 1

    print("[PASS] metadata.json validation passed.")

    if args.show_stats:
        cats = sorted(stats["categories"])
        tgs = sorted(stats["tags"])
        print("\nCategories:")
        for c in cats:
            print(f"- {c}")
        print("\nTags:")
        for t in tgs:
            print(f"- {t}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
