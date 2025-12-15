from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_module_from_path(path: Path, *, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_question(slug: str, *, candidate_path: Path | None = None) -> None:
    """
    Verify a candidate solution against db/<slug>/tests.py.

    Raises:
        FileNotFoundError, AssertionError, RuntimeError
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    db_dir = base_dir / "db"
    question_dir = db_dir / slug

    tests_path = question_dir / "tests.py"
    if not tests_path.exists():
        raise FileNotFoundError(f"Missing tests file: {tests_path}")

    if candidate_path is None:
        candidate_path = question_dir / "solution.py"
    candidate_path = candidate_path.resolve()
    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing candidate file: {candidate_path}")

    tests_mod = _load_module_from_path(tests_path, module_name=f"mlbites_tests_{slug}")
    if not hasattr(tests_mod, "run_tests"):
        raise RuntimeError(
            f"{tests_path} must define run_tests(candidate: ModuleType) -> None"
        )

    candidate_mod = _load_module_from_path(
        candidate_path, module_name=f"mlbites_candidate_{slug}"
    )
    tests_mod.run_tests(candidate_mod)  # type: ignore[attr-defined]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mlbites.verify",
        description="Verify a question's candidate solution against its db/<slug>/tests.py",
    )
    p.add_argument(
        "slug", help="Question slug under db/, e.g. layernorm_forward_backward_numpy"
    )
    p.add_argument(
        "--candidate",
        type=Path,
        default=None,
        help="Path to candidate .py file (defaults to db/<slug>/solution.py)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    try:
        verify_question(args.slug, candidate_path=args.candidate)
    except AssertionError as e:
        print(f"[FAIL] {args.slug}: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {args.slug}: {e}", file=sys.stderr)
        return 2
    else:
        print(f"[PASS] {args.slug}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
