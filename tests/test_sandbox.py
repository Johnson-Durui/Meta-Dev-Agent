"""Tests for isolated subprocess sandbox execution, timeout termination, and stdin=DEVNULL."""

from __future__ import annotations

import sys
from pathlib import Path

from meta_dev.sandbox import (
    execute_workspace_in_sandbox,
    format_error_reflection_feedback,
    run_command_in_sandbox,
)


def test_sandbox_executes_passing_pytest_suite(tmp_path: Path) -> None:
    (tmp_path / "math_mod.py").write_text(
        "def multiply(a: int, b: int) -> int:\n    return a * b\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_math.py").write_text(
        "from math_mod import multiply\n\ndef test_multiply() -> None:\n    assert multiply(6, 7) == 42\n",
        encoding="utf-8",
    )

    report = execute_workspace_in_sandbox(tmp_path, timeout=10.0)
    assert report.passed is True
    assert report.timed_out is False
    assert report.exit_code == 0
    assert "pytest" in " ".join(report.command)


def test_sandbox_captures_failing_unit_tests_and_formats_error_reflection(tmp_path: Path) -> None:
    (tmp_path / "buggy.py").write_text(
        "def divide(a: float, b: float) -> float:\n    return a * b  # wrong operator\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_buggy.py").write_text(
        "from buggy import divide\n\ndef test_divide() -> None:\n    assert divide(10, 2) == 5\n",
        encoding="utf-8",
    )

    report = execute_workspace_in_sandbox(tmp_path, timeout=10.0)
    assert report.passed is False
    assert report.exit_code != 0
    feedback = format_error_reflection_feedback(report)
    assert "STATUS: FAILED" in feedback
    assert "AssertionError" in feedback or "assert " in feedback


def test_sandbox_terminates_infinite_loop_on_timeout(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text(
        "while True:\n    pass\n",
        encoding="utf-8",
    )
    report = execute_workspace_in_sandbox(tmp_path, timeout=0.6)
    assert report.timed_out is True
    assert report.passed is False
    assert "[TIMEOUT]" in report.formatted_output


def test_sandbox_stdin_devnull_never_blocks_and_ast_guard_intercepts(tmp_path: Path) -> None:
    script = tmp_path / "blocking_prompt.py"
    script.write_text(
        "val = input('Waiting forever? ')\nprint(val)\n",
        encoding="utf-8",
    )

    # 1. AST static guard intercepts before execution when block_on_ast_input=True
    guarded = execute_workspace_in_sandbox(tmp_path, default_entrypoint="blocking_prompt.py", block_on_ast_input=True)
    assert guarded.passed is False
    assert "[AST_INPUT_GUARD]" in guarded.formatted_output

    # 2. Even if executed directly in subprocess, stdin=DEVNULL immediately raises EOFError without hanging!
    direct = run_command_in_sandbox(
        [sys.executable, str(script)],
        cwd=tmp_path,
        timeout=5.0,
    )
    assert direct.timed_out is False
    assert direct.passed is False
    assert "EOFError" in direct.stderr
