"""AST input() static guard, isolated subprocess sandbox, smart test router, and error reflection."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from meta_dev.models import ASTGuardReport, SandboxReport
from meta_dev.parser import (
    extract_and_save_generated_artifacts,
    select_primary_generated_file,
)


@dataclass
class _LexicalScope:
    """Tracks bound names in Python LEGB scopes so custom `input` never triggers false positives."""

    is_class_scope: bool = False
    bound_names: set[str] = field(default_factory=set)
    builtin_input_aliases: set[str] = field(default_factory=set)
    builtins_module_aliases: set[str] = field(default_factory=lambda: {"builtins", "__builtins__"})
    getpass_aliases: set[str] = field(default_factory=set)
    getpass_module_aliases: set[str] = field(default_factory=set)
    sys_module_aliases: set[str] = field(default_factory=set)


def _collect_target_names(target: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.update(_collect_target_names(elt))
    elif isinstance(target, ast.Starred):
        names.update(_collect_target_names(target.value))
    return names


def _prebind_scope_declarations(body: list[ast.stmt], scope: _LexicalScope) -> None:
    """Pre-collect function/class/variable/import bindings declared directly within a block."""
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope.bound_names.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                scope.bound_names.update(_collect_target_names(tgt))
        elif isinstance(stmt, ast.AnnAssign) and stmt.target is not None:
            scope.bound_names.update(_collect_target_names(stmt.target))
        elif isinstance(stmt, ast.AugAssign):
            scope.bound_names.update(_collect_target_names(stmt.target))
        elif isinstance(stmt, ast.For | ast.AsyncFor):
            scope.bound_names.update(_collect_target_names(stmt.target))
            _prebind_scope_declarations(stmt.body, scope)
            _prebind_scope_declarations(stmt.orelse, scope)
        elif isinstance(stmt, ast.With | ast.AsyncWith):
            for item in stmt.items:
                if item.optional_vars is not None:
                    scope.bound_names.update(_collect_target_names(item.optional_vars))
            _prebind_scope_declarations(stmt.body, scope)
        elif isinstance(stmt, ast.If):
            _prebind_scope_declarations(stmt.body, scope)
            _prebind_scope_declarations(stmt.orelse, scope)
        elif isinstance(stmt, ast.Try):
            _prebind_scope_declarations(stmt.body, scope)
            _prebind_scope_declarations(stmt.orelse, scope)
            _prebind_scope_declarations(stmt.finalbody, scope)
            for handler in stmt.handlers:
                if handler.name:
                    scope.bound_names.add(handler.name)
                _prebind_scope_declarations(handler.body, scope)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                bound = alias.asname or alias.name.split(".")[0]
                scope.bound_names.add(bound)
                if alias.name == "builtins":
                    scope.builtins_module_aliases.add(bound)
                elif alias.name == "getpass":
                    scope.getpass_module_aliases.add(bound)
                elif alias.name == "sys":
                    scope.sys_module_aliases.add(bound)
        elif isinstance(stmt, ast.ImportFrom):
            mod = stmt.module or ""
            for alias in stmt.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name
                scope.bound_names.add(bound)
                if mod in {"builtins", "__builtins__"} and alias.name == "input":
                    scope.builtin_input_aliases.add(bound)
                elif mod == "getpass" and alias.name == "getpass":
                    scope.getpass_aliases.add(bound)


class _ASTInputAndSafetyVisitor(ast.NodeVisitor):
    """Lexical-scope-aware visitor detecting blocking stdin reads and risky dynamic calls."""

    def __init__(self) -> None:
        self.scopes: list[_LexicalScope] = []
        self.blocking_calls: list[str] = []
        self.dangerous_calls: list[str] = []

    def _lookup_visible_scopes(self) -> list[_LexicalScope]:
        """Return scopes visible under Python LEGB rules (skipping enclosing class bodies)."""
        if not self.scopes:
            return []
        visible: list[_LexicalScope] = [self.scopes[-1]]
        for scope in reversed(self.scopes[:-1]):
            if not scope.is_class_scope:
                visible.append(scope)
        return visible

    def _is_name_shadowed(self, name: str) -> bool:
        for scope in self._lookup_visible_scopes():
            if name in scope.builtin_input_aliases or name in scope.getpass_aliases:
                return False
            if name in scope.bound_names:
                return True
        return False

    def _is_builtin_input_alias(self, name: str) -> bool:
        return any(name in scope.builtin_input_aliases for scope in self._lookup_visible_scopes())

    def _is_getpass_alias(self, name: str) -> bool:
        return any(name in scope.getpass_aliases for scope in self._lookup_visible_scopes())

    def _is_builtins_mod(self, name: str) -> bool:
        return any(name in scope.builtins_module_aliases for scope in self._lookup_visible_scopes())

    def _is_getpass_mod(self, name: str) -> bool:
        return any(name in scope.getpass_module_aliases for scope in self._lookup_visible_scopes())

    def _is_sys_mod(self, name: str) -> bool:
        return any(name in scope.sys_module_aliases for scope in self._lookup_visible_scopes())

    def visit_Module(self, node: ast.Module) -> None:
        scope = _LexicalScope(is_class_scope=False)
        _prebind_scope_declarations(node.body, scope)
        self.scopes.append(scope)
        self.generic_visit(node)
        self.scopes.pop()

    def _visit_function_like(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for dec in node.decorator_list:
            self.visit(dec)
        for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            self.visit(default)

        scope = _LexicalScope(is_class_scope=False)
        all_args = (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
        for arg in all_args:
            scope.bound_names.add(arg.arg)
        if node.args.vararg:
            scope.bound_names.add(node.args.vararg.arg)
        if node.args.kwarg:
            scope.bound_names.add(node.args.kwarg.arg)

        _prebind_scope_declarations(node.body, scope)
        self.scopes.append(scope)
        for stmt in node.body:
            self.visit(stmt)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_like(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_like(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            self.visit(default)
        scope = _LexicalScope(is_class_scope=False)
        for arg in list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs):
            scope.bound_names.add(arg.arg)
        if node.args.vararg:
            scope.bound_names.add(node.args.vararg.arg)
        if node.args.kwarg:
            scope.bound_names.add(node.args.kwarg.arg)
        self.scopes.append(scope)
        self.visit(node.body)
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for dec in node.decorator_list:
            self.visit(dec)
        for base in node.bases:
            self.visit(base)
        for kw in node.keywords:
            self.visit(kw)
        scope = _LexicalScope(is_class_scope=True)
        _prebind_scope_declarations(node.body, scope)
        self.scopes.append(scope)
        for stmt in node.body:
            self.visit(stmt)
        self.scopes.pop()

    def _visit_comprehension_generators(
        self,
        generators: list[ast.comprehension],
        inner_nodes: list[ast.AST],
    ) -> None:
        scope = _LexicalScope(is_class_scope=False)
        for gen in generators:
            scope.bound_names.update(_collect_target_names(gen.target))
        self.scopes.append(scope)
        for gen in generators:
            self.visit(gen.iter)
            for if_clause in gen.ifs:
                self.visit(if_clause)
        for inner in inner_nodes:
            self.visit(inner)
        self.scopes.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension_generators(node.generators, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension_generators(node.generators, [node.elt])

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension_generators(node.generators, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension_generators(node.generators, [node.key, node.value])

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        if self.scopes and isinstance(node.target, ast.Name):
            # Walrus operator binds in enclosing non-comprehension scope
            for scope in reversed(self.scopes):
                scope.bound_names.add(node.target.id)
                break
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        lineno = getattr(node, "lineno", 0)
        func = node.func

        if isinstance(func, ast.Name):
            name = func.id
            if (name == "input" and not self._is_name_shadowed("input")) or self._is_builtin_input_alias(name):
                self.blocking_calls.append(f"line {lineno}: builtin input()")
            elif self._is_getpass_alias(name) or (name == "getpass" and not self._is_name_shadowed("getpass")):
                self.blocking_calls.append(f"line {lineno}: getpass()")
            elif name in {"eval", "exec"} and not self._is_name_shadowed(name):
                self.dangerous_calls.append(f"line {lineno}: builtin {name}()")

        elif isinstance(func, ast.Attribute):
            # Check builtins.input() / __builtins__.input()
            if isinstance(func.value, ast.Name):
                receiver = func.value.id
                if func.attr == "input" and self._is_builtins_mod(receiver):
                    self.blocking_calls.append(f"line {lineno}: {receiver}.input()")
                elif func.attr == "getpass" and self._is_getpass_mod(receiver):
                    self.blocking_calls.append(f"line {lineno}: {receiver}.getpass()")
                elif receiver == "os" and func.attr == "system":
                    self.dangerous_calls.append(f"line {lineno}: os.system()")
                elif receiver == "subprocess" and func.attr in {"run", "Popen", "call", "check_output"}:
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            self.dangerous_calls.append(f"line {lineno}: subprocess.{func.attr}(shell=True)")
            # Check sys.stdin.readline() / sys.stdin.read()
            elif (
                isinstance(func.value, ast.Attribute)
                and isinstance(func.value.value, ast.Name)
                and self._is_sys_mod(func.value.value.id)
                and func.value.attr == "stdin"
                and func.attr in {"read", "readline", "readlines"}
            ):
                self.blocking_calls.append(f"line {lineno}: sys.stdin.{func.attr}()")

        self.generic_visit(node)


def analyze_python_ast(code: str) -> ASTGuardReport:
    """Perform lexical-scope-accurate static analysis on Python source code."""
    if not code or not code.strip():
        return ASTGuardReport()

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ASTGuardReport(
            has_blocking_input=False,
            syntax_valid=False,
            syntax_error=f"{exc.msg} (line {exc.lineno})",
        )

    visitor = _ASTInputAndSafetyVisitor()
    visitor.visit(tree)
    return ASTGuardReport(
        has_blocking_input=bool(visitor.blocking_calls),
        blocking_calls=visitor.blocking_calls,
        dangerous_calls=visitor.dangerous_calls,
        syntax_valid=True,
    )


def should_use_interactive_execution(user_request: str, generated_code: str = "") -> bool:
    """Determine whether user intent or generated AST requires interactive terminal execution."""
    request_text = (user_request or "").lower()
    request_keywords = (
        "在终端里玩",
        "直接运行",
        "现在开始",
        "开始游戏",
        "我想玩",
        "立即运行",
        "interactive",
        "play in terminal",
        "run it now",
    )
    if any(keyword in request_text for keyword in request_keywords):
        return True

    if not generated_code or not generated_code.strip():
        return False

    report = analyze_python_ast(generated_code)
    if not report.syntax_valid:
        return bool(re.search(r"(?<![.\w])input\s*\(", generated_code))
    return report.has_blocking_input


def pytest_available() -> bool:
    """Return True if `pytest` is importable in the active Python runtime (`sys.executable`)."""
    return importlib.util.find_spec("pytest") is not None


def project_has_tests(saved_files: dict[str, str]) -> bool:
    return any(
        path.startswith("tests/") or re.search(r"(^|/)(test_.*|.*_test)\.py$", path)
        for path in saved_files
    )


def project_prefers_pytest(saved_files: dict[str, str]) -> bool:
    for path, content in saved_files.items():
        lowered = content.lower()
        if "import pytest" in lowered or "pytest." in lowered:
            return True
        if path in {"requirements.txt", "pyproject.toml"} and "pytest" in lowered:
            return True
    return False


def build_project_test_command(saved_files: dict[str, str]) -> list[str] | None:
    """Select the optimal automated test runner (`pytest` -> `unittest discover`)."""
    if not project_has_tests(saved_files):
        return None

    if project_prefers_pytest(saved_files) and not pytest_available():
        return None

    if pytest_available():
        return [sys.executable, "-m", "pytest", "-q"]

    if any(path.startswith("tests/") for path in saved_files):
        return [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test*.py", "-v"]

    return [sys.executable, "-m", "unittest", "discover", "-v"]


def run_result_indicates_success(run_result: str) -> bool:
    """Evaluate formatted sandbox output string for clean exit and absence of failure markers."""
    if not run_result or not run_result.strip():
        return False

    if "ERROR:" in run_result or "[SYSTEM_ERROR]" in run_result or "[TIMEOUT]" in run_result:
        return False

    match = re.search(r"\[EXIT_CODE\]\s+(-?\d+)", run_result)
    if match and int(match.group(1)) != 0:
        return False

    failure_markers = (
        "traceback (most recent call last)",
        "assertionerror",
        "modulenotfounderror",
        "syntaxerror",
        "indentationerror",
    )
    lowered = run_result.lower()
    if any(marker in lowered for marker in failure_markers):
        return False

    # Check pytest/unittest summary failure pattern when exit code header wasn't present
    if match is None and re.search(r"\b\d+\s+failed\b|\bfailed\s*\(failures=", lowered):
        return False

    return True


def run_command_in_sandbox(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    timeout: float = 15.0,
    ast_warnings: list[str] | None = None,
    route_reason: str = "",
) -> SandboxReport:
    """Execute a subprocess with `stdin=DEVNULL`, strict timeout, and UTF-8 safe decoding."""
    workspace_path = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{workspace_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(workspace_path)
    )

    warnings_list = list(ast_warnings or [])
    started = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            cwd=str(workspace_path),
            env=env,
        )
        duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
    except subprocess.TimeoutExpired as exc:
        duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
        partial_out = (exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")).strip()
        partial_err = (exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")).strip()
        formatted = "ERROR: [TIMEOUT] 代码运行超时（可能是死循环或阻塞等待），已由沙盒强行终止。"
        if partial_out:
            formatted += f"\n\n[PARTIAL_STDOUT]\n{partial_out}"
        if partial_err:
            formatted += f"\n\n[PARTIAL_STDERR]\n{partial_err}"
        return SandboxReport(
            command=list(command),
            exit_code=None,
            stdout=partial_out,
            stderr=partial_err,
            timed_out=True,
            duration_ms=duration_ms,
            route_reason=route_reason,
            passed=False,
            formatted_output=formatted,
            ast_warnings=warnings_list,
        )
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
        formatted = f"[SYSTEM_ERROR] 无法运行命令 {' '.join(command)}: {exc}"
        return SandboxReport(
            command=list(command),
            exit_code=None,
            stdout="",
            stderr=str(exc),
            timed_out=False,
            duration_ms=duration_ms,
            route_reason=route_reason,
            passed=False,
            formatted_output=formatted,
            ast_warnings=warnings_list,
        )

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()

    parts: list[str] = [
        f"[COMMAND] {' '.join(command)}",
        f"[EXIT_CODE] {completed.returncode}",
    ]
    if warnings_list:
        parts.append("[AST_WARNINGS]\n" + "\n".join(f"- {w}" for w in warnings_list))
    if stdout:
        parts.append(f"[STDOUT]\n{stdout}")
    if stderr:
        parts.append(f"[STDERR]\n{stderr}")
    if not stdout and not stderr:
        parts.append("[NO_OUTPUT] 程序已运行，但没有输出。")

    formatted_output = "\n\n".join(parts)
    passed = completed.returncode == 0 and run_result_indicates_success(formatted_output)

    return SandboxReport(
        command=list(command),
        exit_code=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        duration_ms=duration_ms,
        route_reason=route_reason,
        passed=passed,
        formatted_output=formatted_output,
        ast_warnings=warnings_list,
    )


def scan_workspace_files(workspace_dir: str | Path) -> dict[str, str]:
    """Read Python and config files inside `workspace_dir` into a relative-path map."""
    root = Path(workspace_dir).resolve()
    collected: dict[str, str] = {}
    if not root.exists():
        return collected

    ignored_dirs = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        rel_parts = file_path.relative_to(root).parts
        if any(part in ignored_dirs for part in rel_parts):
            continue
        if file_path.name == "run_manifest.json":
            continue
        rel_posix = "/".join(rel_parts)
        try:
            collected[rel_posix] = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return collected


def execute_workspace_in_sandbox(
    workspace_dir: str | Path,
    *,
    saved_files: dict[str, str] | None = None,
    default_entrypoint: str = "main.py",
    timeout: float = 15.0,
    block_on_ast_input: bool = True,
) -> SandboxReport:
    """Analyze generated ASTs, route to `pytest` / `unittest` / entrypoint, and run in sandbox."""
    root = Path(workspace_dir).resolve()
    files_map = saved_files if saved_files is not None else scan_workspace_files(root)

    ast_warnings: list[str] = []
    blocking_files: list[str] = []
    syntax_failures: list[str] = []

    for rel_path, content in sorted(files_map.items()):
        if not rel_path.endswith(".py"):
            continue
        guard = analyze_python_ast(content)
        if not guard.syntax_valid:
            syntax_failures.append(f"{rel_path}: SyntaxError {guard.syntax_error}")
            continue
        if guard.has_blocking_input:
            for call_desc in guard.blocking_calls:
                msg = f"{rel_path} ({call_desc})"
                blocking_files.append(msg)
                ast_warnings.append(f"Blocking stdin call in {msg}")
        for danger in guard.dangerous_calls:
            ast_warnings.append(f"Potentially dangerous call in {rel_path} ({danger})")

    if syntax_failures:
        formatted = "ERROR: [AST_SYNTAX_ERROR] 静态语法检查未通过：\n" + "\n".join(
            f"- {item}" for item in syntax_failures
        )
        return SandboxReport(
            command=[],
            exit_code=1,
            stdout="",
            stderr=formatted,
            timed_out=False,
            duration_ms=0.0,
            route_reason="AST static syntax guard intercepted invalid Python syntax.",
            passed=False,
            formatted_output=formatted,
            ast_warnings=ast_warnings,
        )

    test_command = build_project_test_command(files_map)
    if test_command is not None:
        return run_command_in_sandbox(
            test_command,
            cwd=root,
            timeout=timeout,
            ast_warnings=ast_warnings,
            route_reason="Detected automated test suite; routing to test runner.",
        )

    if project_has_tests(files_map) and project_prefers_pytest(files_map) and not pytest_available():
        msg = (
            "ERROR: 检测到 pytest 测试文件，但当前 Python 环境未安装 pytest。"
            "请安装 pytest 或将测试改写为标准库 unittest。"
        )
        return SandboxReport(
            command=[],
            exit_code=1,
            stdout="",
            stderr=msg,
            timed_out=False,
            duration_ms=0.0,
            route_reason="Pytest required by test files but unavailable in runtime.",
            passed=False,
            formatted_output=msg,
            ast_warnings=ast_warnings,
        )

    primary_file = select_primary_generated_file(files_map, default_entrypoint)
    if not primary_file or not primary_file.endswith(".py"):
        msg = "ERROR: 未找到可执行的 Python 入口文件，也未检测到测试文件。"
        return SandboxReport(
            command=[],
            exit_code=1,
            stdout="",
            stderr=msg,
            timed_out=False,
            duration_ms=0.0,
            route_reason="No executable Python entrypoint or test suite found.",
            passed=False,
            formatted_output=msg,
            ast_warnings=ast_warnings,
        )

    if block_on_ast_input and blocking_files:
        msg = (
            "ERROR: [AST_INPUT_GUARD] 非交互沙盒拦截到阻塞型终端输入调用：\n"
            + "\n".join(f"- {item}" for item in blocking_files)
            + "\n请移除阻塞型 input() 调用，或提供参数化 CLI / 单元测试以便自动化验证。"
        )
        return SandboxReport(
            command=[sys.executable, primary_file],
            exit_code=1,
            stdout="",
            stderr=msg,
            timed_out=False,
            duration_ms=0.0,
            route_reason=f"AST static guard blocked interactive stdin call in {primary_file}.",
            passed=False,
            formatted_output=msg,
            ast_warnings=ast_warnings,
        )

    return run_command_in_sandbox(
        [sys.executable, primary_file],
        cwd=root,
        timeout=timeout,
        ast_warnings=ast_warnings,
        route_reason=f"No test files detected; executing entrypoint script {primary_file}.",
    )


def format_error_reflection_feedback(report: SandboxReport) -> str:
    """Format a failed `SandboxReport` into structured diagnostic context for QA & DEV self-healing."""
    cmd_display = " ".join(report.command) if report.command else "<blocked-before-exec>"
    status_tag = (
        "TIMED_OUT"
        if report.timed_out
        else ("PASSED" if report.passed else f"FAILED (exit_code={report.exit_code})")
    )
    sections = [
        "=== SANDBOX ERROR-REFLECTION DIAGNOSTICS ===",
        f"STATUS: {status_tag}",
        f"ROUTE_REASON: {report.route_reason}",
        f"COMMAND: {cmd_display}",
        f"DURATION_MS: {report.duration_ms}",
    ]
    if report.ast_warnings:
        sections.append("AST_STATIC_WARNINGS:\n" + "\n".join(f"  - {w}" for w in report.ast_warnings))
    if report.stdout:
        sections.append(f"STDOUT:\n{report.stdout}")
    if report.stderr:
        sections.append(f"STDERR:\n{report.stderr}")
    if not report.stdout and not report.stderr:
        sections.append(f"RAW_SUMMARY:\n{report.formatted_output}")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Backwards-compatible execution helpers
# ---------------------------------------------------------------------------


def run_and_catch_error(
    filename: str,
    timeout: int = 10,
    cwd: str | Path | None = None,
) -> str:
    """Run a Python file non-interactively (`stdin=DEVNULL`) and return formatted output."""
    report = run_command_in_sandbox(
        [sys.executable, filename],
        cwd=cwd,
        timeout=float(timeout),
        route_reason=f"Direct script execution: {filename}",
    )
    return report.formatted_output


def run_interactive_python_file(filename: str, cwd: str | Path | None = None) -> int:
    """Launch an interactive Python script attached to the user's terminal."""
    completed = subprocess.run([sys.executable, filename], check=False, cwd=cwd)
    return completed.returncode


def save_and_execute_generated_code(
    generated_code_text: str,
    user_request: str,
    filename: str = "auto_generated.py",
    timeout: int = 10,
    workspace_dir: str | Path | None = None,
    non_interactive: bool = False,
) -> dict[str, Any]:
    """Extract generated artifacts into `workspace_dir` and execute via smart sandbox routing."""
    base_dir = Path(workspace_dir).resolve() if workspace_dir is not None else Path.cwd().resolve()
    artifact_info = extract_and_save_generated_artifacts(
        generated_code_text,
        filename=filename,
        workspace_dir=base_dir,
    )
    saved_files = artifact_info["files"]
    saved_code = artifact_info["saved_code"]
    primary_file = artifact_info["primary_file"]

    if (
        not non_interactive
        and not project_has_tests(saved_files)
        and primary_file
        and should_use_interactive_execution(user_request, saved_files.get(primary_file, ""))
    ):
        exit_code = run_interactive_python_file(primary_file, cwd=base_dir)
        return {
            "mode": "interactive",
            "filename": primary_file,
            "saved_code": saved_code,
            "saved_files": saved_files,
            "artifact_kind": artifact_info["artifact_kind"],
            "execution_reason": f"Interactive script execution: {primary_file}",
            "run_result": None,
            "exit_code": exit_code,
            "sandbox_report": None,
        }

    report = execute_workspace_in_sandbox(
        base_dir,
        saved_files=saved_files,
        default_entrypoint=filename,
        timeout=float(timeout),
        block_on_ast_input=non_interactive,
    )
    return {
        "mode": "sandbox",
        "filename": primary_file or filename,
        "saved_code": saved_code,
        "saved_files": saved_files,
        "artifact_kind": artifact_info["artifact_kind"],
        "execution_reason": report.route_reason,
        "run_result": report.formatted_output,
        "exit_code": report.exit_code,
        "sandbox_report": report,
    }
