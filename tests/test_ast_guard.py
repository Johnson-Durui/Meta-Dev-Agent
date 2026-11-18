"""Tests for lexical-scope-aware AST input() and dangerous-call static analysis."""

from __future__ import annotations

from meta_dev.sandbox import analyze_python_ast, should_use_interactive_execution


def test_detects_bare_builtin_input() -> None:
    code = """
name = input("Enter your name: ")
print(f"Hello, {name}")
"""
    report = analyze_python_ast(code)
    assert report.syntax_valid is True
    assert report.has_blocking_input is True
    assert any("builtin input()" in item for item in report.blocking_calls)


def test_detects_builtins_and_aliased_input_and_stdin_readline() -> None:
    code = """
import builtins
import sys
from builtins import input as raw_prompt

a = builtins.input("a:")
b = raw_prompt("b:")
c = sys.stdin.readline()
"""
    report = analyze_python_ast(code)
    assert report.has_blocking_input is True
    assert len(report.blocking_calls) == 3


def test_allows_method_call_named_input_on_object() -> None:
    code = """
class StreamProcessor:
    def input(self, payload: str) -> str:
        return payload.upper()

proc = StreamProcessor()
result = proc.input("hello")
"""
    report = analyze_python_ast(code)
    assert report.syntax_valid is True
    assert report.has_blocking_input is False
    assert report.blocking_calls == []


def test_allows_locally_defined_input_function_at_module_and_function_scope() -> None:
    module_shadow_code = """
def input(prompt: str = "") -> str:
    return "deterministic-stub"

answer = input("Ignored prompt")
"""
    mod_report = analyze_python_ast(module_shadow_code)
    assert mod_report.has_blocking_input is False

    param_shadow_code = """
def run_step(input=lambda _: "injected"):
    return input("prompt")
"""
    fn_report = analyze_python_ast(param_shadow_code)
    assert fn_report.has_blocking_input is False


def test_class_method_named_input_does_not_hide_bare_builtin_input_in_sibling_method() -> None:
    code = """
class FormWidget:
    def input(self) -> str:
        return "field"

    def ask_terminal(self) -> str:
        # In Python LEGB scoping, class body names are not in method lexical scope!
        return input("Real blocking terminal prompt: ")
"""
    report = analyze_python_ast(code)
    assert report.has_blocking_input is True
    assert len(report.blocking_calls) == 1


def test_dangerous_calls_and_syntax_errors_reported() -> None:
    risky_code = """
import os
import subprocess

x = eval("1 + 2")
os.system("echo hi")
subprocess.run("echo shell", shell=True)
"""
    report = analyze_python_ast(risky_code)
    assert report.syntax_valid is True
    assert report.has_blocking_input is False
    assert len(report.dangerous_calls) == 3

    broken = analyze_python_ast("def broken(:\n    pass")
    assert broken.syntax_valid is False
    assert broken.syntax_error is not None


def test_should_use_interactive_execution_respects_ast_scope() -> None:
    assert should_use_interactive_execution("write a script", "obj.input('ok')") is False
    assert should_use_interactive_execution("write a script", "x = input('prompt:')") is True
    assert should_use_interactive_execution("我想在终端里玩猜数字", "print('hi')") is True
