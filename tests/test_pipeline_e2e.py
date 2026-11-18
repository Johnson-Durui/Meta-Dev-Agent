"""End-to-end tests for the 5-stage --dry-run Software Factory, Self-Healing, and CLI facades."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import main as root_main_facade
import meta_dev_core as root_core_facade
from meta_dev.arbiter import build_arbitration_template
from meta_dev.cli import main as cli_main
from meta_dev.git_ops import sanitize_commit_message
from meta_dev.llm import (
    LLMConfigError,
    MockLLMClient,
    OpenAICompatibleLLMClient,
    normalize_gateway_url,
    resolve_api_base_url,
)
from meta_dev.orchestrator import SoftwareFactoryOrchestrator


def test_dry_run_e2e_self_heals_round1_bug_commits_git_and_writes_manifest(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_factory_ws"
    exit_code = cli_main(
        [
            "--requirement",
            "Build a prioritized task scheduling package with CLI and pytest unit tests",
            "--workspace",
            str(workspace),
            "--dry-run",
            "--auto-git",
            "--non-interactive",
            "--quiet",
        ]
    )
    assert exit_code == 0

    manifest_file = workspace / "run_manifest.json"
    assert manifest_file.is_file()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    assert manifest["dry_run"] is True
    assert manifest["stages_completed"] == ["Clarify", "Design", "Build", "Verify", "Release"]
    # Round 0 failed on the deliberate sort bug -> Self-Heal Round 1 fixed it -> Round 1 passed!
    assert manifest["heal_rounds_used"] == 1
    assert len(manifest["heal_history"]) == 2
    assert manifest["heal_history"][0]["sandbox_passed"] is False
    assert manifest["heal_history"][1]["sandbox_passed"] is True

    assert manifest["quality_gate"]["passed"] is True
    assert manifest["release_decision"] == "GO"
    assert manifest["commit_hash"] is not None and len(manifest["commit_hash"]) >= 7
    assert (workspace / ".git").is_dir()
    assert (workspace / "task_forge" / "engine.py").is_file()
    assert (workspace / "tests" / "test_task_forge.py").is_file()


def test_unfixable_defect_triggers_no_go_and_blocks_git_commit(tmp_path: Path) -> None:
    workspace = tmp_path / "blocked_ws"
    mock_client = MockLLMClient(simulate_unfixable_bug=True)
    orchestrator = SoftwareFactoryOrchestrator(
        llm_client=mock_client,
        workspace_dir=workspace,
        max_heal_rounds=2,
        auto_git=True,
        non_interactive=True,
        verbose=False,
    )
    summary = orchestrator.run("Build task engine")

    assert summary.heal_rounds_used == 2
    assert summary.quality_gate.passed is False
    assert summary.release_decision == "NO-GO"
    assert summary.commit_hash is None
    assert not (workspace / ".git").exists()
    assert len(summary.arbiter_decisions) >= 1
    assert (workspace / "run_manifest.json").is_file()


def test_backwards_compatible_facades_export_api_and_cli(capsys) -> None:
    assert hasattr(root_core_facade, "PM_SYSTEM_PROMPT")
    assert hasattr(root_core_facade, "arbitrate_issue")
    assert hasattr(root_main_facade, "SoftwareFactoryOrchestrator")

    rc = root_core_facade.main(["--print-arbitration-template"])
    assert rc == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert "issue" in parsed and "claims" in parsed


def test_cli_arbitrate_file_with_explain_and_main_facade_subprocess(tmp_path: Path) -> None:
    arb_json = tmp_path / "arb_input.json"
    arb_json.write_text(
        json.dumps(build_arbitration_template(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rc = cli_main(
        [
            "--arbitrate-file",
            str(arb_json),
            "--explain-arbitration",
            "--dry-run",
        ]
    )
    assert rc == 0

    repo_root = Path(__file__).resolve().parent.parent
    sub_ws = tmp_path / "subproc_ws"
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "main.py"),
            "--dry-run",
            "--workspace",
            str(sub_ws),
            "--non-interactive",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0
    assert (sub_ws / "run_manifest.json").is_file()


def test_commit_sanitizer_and_llm_config_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    noisy = (
        "ROLE: PM\nSTAGE: Release\n"
        "Commit Message: \"feat: add deterministic self-healing sandbox router\"\n"
        "Extra commentary line"
    )
    assert sanitize_commit_message(noisy) == "feat: add deterministic self-healing sandbox router"

    assert normalize_gateway_url("https://api.example.com/v1/") == "https://api.example.com"
    assert resolve_api_base_url("https://api.example.com") == "https://api.example.com/v1"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMConfigError):
        OpenAICompatibleLLMClient(api_key="")

