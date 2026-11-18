"""Safe workspace-isolated Git operations, commit message sanitization, and run manifest persistence."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from meta_dev.models import PipelineRunSummary


def sanitize_commit_message(raw_message: str) -> str:
    """Clean an LLM response into a single-line, safe conventional Git commit message."""
    if not isinstance(raw_message, str) or not raw_message.strip():
        return "chore: automated update via Meta-Dev factory"

    text = raw_message.replace("\r\n", "\n").replace("\r", "\n").strip()
    fenced_blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced_blocks:
        text = fenced_blocks[0].strip()

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("`").strip()
        if not line:
            continue
        if re.match(
            r"^(ROLE|STAGE|CURRENT_POSITION|QUESTIONS|OBJECTIONS|RESPONSES_TO_OBJECTIONS|"
            r"PROPOSAL|AGREEMENTS|OPEN_ISSUES|CONSENSUS_STATUS|Release\s*Verdict)\s*[:：]",
            line,
            flags=re.IGNORECASE,
        ):
            continue
        line = re.sub(
            r"^(?:[-*]\s*)?(?:commit\s*message|commit|提交说明|提交标签|标签)\s*[:：-]\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        line = line.lstrip("-* ").strip().strip("\"'“”")
        if line:
            cleaned_lines.append(line)

    message = cleaned_lines[0] if cleaned_lines else text.splitlines()[0].strip()
    message = re.sub(r"[\x00-\x1f\x7f]+", " ", message)
    message = re.sub(r"\s+", " ", message).strip().strip("\"'“”")
    if len(message) > 72:
        message = message[:72].rstrip("，。；;,. ")
    return message or "chore: automated update via Meta-Dev factory"


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=check,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Git executable not found in PATH.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or (exc.stdout or "").strip() or str(exc)
        raise RuntimeError(detail) from exc


def _ensure_local_git_identity(workspace_root: Path) -> None:
    """Configure a local fallback Git author inside `workspace_root` only if none exists."""
    name_res = _run_git(["config", "--get", "user.name"], cwd=workspace_root, check=False)
    email_res = _run_git(["config", "--get", "user.email"], cwd=workspace_root, check=False)
    if not (name_res.stdout or "").strip():
        _run_git(["config", "--local", "user.name", "Meta-Dev Agent"], cwd=workspace_root)
    if not (email_res.stdout or "").strip():
        _run_git(["config", "--local", "user.email", "meta-dev@local.invalid"], cwd=workspace_root)


def auto_git_commit(
    commit_message: str,
    workspace_dir: str | Path | None = None,
) -> str:
    """Initialize an isolated Git repository in `workspace_dir`, stage changes, and commit."""
    if not isinstance(commit_message, str) or not commit_message.strip():
        raise ValueError("commit_message 不能为空。")

    sanitized = sanitize_commit_message(commit_message)
    workspace_root = (
        Path(workspace_dir).resolve() if workspace_dir is not None else Path.cwd().resolve()
    )
    workspace_root.mkdir(parents=True, exist_ok=True)

    _run_git(["init"], cwd=workspace_root)
    _ensure_local_git_identity(workspace_root)
    _run_git(["add", "-A"], cwd=workspace_root)

    commit_result = _run_git(["commit", "-m", sanitized], cwd=workspace_root, check=False)
    combined_output = "\n".join(
        part
        for part in ((commit_result.stdout or "").strip(), (commit_result.stderr or "").strip())
        if part
    ).lower()

    if commit_result.returncode == 0:
        rev = _run_git(["rev-parse", "--short", "HEAD"], cwd=workspace_root)
        return (rev.stdout or "").strip()

    if "nothing to commit" in combined_output or "no changes added to commit" in combined_output:
        raise RuntimeError("当前没有新的代码变更可提交。")

    raise RuntimeError(combined_output or "git commit 执行失败。")


def save_run_manifest(
    summary: PipelineRunSummary,
    workspace_dir: str | Path,
) -> Path:
    """Persist `run_manifest.json` inside `workspace_dir` and update `summary.manifest_path`."""
    workspace_root = Path(workspace_dir).resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    manifest_path = workspace_root / "run_manifest.json"
    summary.manifest_path = str(manifest_path)
    manifest_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path
