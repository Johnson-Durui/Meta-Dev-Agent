"""Tests for structured TurnEnvelope parsing, multi-file extraction, and path traversal defense."""

from __future__ import annotations

from pathlib import Path

import pytest

from meta_dev.parser import (
    PathTraversalError,
    extract_and_save_generated_artifacts,
    extract_explicit_file_blocks,
    parse_turn_envelope,
    validate_safe_relative_path,
)


def test_multi_file_extraction_writes_clean_project_layout(tmp_path: Path) -> None:
    llm_output = """
ROLE: DEV
STAGE: Build

PROPOSAL:
File: calc_pkg/__init__.py
```python
from calc_pkg.ops import add
__all__ = ["add"]
```

### File: calc_pkg/ops.py
```python
def add(a: int, b: int) -> int:
    return a + b
```

```python:tests/test_ops.py
from calc_pkg import add

def test_add() -> None:
    assert add(2, 3) == 5
```
"""
    result = extract_and_save_generated_artifacts(llm_output, workspace_dir=tmp_path)
    assert result["artifact_kind"] == "project"
    assert set(result["files"].keys()) == {
        "calc_pkg/__init__.py",
        "calc_pkg/ops.py",
        "tests/test_ops.py",
    }
    assert (tmp_path / "calc_pkg" / "ops.py").is_file()
    assert (tmp_path / "tests" / "test_ops.py").is_file()


@pytest.mark.parametrize(
    "malicious_path",
    [
        "../escaped.py",
        "../../etc/passwd",
        "pkg/../../outside.py",
        "..\\windows_escape.py",
        "/etc/shadow",
        "C:\\Windows\\System32\\drivers\\etc\\hosts",
        "",
        ".",
    ],
)
def test_validate_safe_relative_path_rejects_traversal_and_absolute_paths(
    tmp_path: Path,
    malicious_path: str,
) -> None:
    with pytest.raises(PathTraversalError):
        validate_safe_relative_path(malicious_path, workspace_dir=tmp_path)


@pytest.mark.parametrize(
    "payload",
    [
        "File: ../pwned.py\n```python\nprint('escaped')\n```",
        "### File: sub/../../outside.py\n```python\nx = 1\n```",
        "```python:../escape_fence.py\nx = 1\n```",
        "File: /abs/root_write.py\n```python\nx = 1\n```",
    ],
)
def test_multi_file_extractor_blocks_path_traversal_attacks(
    tmp_path: Path,
    payload: str,
) -> None:
    with pytest.raises(PathTraversalError):
        extract_explicit_file_blocks(payload, workspace_dir=tmp_path)

    with pytest.raises(PathTraversalError):
        extract_and_save_generated_artifacts(payload, workspace_dir=tmp_path)


def test_parse_turn_envelope_extracts_metadata_objections_and_consensus(tmp_path: Path) -> None:
    raw_turn = """
ROLE: QA
STAGE: Verify

CURRENT_POSITION:
- 发现优先级排序反转缺陷 B1，阻塞发布。

QUESTIONS:
- TO_PM: None
- TO_DEV: 是否遗漏降序负号？
- TO_QA: None

OBJECTIONS:
- [DEV][B1][质量][Blocker]
  Reason: 排序方向写反导致单元测试失败
  Risk: 高优任务被延后处理
  Counterproposal: 改为 -item.priority
  Accept_Condition: pytest 全绿

RESPONSES_TO_OBJECTIONS:
None

PROPOSAL:
修复排序键并重跑测试。

AGREEMENTS:
- 模型边界校验通过

OPEN_ISSUES:
- B1 (Blocker)

CONSENSUS_STATUS:
- Not Ready: 存在未解决的 Blocker 异议
"""
    env = parse_turn_envelope(raw_turn, workspace_dir=tmp_path)
    assert env.role == "QA"
    assert env.stage == "Verify"
    assert env.consensus_status == "Not Ready"
    assert env.has_reached_consensus is False
    assert env.questions["TO_DEV"] == ["是否遗漏降序负号？"]
    assert len(env.objections) == 1
    assert env.objections[0].is_blocker is True
    assert env.objections[0].reference_id == "B1"
    assert env.objections[0].counterproposal == "改为 -item.priority"
