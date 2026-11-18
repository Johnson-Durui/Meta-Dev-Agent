"""OpenAI-compatible LLM client with exponential backoff and deterministic offline MockLLMClient."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

load_dotenv(override=False)


class LLMError(RuntimeError):
    """Base error raised by LLM client backends."""


class LLMConfigError(LLMError):
    """Raised when required API credentials or endpoints are missing."""


def normalize_gateway_url(base_url: str) -> str:
    """Normalize an OpenAI-compatible gateway URL without duplicating `/v1`."""
    cleaned = (base_url or "").strip().rstrip("/")
    if not cleaned:
        return "https://api.openai.com"
    if cleaned.endswith("/v1"):
        return cleaned[:-3]
    return cleaned


def resolve_api_base_url(base_url: str | None = None) -> str:
    """Return an OpenAI SDK compatible base URL ending with `/v1` (unless custom path already versioned)."""
    raw = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip().rstrip("/")
    if not raw:
        return "https://api.openai.com/v1"
    if raw.endswith("/v1") or "/v1/" in raw or raw.endswith("/api/paas/v4"):
        return raw
    return f"{raw}/v1"


def get_default_model() -> str:
    return (
        os.getenv("META_DEV_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o"
    ).strip()


DEFAULT_BASE_URL: str = normalize_gateway_url(
    os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)
DEFAULT_MODEL: str = get_default_model()


class BaseLLMClient(ABC):
    """Abstract interface implemented by live OpenAI-compatible and offline mock clients."""

    model: str

    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        user_message: str,
        *,
        role: str = "",
        stage: str = "",
        temperature: float = 0.2,
        stream: bool = False,
    ) -> str:
        """Send a chat completion request and return the assistant's text response."""

    def list_models(self) -> list[str]:
        """Return available model identifiers supported by the backend."""
        return [self.model]


class OpenAICompatibleLLMClient(BaseLLMClient):
    """Production OpenAI SDK client with exponential backoff retry."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 90.0,
        max_retries: int = 3,
        base_retry_delay: float = 0.8,
    ) -> None:
        resolved_key = (api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")).strip()
        if not resolved_key:
            raise LLMConfigError(
                "OPENAI_API_KEY is not configured. Set OPENAI_API_KEY in your environment / .env "
                "or run with `--dry-run` for offline deterministic execution."
            )
        self.api_key = resolved_key
        self.base_url = resolve_api_base_url(base_url)
        self.model = (model or get_default_model()).strip()
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.base_retry_delay = max(0.05, base_retry_delay)
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )

    def list_models(self) -> list[str]:
        page = self._client.models.list()
        return sorted(item.id for item in page.data)

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        *,
        role: str = "",
        stage: str = "",
        temperature: float = 0.2,
        stream: bool = False,
    ) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if stream:
                    stream_resp = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=temperature,
                        stream=True,
                    )
                    chunks: list[str] = []
                    for event in stream_resp:
                        if not event.choices:
                            continue
                        delta = event.choices[0].delta.content
                        if delta:
                            print(delta, end="", flush=True)
                            chunks.append(delta)
                    full = "".join(chunks).strip()
                    if not full:
                        raise LLMError("Streaming response yielded empty text.")
                    return full

                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    stream=False,
                )
                if not resp.choices:
                    raise LLMError("Chat completion returned no choices.")
                content = resp.choices[0].message.content or ""
                return content.strip()

            except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = min(self.base_retry_delay * (2**attempt), 8.0)
                time.sleep(delay)
            except APIStatusError as exc:
                raise LLMError(f"API HTTP {exc.status_code} error: {exc.message}") from exc
            except Exception as exc:
                raise LLMError(f"Unexpected LLM call failure: {exc}") from exc

        raise LLMError(
            f"LLM request failed after {self.max_retries + 1} attempts ({self.base_url}): {last_exc}"
        ) from last_exc


# ---------------------------------------------------------------------------
# Deterministic Mock / Dry-Run Backend (`MockLLMClient`)
# ---------------------------------------------------------------------------

_MOCK_INIT_PY = '''"""Deterministic task priority & SLA analytics package generated by Meta-Dev."""

from task_forge.engine import TaskEngine
from task_forge.models import TaskRecord

__all__ = ["TaskEngine", "TaskRecord"]
'''

_MOCK_MODELS_PY = '''"""Typed domain model for prioritized tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskRecord:
    """Represents an actionable work item with numeric priority and completion state."""

    task_id: str
    title: str
    priority: int
    completed: bool = False

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id cannot be empty")
        if not (1 <= self.priority <= 5):
            raise ValueError("priority must be between 1 and 5")
'''

_MOCK_ENGINE_BUGGY_PY = '''"""Core scheduling and metrics engine (Round 1 build containing a deliberate sort-order defect)."""

from __future__ import annotations

from typing import Any

from task_forge.models import TaskRecord


class TaskEngine:
    """Manages task registration, priority ordering, and completion metrics."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}

    def add_task(self, task: TaskRecord) -> None:
        self._tasks[task.task_id] = task

    def complete_task(self, task_id: str) -> TaskRecord:
        if task_id not in self._tasks:
            raise KeyError(f"Unknown task_id: {task_id}")
        existing = self._tasks[task_id]
        updated = TaskRecord(
            task_id=existing.task_id,
            title=existing.title,
            priority=existing.priority,
            completed=True,
        )
        self._tasks[task_id] = updated
        return updated

    def pending_by_priority(self) -> list[TaskRecord]:
        pending = [t for t in self._tasks.values() if not t.completed]
        # DELIBERATE ROUND-1 BUG: sorts ascending instead of highest-priority (5 -> 1) first!
        return sorted(pending, key=lambda item: (item.priority, item.task_id))

    def summary(self) -> dict[str, Any]:
        total = len(self._tasks)
        done = sum(1 for t in self._tasks.values() if t.completed)
        rate = round((done / total) * 100.0, 2) if total else 0.0
        return {
            "total": total,
            "completed": done,
            "pending": total - done,
            "completion_rate": rate,
        }
'''

_MOCK_ENGINE_FIXED_PY = '''"""Core scheduling and metrics engine (Self-Healed in Round 2)."""

from __future__ import annotations

from typing import Any

from task_forge.models import TaskRecord


class TaskEngine:
    """Manages task registration, priority ordering, and completion metrics."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}

    def add_task(self, task: TaskRecord) -> None:
        self._tasks[task.task_id] = task

    def complete_task(self, task_id: str) -> TaskRecord:
        if task_id not in self._tasks:
            raise KeyError(f"Unknown task_id: {task_id}")
        existing = self._tasks[task_id]
        updated = TaskRecord(
            task_id=existing.task_id,
            title=existing.title,
            priority=existing.priority,
            completed=True,
        )
        self._tasks[task_id] = updated
        return updated

    def pending_by_priority(self) -> list[TaskRecord]:
        pending = [t for t in self._tasks.values() if not t.completed]
        # Fixed per QA B1: highest numeric priority (5 -> 1) first, deterministic task_id tie-breaker
        return sorted(pending, key=lambda item: (-item.priority, item.task_id))

    def summary(self) -> dict[str, Any]:
        total = len(self._tasks)
        done = sum(1 for t in self._tasks.values() if t.completed)
        rate = round((done / total) * 100.0, 2) if total else 0.0
        return {
            "total": total,
            "completed": done,
            "pending": total - done,
            "completion_rate": rate,
        }
'''

_MOCK_MAIN_PY = '''"""Non-interactive CLI entrypoint for task_forge."""

from __future__ import annotations

import json

from task_forge import TaskEngine, TaskRecord


def main() -> int:
    engine = TaskEngine()
    engine.add_task(TaskRecord(task_id="T-101", title="Draft spec", priority=3))
    engine.add_task(TaskRecord(task_id="T-102", title="Fix critical bug", priority=5))
    engine.add_task(TaskRecord(task_id="T-103", title="Write docs", priority=2))
    engine.complete_task("T-101")

    payload = {
        "next_up": [t.task_id for t in engine.pending_by_priority()],
        "metrics": engine.summary(),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_MOCK_TESTS_PY = '''"""Automated pytest verification suite covering R1-R4 and AC1-AC4."""

from __future__ import annotations

import pytest

from task_forge import TaskEngine, TaskRecord


def test_empty_engine_summary() -> None:
    engine = TaskEngine()
    assert engine.summary() == {
        "total": 0,
        "completed": 0,
        "pending": 0,
        "completion_rate": 0.0,
    }


def test_priority_ordering_highest_first_with_tie_breaker() -> None:
    engine = TaskEngine()
    engine.add_task(TaskRecord(task_id="T-2", title="Medium B", priority=3))
    engine.add_task(TaskRecord(task_id="T-9", title="Critical outage", priority=5))
    engine.add_task(TaskRecord(task_id="T-1", title="Medium A", priority=3))
    engine.add_task(TaskRecord(task_id="T-4", title="Low cleanup", priority=1))

    ordered_ids = [item.task_id for item in engine.pending_by_priority()]
    assert ordered_ids == ["T-9", "T-1", "T-2", "T-4"]


def test_complete_task_updates_metrics_and_excludes_from_pending() -> None:
    engine = TaskEngine()
    engine.add_task(TaskRecord(task_id="T-1", title="Alpha", priority=4))
    engine.add_task(TaskRecord(task_id="T-2", title="Beta", priority=2))
    engine.complete_task("T-1")

    assert [t.task_id for t in engine.pending_by_priority()] == ["T-2"]
    assert engine.summary()["completion_rate"] == 50.0


def test_validation_rejects_invalid_priority_and_unknown_task() -> None:
    with pytest.raises(ValueError):
        TaskRecord(task_id="T-bad", title="Invalid", priority=9)

    engine = TaskEngine()
    with pytest.raises(KeyError):
        engine.complete_task("MISSING")
'''


def _render_mock_project_bundle(engine_code: str) -> str:
    files = [
        ("task_forge/__init__.py", _MOCK_INIT_PY),
        ("task_forge/models.py", _MOCK_MODELS_PY),
        ("task_forge/engine.py", engine_code),
        ("main.py", _MOCK_MAIN_PY),
        ("tests/test_task_forge.py", _MOCK_TESTS_PY),
    ]
    blocks = [
        f"File: {path}\n```python\n{content.strip()}\n```"
        for path, content in files
    ]
    return "\n\n".join(blocks)


class MockLLMClient(BaseLLMClient):
    """Deterministic zero-API-key backend simulating the full 5-stage PM->DEV->QA->Self-Heal->Release cycle."""

    def __init__(
        self,
        *,
        model: str = "mock-deterministic-v1",
        simulate_first_round_bug: bool = True,
        simulate_unfixable_bug: bool = False,
        scripted_responses: list[str] | None = None,
    ) -> None:
        self.model = model
        self.simulate_first_round_bug = simulate_first_round_bug
        self.simulate_unfixable_bug = simulate_unfixable_bug
        self._scripted = list(scripted_responses) if scripted_responses is not None else None
        self.call_log: list[dict[str, Any]] = []
        self._dev_build_count = 0

    def list_models(self) -> list[str]:
        return [self.model]

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        *,
        role: str = "",
        stage: str = "",
        temperature: float = 0.2,
        stream: bool = False,
    ) -> str:
        inferred_role = (role or self._infer_role(system_prompt)).upper()
        inferred_stage = stage or self._infer_stage(user_message)
        self.call_log.append(
            {
                "role": inferred_role,
                "stage": inferred_stage,
                "user_message": user_message,
            }
        )

        if self._scripted is not None:
            if not self._scripted:
                raise LLMError("MockLLMClient exhausted scripted_responses.")
            reply = self._scripted.pop(0)
            if stream:
                print(reply, end="", flush=True)
            return reply

        reply = self._dispatch_deterministic_turn(inferred_role, inferred_stage, user_message)
        if stream:
            print(reply, end="", flush=True)
        return reply

    @staticmethod
    def _infer_role(system_prompt: str) -> str:
        if "PM-Agent" in system_prompt:
            return "PM"
        if "QA-Agent" in system_prompt:
            return "QA"
        if "Judge-Agent" in system_prompt:
            return "JUDGE"
        if "Arbiter Agent" in system_prompt:
            return "ARBITER"
        return "DEV"

    @staticmethod
    def _infer_stage(user_message: str) -> str:
        for st in ("Clarify", "Design", "Build", "Verify", "Release"):
            if f"CURRENT_STAGE: {st}" in user_message:
                return st
        return "Build"

    def _dispatch_deterministic_turn(self, role: str, stage: str, user_message: str) -> str:
        # Commit message generation shortcut
        if role == "PM" and ("Git commit" in user_message or "提交说明" in user_message):
            return "feat: deliver verified multi-file Python package via Meta-Dev pipeline"

        if role == "PM" and stage == "Clarify":
            return (
                "ROLE: PM\n"
                "STAGE: Clarify\n\n"
                "CURRENT_POSITION:\n"
                "- 已将用户原始需求收敛为可实现、可自动化验证的多文件模块化方案（R1-R4，AC1-AC4）。\n"
                "- 明确采用非交互式 CLI 与 pytest 自动化测试套件（A1），杜绝阻塞型终端等待。\n\n"
                "QUESTIONS:\n"
                "- TO_PM: None\n"
                "- TO_DEV: 请确认模块拆分满足零外部重依赖约束。\n"
                "- TO_QA: 请确认 AC1-AC4 可直接映射为确定性单元测试。\n\n"
                "OBJECTIONS:\n"
                "None\n\n"
                "RESPONSES_TO_OBJECTIONS:\n"
                "None\n\n"
                "PROPOSAL:\n"
                "- Requirements:\n"
                "  - R1: 提供类型安全的任务实体与输入校验（优先级 1~5，非空 ID）。\n"
                "  - R2: 支持按优先级降序（5 -> 1）与确定性字典序返回待办队列。\n"
                "  - R3: 支持任务完成状态流转及完成率百分比统计（空队列安全返回 0.0）。\n"
                "  - R4: 提供非交互式 CLI 入口 `main.py` 与完整 `tests/` 单元测试。\n"
                "- Acceptance Criteria:\n"
                "  - AC1: 空引擎统计无除零异常且 completion_rate == 0.0。\n"
                "  - AC2: pending_by_priority 严格按 priority 降序、task_id 升序排列。\n"
                "  - AC3: 非法优先级或未知任务 ID 抛出明确异常（ValueError / KeyError）。\n"
                "  - AC4: pytest 全量测试退出码为 0。\n\n"
                "AGREEMENTS:\n"
                "- MVP 聚焦核心模块化引擎、CLI 演示入口与全覆盖单元测试。\n\n"
                "OPEN_ISSUES:\n"
                "None\n\n"
                "CONSENSUS_STATUS:\n"
                "- Reached: 需求编号、边界假设与可测验收标准已完整对齐。"
            )

        if role == "DEV" and stage == "Design":
            return (
                "ROLE: DEV\n"
                "STAGE: Design\n\n"
                "CURRENT_POSITION:\n"
                "- 基于 R1-R4 与 AC1-AC4，设计了三层解耦的多文件包结构 `task_forge/` + `main.py` + `tests/`。\n"
                "- 全程零阻塞 `input()`，所有核心逻辑具备纯函数式可测接口。\n\n"
                "QUESTIONS:\n"
                "- TO_PM: None\n"
                "- TO_DEV: None\n"
                "- TO_QA: None\n\n"
                "OBJECTIONS:\n"
                "None\n\n"
                "RESPONSES_TO_OBJECTIONS:\n"
                "None\n\n"
                "PROPOSAL:\n"
                "- Architecture Plan:\n"
                "  - `task_forge/models.py`: `TaskRecord` frozen dataclass + 边界校验（R1, AC3）。\n"
                "  - `task_forge/engine.py`: `TaskEngine` 状态管理、优先级排序与完成率计算（R2, R3）。\n"
                "  - `main.py`: 非交互 JSON 输出入口（R4）。\n"
                "  - `tests/test_task_forge.py`: 覆盖 AC1-AC4 的 pytest 用例集。\n\n"
                "AGREEMENTS:\n"
                "- 同意全部验收标准 AC1-AC4。\n\n"
                "OPEN_ISSUES:\n"
                "None\n\n"
                "CONSENSUS_STATUS:\n"
                "- Reached: 模块职责、数据流与测试切面清晰可落地。"
            )

        if role == "QA" and stage == "Design":
            return (
                "ROLE: QA\n"
                "STAGE: Design\n\n"
                "CURRENT_POSITION:\n"
                "- DEV 的多文件架构具备良好可观测性与纯内存可测性，映射测试用例 TC1-TC4。\n\n"
                "QUESTIONS:\n"
                "- TO_PM: None\n"
                "- TO_DEV: None\n"
                "- TO_QA: None\n\n"
                "OBJECTIONS:\n"
                "None\n\n"
                "RESPONSES_TO_OBJECTIONS:\n"
                "None\n\n"
                "PROPOSAL:\n"
                "- Requirement-to-Test Mapping:\n"
                "  - R1/AC3 -> TC1: 非法 priority 与未知 task_id 异常断言。\n"
                "  - R2/AC2 -> TC2: 多优先级与同优先级 task_id 稳定排序断言。\n"
                "  - R3/AC1 -> TC3: 空集合及完成后 completion_rate 精度断言。\n"
                "  - R4/AC4 -> TC4: 沙盒自动运行 pytest 零错误退出。\n\n"
                "AGREEMENTS:\n"
                "- 测试矩阵与 DEV 模块计划达成一致。\n\n"
                "OPEN_ISSUES:\n"
                "None\n\n"
                "CONSENSUS_STATUS:\n"
                "- Reached: 验证策略完备，可进入 Build 阶段。"
            )

        if role == "DEV" and stage in {"Build", "Verify"}:
            self._dev_build_count += 1
            use_buggy = self.simulate_unfixable_bug or (
                self.simulate_first_round_bug and self._dev_build_count == 1
            )
            engine_src = _MOCK_ENGINE_BUGGY_PY if use_buggy else _MOCK_ENGINE_FIXED_PY
            bundle = _render_mock_project_bundle(engine_src)
            note = (
                "已完成首轮多文件代码与单元测试生成（T1-T4）。"
                if self._dev_build_count == 1
                else "已根据 QA Error-Reflection 报告修复 B1（将 pending_by_priority 修正为 `-item.priority` 降序排列）。"
            )
            resp_obj = (
                "None"
                if self._dev_build_count == 1
                else "- [B1] Accepted\n  Reason: 确认排序方向写反导致 AC2 断言失败。\n  Update: 已修正排序键为 `(-item.priority, item.task_id)`。"
            )
            return (
                f"ROLE: DEV\n"
                f"STAGE: {stage}\n\n"
                f"CURRENT_POSITION:\n"
                f"- {note}\n\n"
                f"QUESTIONS:\n"
                f"None\n\n"
                f"OBJECTIONS:\n"
                f"None\n\n"
                f"RESPONSES_TO_OBJECTIONS:\n"
                f"{resp_obj}\n\n"
                f"PROPOSAL:\n"
                f"{bundle}\n\n"
                f"AGREEMENTS:\n"
                f"- 全部模块与自动化测试已按规范多文件结构落盘。\n\n"
                f"OPEN_ISSUES:\n"
                f"None\n\n"
                f"CONSENSUS_STATUS:\n"
                f"- Reached: 代码产物已就绪，移交本地沙盒与 QA 验证。"
            )

        if role == "QA" and stage == "Verify":
            sandbox_failed = (
                "STATUS: FAILED" in user_message
                or "STATUS: TIMED_OUT" in user_message
                or "[EXIT_CODE] 1" in user_message
                or "AssertionError" in user_message
            )
            if sandbox_failed:
                return (
                    "ROLE: QA\n"
                    "STAGE: Verify\n\n"
                    "CURRENT_POSITION:\n"
                    "- 本地沙盒执行 `pytest` 捕获真实断言失败，违反 AC2（高优先级任务未排在队首），触发阻断级缺陷 B1。\n\n"
                    "QUESTIONS:\n"
                    "None\n\n"
                    "OBJECTIONS:\n"
                    "- [DEV][B1][质量][Blocker]\n"
                    "  Reason: `TaskEngine.pending_by_priority` 使用了升序 `item.priority`，导致低优先级排在前面。\n"
                    "  Risk: 核心调度顺序完全反转，直接违反 R2 与 AC2。\n"
                    "  Counterproposal: 将排序 key 修改为 `(-item.priority, item.task_id)`。\n"
                    "  Accept_Condition: `pytest` 全量测试通过且退出码为 0。\n\n"
                    "RESPONSES_TO_OBJECTIONS:\n"
                    "None\n\n"
                    "PROPOSAL:\n"
                    "BUG_ID: B1\n"
                    "TITLE: pending_by_priority sorts priority ascending instead of descending\n"
                    "SEVERITY: Blocker\n"
                    "REFERENCE: R2 / AC2 / task_forge/engine.py\n"
                    "SUGGESTED_DIRECTION: Update sort key to `(-item.priority, item.task_id)` and re-run pytest.\n\n"
                    "AGREEMENTS:\n"
                    "- 其余边界测试（AC1, AC3）已通过。\n\n"
                    "OPEN_ISSUES:\n"
                    "- B1 (Blocker): 优先级排序反转导致单元测试失败。\n\n"
                    "CONSENSUS_STATUS:\n"
                    "- Not Ready: 存在未修复的 Blocker 缺陷 B1，禁止放行 Release。"
                )

            return (
                "ROLE: QA\n"
                "STAGE: Verify\n\n"
                "CURRENT_POSITION:\n"
                "- 本地沙盒 `pytest` 全部测试用例通过（exit_code=0），缺陷 B1 已闭环验证，AC1-AC4 全部满足。\n\n"
                "QUESTIONS:\n"
                "None\n\n"
                "OBJECTIONS:\n"
                "None\n\n"
                "RESPONSES_TO_OBJECTIONS:\n"
                "None\n\n"
                "PROPOSAL:\n"
                "- Go / No-Go Recommendation: GO — 所有关键路径、异常边界与回归测试均 100% 通过。\n\n"
                "AGREEMENTS:\n"
                "- R1-R4 与 AC1-AC4 全部验证通过。\n\n"
                "OPEN_ISSUES:\n"
                "None\n\n"
                "CONSENSUS_STATUS:\n"
                "- Reached: 质量门禁全部绿灯，准予进入 Release 阶段。"
            )

        if role == "PM" and stage == "Release":
            return (
                "ROLE: PM\n"
                "STAGE: Release\n\n"
                "CURRENT_POSITION:\n"
                "- 需求 R1-R4 已完整交付并通过沙盒自动化测试与 QA 质量门禁，裁决结论为 GO。\n\n"
                "QUESTIONS:\n"
                "None\n\n"
                "OBJECTIONS:\n"
                "None\n\n"
                "RESPONSES_TO_OBJECTIONS:\n"
                "None\n\n"
                "PROPOSAL:\n"
                "- Release Verdict: GO\n"
                "- Commit Message: feat: deliver verified task_forge package via Meta-Dev pipeline\n\n"
                "AGREEMENTS:\n"
                "- PM、DEV、QA 三方达成最终发布共识。\n\n"
                "OPEN_ISSUES:\n"
                "None\n\n"
                "CONSENSUS_STATUS:\n"
                "- Reached: 满足全部上线条件。"
            )

        return (
            f"ROLE: {role}\n"
            f"STAGE: {stage}\n\n"
            "CURRENT_POSITION:\n- Deterministic mock response.\n\n"
            "OBJECTIONS:\nNone\n\n"
            "CONSENSUS_STATUS:\n- Reached: Completed."
        )


# ---------------------------------------------------------------------------
# Backwards-compatible functional helpers
# ---------------------------------------------------------------------------


def list_available_models(
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> list[str]:
    """List accessible model IDs on the configured OpenAI-compatible gateway."""
    client = OpenAICompatibleLLMClient(api_key=api_key, base_url=base_url)
    return client.list_models()


def chat_with_agent(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    stream: bool = False,
) -> str:
    """Invoke an OpenAI-compatible chat model with system + user prompt."""
    client = OpenAICompatibleLLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    return client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        stream=stream,
    )
