"""Five-stage multi-agent Software Factory orchestrator with Error-Reflection self-healing."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from meta_dev.arbiter import resolve_dev_qa_conflict, verify_quality_gate
from meta_dev.git_ops import auto_git_commit, sanitize_commit_message, save_run_manifest
from meta_dev.llm import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    BaseLLMClient,
    MockLLMClient,
    chat_with_agent,
)
from meta_dev.models import (
    ArbiterDecision,
    DebateRound,
    HealIterationRecord,
    PipelineRunSummary,
    SandboxReport,
    Stage,
    TurnEnvelope,
)
from meta_dev.parser import (
    extract_and_save_generated_artifacts,
    format_saved_files_bundle,
    parse_turn_envelope,
)
from meta_dev.prompts import (
    DEV_SYSTEM_PROMPT,
    PM_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    DynamicPromptContext,
    assemble_stage_user_prompt,
)
from meta_dev.sandbox import (
    execute_workspace_in_sandbox,
    format_error_reflection_feedback,
    run_result_indicates_success,
    save_and_execute_generated_code,
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def looks_like_code_generation_response(text: str) -> bool:
    """Return True if an LLM response contains extractable code blocks or Python definitions."""
    if not text or not text.strip():
        return False
    code_indicators = (
        "```python",
        "```",
        "def ",
        "class ",
        "import ",
        "if __name__",
    )
    return any(indicator in text for indicator in code_indicators)


def request_implies_code_pipeline(user_request: str, role: str) -> bool:
    """Heuristic detecting whether a direct `--role dev` message asks for automated execution."""
    if role.lower() != "dev":
        return False

    text = (user_request or "").lower()
    if not text:
        return False

    explicit_pipeline_keywords = (
        "自动测试",
        "直接运行",
        "立即运行",
        "现在开始",
        "在终端里玩",
        "开始游戏",
        "auto test",
        "run it now",
        "run directly",
    )
    if any(keyword in text for keyword in explicit_pipeline_keywords):
        return True

    code_action_keywords = (
        "帮我写",
        "请写",
        "写一个",
        "写一段",
        "生成一个",
        "生成一段",
        "实现一个",
        "实现一段",
        "create",
        "write",
        "implement",
        "build",
    )
    code_artifact_keywords = (
        "python 脚本",
        "python脚本",
        "python 代码",
        "python代码",
        "脚本",
        "代码",
        "函数",
        "小游戏",
        "package",
        "project",
        ".py",
    )
    return any(keyword in text for keyword in code_action_keywords) and any(
        keyword in text for keyword in code_artifact_keywords
    )


def generate_commit_message_with_pm(
    original_request: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    llm_client: BaseLLMClient | None = None,
) -> str:
    """Request a concise single-line Git commit message from PM-Agent."""
    pm_prompt = (
        "开发已结束，当前代码已经全部通过自动化测试。\n"
        f"原始开发需求如下：\n{original_request}\n\n"
        "请生成一条简短规范的 Git commit 提交说明（单行，不超过 72 字符），"
        "不要输出 ROLE、STAGE、项目符号、引号、代码块或解释。"
    )
    if llm_client is not None:
        raw_message = llm_client.chat(
            PM_SYSTEM_PROMPT,
            pm_prompt,
            role="PM",
            stage=Stage.RELEASE.value,
            stream=False,
        )
    else:
        raw_message = chat_with_agent(
            system_prompt=PM_SYSTEM_PROMPT,
            user_message=pm_prompt,
            model=model,
            base_url=base_url,
            api_key=api_key,
            stream=False,
        )
    return sanitize_commit_message(raw_message)


def request_dev_code_retry(
    original_request: str,
    first_response: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    stream: bool = False,
    llm_client: BaseLLMClient | None = None,
) -> str:
    """Prompt DEV-Agent to advance from abstract analysis to concrete executable Python files."""
    retry_prompt = (
        "你刚才停留在澄清/方案阶段，但当前流水线需要推进到可执行代码与单元测试产物。\n"
        f"原始用户需求如下：\n{original_request}\n\n"
        f"你刚才的分析如下：\n{first_response}\n\n"
        "请严格遵守以下要求：\n"
        "1. 若信息不足，请写出最小假设并直接给出完整实现。\n"
        "2. 多文件项目请严格使用 `File: <relative/path.py>` 加 ```python 代码块格式。\n"
        "3. 禁止使用阻塞型 `input()`，并附带 `tests/` 自动化测试。"
    )
    if llm_client is not None:
        return llm_client.chat(
            DEV_SYSTEM_PROMPT,
            retry_prompt,
            role="DEV",
            stage=Stage.BUILD.value,
            stream=stream,
        )
    return chat_with_agent(
        system_prompt=DEV_SYSTEM_PROMPT,
        user_message=retry_prompt,
        model=model,
        base_url=base_url,
        api_key=api_key,
        stream=stream,
    )


def request_dev_single_file_retry(
    original_request: str,
    previous_response: str,
    extraction_error: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    stream: bool = False,
    llm_client: BaseLLMClient | None = None,
) -> str:
    """Prompt DEV-Agent to reformat unextractable output into clean `File: <path>` code blocks."""
    retry_prompt = (
        "你上一次的输出无法被系统自动提取成可执行文件。\n"
        f"原始用户需求如下：\n{original_request}\n\n"
        f"提取失败原因：{extraction_error}\n\n"
        f"上一次输出摘要：\n{previous_response[:2000]}\n\n"
        "请重写输出：使用 `File: main.py` + ```python 代码块（以及 `File: tests/test_main.py`）格式。"
    )
    if llm_client is not None:
        return llm_client.chat(
            DEV_SYSTEM_PROMPT,
            retry_prompt,
            role="DEV",
            stage=Stage.BUILD.value,
            stream=stream,
        )
    return chat_with_agent(
        system_prompt=DEV_SYSTEM_PROMPT,
        user_message=retry_prompt,
        model=model,
        base_url=base_url,
        api_key=api_key,
        stream=stream,
    )


class SoftwareFactoryOrchestrator:
    """Coordinates Clarify -> Design -> Build -> Verify (Self-Heal) -> Release in an isolated workspace."""

    def __init__(
        self,
        *,
        llm_client: BaseLLMClient,
        workspace_dir: str | Path,
        max_heal_rounds: int = 3,
        sandbox_timeout: float = 15.0,
        auto_git: bool = False,
        non_interactive: bool = True,
        verbose: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.workspace_root = Path(workspace_dir).resolve()
        self.max_heal_rounds = max(0, max_heal_rounds)
        self.sandbox_timeout = max(0.5, float(sandbox_timeout))
        self.auto_git = auto_git
        self.non_interactive = non_interactive
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def run(self, requirement: str) -> PipelineRunSummary:
        if not isinstance(requirement, str) or not requirement.strip():
            raise ValueError("requirement cannot be empty.")

        self.workspace_root.mkdir(parents=True, exist_ok=True)
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        started_at = _utc_iso()
        is_dry_run = isinstance(self.llm_client, MockLLMClient)

        stages_completed: list[str] = []
        decision_log: list[str] = []
        open_issues: list[str] = []
        arbiter_decisions: list[ArbiterDecision] = []
        heal_history: list[HealIterationRecord] = []

        # ------------------------------------------------------------------
        # Stage 1: Clarify (PM)
        # ------------------------------------------------------------------
        self._log(f"\n[1/5 Clarify] PM-Agent clarifying requirement in {self.workspace_root} ...")
        clarify_prompt = assemble_stage_user_prompt(
            DynamicPromptContext(
                project_brief=requirement,
                current_stage=Stage.CLARIFY,
                extra_instructions=(
                    "请输出结构化产品定义（编号 R1..Rn、验收标准 AC1..ACn、显式假设 A1..An 及 MVP 边界），"
                    "确保所有关键行为可通过自动化测试在非交互沙盒中验证。"
                ),
            )
        )
        pm_clarify_raw = self.llm_client.chat(
            PM_SYSTEM_PROMPT,
            clarify_prompt,
            role="PM",
            stage=Stage.CLARIFY.value,
        )
        pm_clarify_env = parse_turn_envelope(
            pm_clarify_raw,
            default_role="PM",
            default_stage=Stage.CLARIFY.value,
            workspace_dir=self.workspace_root,
        )
        decision_log.append(
            f"D1 (Clarify): PM finalized requirements and acceptance criteria "
            f"(consensus={pm_clarify_env.consensus_status})."
        )
        if pm_clarify_env.open_issues:
            open_issues.extend(pm_clarify_env.open_issues)
        stages_completed.append(Stage.CLARIFY.value)

        # ------------------------------------------------------------------
        # Stage 2: Design (DEV + QA)
        # ------------------------------------------------------------------
        self._log("[2/5 Design] DEV-Agent & QA-Agent aligning modular architecture & test matrix ...")
        dev_design_prompt = assemble_stage_user_prompt(
            DynamicPromptContext(
                project_brief=requirement,
                current_stage=Stage.DESIGN,
                decision_log=decision_log,
                latest_messages=[pm_clarify_raw],
                extra_instructions=(
                    "请基于 PM 的需求与验收标准，给出清晰的模块化多文件架构设计（包含核心包、入口脚本与 `tests/` 单元测试规划）。"
                ),
            )
        )
        dev_design_raw = self.llm_client.chat(
            DEV_SYSTEM_PROMPT,
            dev_design_prompt,
            role="DEV",
            stage=Stage.DESIGN.value,
        )
        dev_design_env = parse_turn_envelope(
            dev_design_raw,
            default_role="DEV",
            default_stage=Stage.DESIGN.value,
            workspace_dir=self.workspace_root,
        )

        qa_design_prompt = assemble_stage_user_prompt(
            DynamicPromptContext(
                project_brief=requirement,
                current_stage=Stage.DESIGN,
                decision_log=decision_log,
                latest_messages=[pm_clarify_raw, dev_design_raw],
                extra_instructions=(
                    "请评审 DEV 的架构可测性，输出 Requirement-to-Test 映射（R# -> TC#）与关键边界/异常测试点。"
                ),
            )
        )
        qa_design_raw = self.llm_client.chat(
            QA_SYSTEM_PROMPT,
            qa_design_prompt,
            role="QA",
            stage=Stage.DESIGN.value,
        )
        qa_design_env = parse_turn_envelope(
            qa_design_raw,
            default_role="QA",
            default_stage=Stage.DESIGN.value,
            workspace_dir=self.workspace_root,
        )

        if qa_design_env.blocker_objections:
            arb_decision = resolve_dev_qa_conflict(
                issue_id="I_DESIGN_1",
                stage=Stage.DESIGN.value,
                dev_envelope=dev_design_env,
                qa_envelope=qa_design_env,
            )
            arbiter_decisions.append(arb_decision)
            decision_log.append(
                f"D_ARB_DESIGN: Arbiter resolved design conflict -> {arb_decision.decision}"
            )

        decision_log.append("D2 (Design): Architecture and QA test matrix aligned.")
        stages_completed.append(Stage.DESIGN.value)

        # ------------------------------------------------------------------
        # Stage 3: Build (DEV Multi-File Code + Unit Tests Generation)
        # ------------------------------------------------------------------
        self._log("[3/5 Build] DEV-Agent generating multi-file project and automated unit tests ...")
        build_prompt = assemble_stage_user_prompt(
            DynamicPromptContext(
                project_brief=requirement,
                current_stage=Stage.BUILD,
                decision_log=decision_log,
                latest_messages=[pm_clarify_raw, dev_design_raw, qa_design_raw],
                extra_instructions=(
                    "现在进入 Build 阶段，请输出完整可运行的多文件 Python 项目代码与 `tests/` 自动化测试。\n"
                    "硬性格式约束：\n"
                    "1. 每个文件必须使用 `File: <相对路径>` 独占一行，紧跟 ```python ... ``` 代码块。\n"
                    "2. 严禁使用任何 `../` 或绝对路径。\n"
                    "3. 严禁调用阻塞型内置 `input()`（沙盒会将 stdin 重定向为 DEVNULL 并执行 AST 静态拦截）。\n"
                    "4. 必须包含覆盖核心验收标准与异常边界的 `tests/test_*.py` 测试文件。"
                ),
            )
        )
        dev_build_raw = self.llm_client.chat(
            DEV_SYSTEM_PROMPT,
            build_prompt,
            role="DEV",
            stage=Stage.BUILD.value,
        )
        if not looks_like_code_generation_response(dev_build_raw):
            dev_build_raw = request_dev_code_retry(
                original_request=requirement,
                first_response=dev_build_raw,
                llm_client=self.llm_client,
            )

        artifact_info = extract_and_save_generated_artifacts(
            dev_build_raw,
            filename="main.py",
            workspace_dir=self.workspace_root,
        )
        current_files: dict[str, str] = dict(artifact_info["files"])
        dev_last_env = parse_turn_envelope(
            dev_build_raw,
            default_role="DEV",
            default_stage=Stage.BUILD.value,
            workspace_dir=self.workspace_root,
        )
        stages_completed.append(Stage.BUILD.value)
        self._log(f"    Saved {len(current_files)} file(s): {', '.join(sorted(current_files))}")

        # ------------------------------------------------------------------
        # Stage 4: Verify (Sandbox + QA Error-Reflection Self-Healing Loop)
        # ------------------------------------------------------------------
        self._log("[4/5 Verify] Running local AST-guarded sandbox & QA verification ...")
        heal_rounds_used = 0
        final_sandbox_report: SandboxReport | None = None
        qa_verify_env: TurnEnvelope | None = None
        verify_debate_rounds: list[DebateRound] = []

        for attempt in range(self.max_heal_rounds + 1):
            sandbox_report = execute_workspace_in_sandbox(
                self.workspace_root,
                saved_files=current_files,
                default_entrypoint="main.py",
                timeout=self.sandbox_timeout,
                block_on_ast_input=self.non_interactive,
            )
            final_sandbox_report = sandbox_report
            diagnostics = format_error_reflection_feedback(sandbox_report)

            qa_verify_prompt = assemble_stage_user_prompt(
                DynamicPromptContext(
                    project_brief=requirement,
                    current_stage=Stage.VERIFY,
                    repo_context=format_saved_files_bundle(current_files),
                    decision_log=decision_log,
                    open_issues=open_issues,
                    latest_messages=[pm_clarify_raw, diagnostics],
                    extra_instructions=(
                        "请结合真实沙盒运行诊断（SANDBOX ERROR-REFLECTION DIAGNOSTICS）与源码进行质量审查。\n"
                        "- 若沙盒测试失败或超时，你必须在 OBJECTIONS 中提出 Blocker 级异议并给出精确根因与修复方向，CONSENSUS_STATUS 设为 Not Ready。\n"
                        "- 若沙盒测试全部通过（STATUS: PASSED, exit_code=0）且满足全部关键验收标准，请将 OBJECTIONS 设为 None，CONSENSUS_STATUS 设为 Reached。"
                    ),
                )
            )
            qa_verify_raw = self.llm_client.chat(
                QA_SYSTEM_PROMPT,
                qa_verify_prompt,
                role="QA",
                stage=Stage.VERIFY.value,
            )
            qa_verify_env = parse_turn_envelope(
                qa_verify_raw,
                default_role="QA",
                default_stage=Stage.VERIFY.value,
                workspace_dir=self.workspace_root,
            )

            gate_check = verify_quality_gate(
                sandbox_report,
                qa_verify_env,
                stage=Stage.VERIFY.value,
            )
            qa_headline = (
                qa_verify_env.current_position[0]
                if qa_verify_env.current_position
                else qa_verify_env.consensus_status
            )
            heal_history.append(
                HealIterationRecord(
                    round_index=attempt,
                    sandbox_passed=sandbox_report.passed,
                    exit_code=sandbox_report.exit_code,
                    timed_out=sandbox_report.timed_out,
                    qa_summary=qa_headline,
                    files_updated=sorted(current_files.keys()),
                )
            )

            if gate_check.passed:
                self._log(f"    Verify round {attempt}: PASSED (exit_code=0, zero blockers).")
                open_issues.clear()
                break

            open_issues = list(gate_check.blocking_issues)
            self._log(
                f"    Verify round {attempt}: FAILED ({'; '.join(gate_check.blocking_issues)})"
            )

            if attempt >= self.max_heal_rounds:
                self._log("    Max self-healing rounds reached; halting further repair attempts.")
                break

            # Track DEV/QA debate state and invoke Arbiter if deadlocked
            verify_debate_rounds.append(
                DebateRound(
                    dev_position="Implementation ready",
                    qa_position="Blocked by verification failure",
                    dev_novelty=0.10 if attempt >= 1 else 0.40,
                    qa_novelty=0.10 if attempt >= 1 else 0.40,
                    resolved_objections=0,
                    active_objections=max(1, len(qa_verify_env.objections)),
                    consensus_status="Not Ready",
                )
            )
            arbiter_directive = ""
            if len(verify_debate_rounds) >= 2:
                arb_decision = resolve_dev_qa_conflict(
                    issue_id=f"I_VERIFY_{attempt}",
                    stage=Stage.VERIFY.value,
                    dev_envelope=dev_last_env,
                    qa_envelope=qa_verify_env,
                    sandbox_report=sandbox_report,
                    history_rounds=verify_debate_rounds,
                )
                arbiter_decisions.append(arb_decision)
                arbiter_directive = (
                    f"\n[ARBITER MANDATE] Decision={arb_decision.decision}; "
                    f"Guardrails={'; '.join(arb_decision.guardrails) or 'Fix failing tests immediately.'}"
                )

            heal_rounds_used += 1
            self._log(
                f"    [Self-Heal {heal_rounds_used}/{self.max_heal_rounds}] DEV-Agent repairing defects from Error-Reflection ..."
            )
            dev_heal_prompt = assemble_stage_user_prompt(
                DynamicPromptContext(
                    project_brief=requirement,
                    current_stage=Stage.VERIFY,
                    repo_context=format_saved_files_bundle(current_files),
                    decision_log=decision_log,
                    open_issues=open_issues,
                    latest_messages=[diagnostics, qa_verify_raw],
                    extra_instructions=(
                        "请根据沙盒真实报错日志与 QA 的缺陷分析完成代码自愈修复。\n"
                        "输出修复后的完整文件（严格使用 `File: <相对路径>` + ```python 代码块格式），"
                        "确保所有单元测试通过且无阻塞调用。" + arbiter_directive
                    ),
                )
            )
            dev_heal_raw = self.llm_client.chat(
                DEV_SYSTEM_PROMPT,
                dev_heal_prompt,
                role="DEV",
                stage=Stage.VERIFY.value,
            )
            dev_last_env = parse_turn_envelope(
                dev_heal_raw,
                default_role="DEV",
                default_stage=Stage.VERIFY.value,
                workspace_dir=self.workspace_root,
            )
            healed_artifacts = extract_and_save_generated_artifacts(
                dev_heal_raw,
                filename="main.py",
                workspace_dir=self.workspace_root,
            )
            current_files.update(healed_artifacts["files"])

        stages_completed.append(Stage.VERIFY.value)

        # ------------------------------------------------------------------
        # Stage 5: Release & Quality Gate Enforcement
        # ------------------------------------------------------------------
        self._log("[5/5 Release] Evaluating final Quality Gate & preparing release manifest ...")
        final_gate = verify_quality_gate(
            final_sandbox_report,
            qa_verify_env,
            stage=Stage.RELEASE.value,
        )

        release_decision = "GO" if final_gate.passed else "NO-GO"
        commit_message: str | None = None
        commit_hash: str | None = None

        if final_gate.passed:
            pm_release_prompt = assemble_stage_user_prompt(
                DynamicPromptContext(
                    project_brief=requirement,
                    current_stage=Stage.RELEASE,
                    decision_log=decision_log,
                    latest_messages=[
                        qa_verify_env.raw_text if qa_verify_env else "",
                    ],
                    extra_instructions=(
                        "代码与自动化测试已全部通过质量门禁。请输出 Release 阶段 GO 结论与简洁规范的提交摘要。"
                    ),
                )
            )
            self.llm_client.chat(
                PM_SYSTEM_PROMPT,
                pm_release_prompt,
                role="PM",
                stage=Stage.RELEASE.value,
            )
            commit_message = generate_commit_message_with_pm(
                original_request=requirement,
                llm_client=self.llm_client,
            )
            if self.auto_git:
                commit_hash = auto_git_commit(
                    commit_message,
                    workspace_dir=self.workspace_root,
                )
                self._log(f"    Auto-Git committed in workspace: {commit_hash} ({commit_message})")

        stages_completed.append(Stage.RELEASE.value)
        completed_at = _utc_iso()

        summary = PipelineRunSummary(
            run_id=run_id,
            requirement=requirement,
            workspace=str(self.workspace_root),
            model=self.llm_client.model,
            dry_run=is_dry_run,
            stages_completed=stages_completed,
            heal_rounds_used=heal_rounds_used,
            max_heal_rounds=self.max_heal_rounds,
            saved_files=sorted(current_files.keys()),
            final_sandbox_report=final_sandbox_report,
            quality_gate=final_gate,
            arbiter_decisions=arbiter_decisions,
            heal_history=heal_history,
            release_decision=release_decision,
            commit_hash=commit_hash,
            commit_message=commit_message,
            started_at=started_at,
            completed_at=completed_at,
        )
        save_run_manifest(summary, self.workspace_root)
        self._log(
            f"[Done] Verdict={release_decision} | HealRounds={heal_rounds_used} | Manifest={summary.manifest_path}"
        )
        return summary


def run_boss_review_loop(
    original_request: str,
    generated_code: str,
    run_result: str,
    generated_filename: str = "auto_generated.py",
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    timeout: int = 10,
    workspace_dir: str | Path | None = None,
) -> None:
    """Interactive human-in-the-loop review -> QA analysis -> DEV fix -> auto-git archive."""
    current_code = generated_code
    current_run_result = run_result

    while True:
        try:
            feedback = input(
                "\n老板，请输入测试反馈 "
                "(输入 'pass' 代表完美通过，或输入具体的报错/修改建议让 QA 和 DEV 继续加班): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[系统] 审核环节已结束。")
            return

        if not feedback:
            continue

        if feedback.lower() == "pass":
            if not run_result_indicates_success(current_run_result):
                print("\n[系统] 最近一次自动运行结果仍然显示失败或超时，质量门禁拒绝直接放行存档。")
                continue

            commit_msg = generate_commit_message_with_pm(
                original_request=original_request,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
            auto_git_commit(commit_msg, workspace_dir=workspace_dir)
            print(f"\n🎉 已通过质量门禁并完成 Git 存档：{commit_msg}")
            return

        qa_prompt = (
            f"原始开发需求如下:\n{original_request}\n\n"
            f"当前项目/代码如下:\n{current_code}\n\n"
            f"真实沙盒结果/报错如下:\n{current_run_result}\n\n"
            f"最新测试反馈如下:\n{feedback}\n\n"
            "请分析问题根因并给出具体可执行的修改指导。"
        )
        test_report = chat_with_agent(
            system_prompt=QA_SYSTEM_PROMPT,
            user_message=qa_prompt,
            model=model,
            base_url=base_url,
            api_key=api_key,
            stream=False,
        )

        dev_fix_prompt = (
            f"原始开发需求如下:\n{original_request}\n\n"
            f"当前项目/代码如下:\n{current_code}\n\n"
            f"当前运行结果如下:\n{current_run_result}\n\n"
            f"QA 给出的修改指导如下:\n{test_report}\n\n"
            f"最新测试反馈如下:\n{feedback}\n\n"
            "请直接输出修复后的完整代码（多文件请使用 `File: <path>` + 代码块格式）。"
        )
        revised_response = chat_with_agent(
            system_prompt=DEV_SYSTEM_PROMPT,
            user_message=dev_fix_prompt,
            model=model,
            base_url=base_url,
            api_key=api_key,
            stream=False,
        )

        execution_result = save_and_execute_generated_code(
            generated_code_text=revised_response,
            user_request=original_request,
            filename=generated_filename,
            timeout=timeout,
            workspace_dir=workspace_dir,
        )
        if execution_result["mode"] == "interactive":
            return
        current_code = execution_result["saved_code"]
        current_run_result = execution_result["run_result"] or ""
        print(f"[系统] 新版本运行结果:\n{current_run_result}")
