"""Tests for DEV/QA conflict arbitration, deadlock index, Judge explainer, and Quality Gates."""

from __future__ import annotations

from meta_dev.arbiter import (
    arbitrate_issue,
    build_arbitration_template,
    build_local_arbitration_explanation,
    compute_deadlock_index,
    is_deadlocked,
    resolve_dev_qa_conflict,
    verify_quality_gate,
)
from meta_dev.models import (
    ArbitrationInput,
    Objection,
    SandboxReport,
    TurnEnvelope,
)


def test_template_arbitration_detects_deadlock_and_follows_qa_on_strong_blocker() -> None:
    arb_input = ArbitrationInput.from_dict(build_arbitration_template())
    deadlock_idx = compute_deadlock_index(arb_input.history_rounds)
    assert deadlock_idx >= 0.70
    assert is_deadlocked(arb_input, deadlock_idx) is True

    result = arbitrate_issue(arb_input)
    assert result.arbitration_mode is True
    assert result.decision == "FOLLOW_QA"
    assert result.winner_agent == "QA"
    assert result.chosen_proposal == "P_QA"

    explanation = build_local_arbitration_explanation(arb_input, result)
    assert "ROLE: JUDGE" in explanation
    assert "FOLLOW_QA" in explanation


def test_arbitration_escalates_to_pm_when_requirement_ambiguity_is_high() -> None:
    payload = build_arbitration_template()
    for claim in payload["claims"]:
        claim["depends_on_undefined_requirement"] = True
    arb_input = ArbitrationInput.from_dict(payload)
    result = arbitrate_issue(arb_input)
    assert result.decision == "ESCALATE_TO_PM_AGENT"


def test_quality_gate_blocks_failed_or_timed_out_sandbox_and_unresolved_blockers() -> None:
    # 1. Missing sandbox report -> blocked
    verdict_none = verify_quality_gate(None)
    assert verdict_none.passed is False

    # 2. Failed unit tests -> blocked even if QA says no objections
    failed_report = SandboxReport(
        command=["pytest", "-q"],
        exit_code=1,
        stdout="1 failed",
        stderr="",
        timed_out=False,
        duration_ms=50.0,
        route_reason="pytest",
        passed=False,
        formatted_output="[COMMAND] pytest -q\n\n[EXIT_CODE] 1\n\n[STDOUT]\n1 failed",
    )
    clean_qa = TurnEnvelope(role="QA", stage="Verify", raw_text="", consensus_status="Reached")
    verdict_failed = verify_quality_gate(failed_report, clean_qa)
    assert verdict_failed.passed is False
    assert any("failed" in item.lower() for item in verdict_failed.blocking_issues)

    # 3. Timed-out sandbox -> blocked
    timeout_report = SandboxReport(
        command=["python", "main.py"],
        exit_code=None,
        stdout="",
        stderr="",
        timed_out=True,
        duration_ms=1000.0,
        route_reason="entrypoint",
        passed=False,
        formatted_output="ERROR: [TIMEOUT] timed out",
    )
    verdict_timeout = verify_quality_gate(timeout_report, clean_qa)
    assert verdict_timeout.passed is False
    assert verdict_timeout.timed_out is True

    # 4. Passing sandbox BUT unresolved QA Blocker -> blocked
    passing_report = SandboxReport(
        command=["pytest", "-q"],
        exit_code=0,
        stdout="4 passed",
        stderr="",
        timed_out=False,
        duration_ms=42.0,
        route_reason="pytest",
        passed=True,
        formatted_output="[COMMAND] pytest -q\n\n[EXIT_CODE] 0\n\n[STDOUT]\n4 passed",
    )
    blocker_qa = TurnEnvelope(
        role="QA",
        stage="Verify",
        raw_text="",
        objections=[
            Objection(
                target="DEV",
                reference_id="B99",
                objection_type="安全",
                severity="Blocker",
                reason="Hardcoded secret in config",
            )
        ],
    )
    verdict_qa_blocked = verify_quality_gate(passing_report, blocker_qa)
    assert verdict_qa_blocked.passed is False

    # 5. Passing sandbox + zero QA blockers -> GO!
    verdict_ok = verify_quality_gate(passing_report, clean_qa)
    assert verdict_ok.passed is True


def test_resolve_dev_qa_conflict_from_envelopes() -> None:
    dev_env = TurnEnvelope(
        role="DEV",
        stage="Verify",
        raw_text="",
        current_position=["Current code is minimal and sufficient."],
    )
    qa_env = TurnEnvelope(
        role="QA",
        stage="Verify",
        raw_text="",
        current_position=["Unit test fails on priority ordering."],
        objections=[
            Objection(
                target="DEV",
                reference_id="B1",
                objection_type="质量",
                severity="Blocker",
                reason="Ascending sort breaks R2",
                counterproposal="Sort by -item.priority",
            )
        ],
    )
    decision = resolve_dev_qa_conflict(
        issue_id="I_TEST",
        stage="Verify",
        dev_envelope=dev_env,
        qa_envelope=qa_env,
    )
    assert decision.decision == "FOLLOW_QA"
    assert decision.winner_agent == "QA"


def test_rebuttal_graph_discounts_defeated_claim_and_ranks_experiments() -> None:
    arb_input = ArbitrationInput.from_dict(build_arbitration_template())
    result = arbitrate_issue(arb_input)
    score_by_id = {item.claim_id: item for item in result.claim_scores}

    # C_QA_1 rebuts C_DEV_1 with much stronger evidence/specificity -> C_DEV_1 is defeated
    assert score_by_id["C_DEV_1"].defeated_by == "C_QA_1"
    assert score_by_id["C_DEV_1"].defeat_factor == 1.0
    assert score_by_id["C_DEV_1"].effective_score == 0.0

    assert result.experiment_recommendation is not None
    assert result.experiment_recommendation.experiment_id == "E_1"
    assert result.experiment_recommendation.utility > 2.0

