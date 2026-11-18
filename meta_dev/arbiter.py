"""DEV/QA deadlock detector, quantitative conflict arbiter, Judge explainer, and Quality Gates."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Final

from meta_dev.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, BaseLLMClient, chat_with_agent
from meta_dev.models import (
    AgentScore,
    ArbiterDecision,
    ArbitrationInput,
    ArbitrationResult,
    Claim,
    ClaimMetrics,
    ClaimScoreResult,
    DebateRound,
    DiscriminativeExperiment,
    ExperimentRecommendation,
    ProposalContext,
    ProposalScore,
    QualityGateVerdict,
    SandboxReport,
    TurnEnvelope,
)
from meta_dev.prompts import JUDGE_SYSTEM_PROMPT
from meta_dev.sandbox import run_result_indicates_success

SEVERITY_SCORES: Final[dict[str, float]] = {
    "LOW": 0.25,
    "MEDIUM": 0.50,
    "HIGH": 0.75,
    "BLOCKER": 1.00,
}

STAGE_RISK_BONUS: Final[dict[str, float]] = {
    "BUILD": 0.10,
    "VERIFY": 0.20,
    "RELEASE": 0.35,
}

ISSUE_TYPE_RISK_BONUS: Final[dict[str, float]] = {
    "SECURITY": 0.20,
    "PERMISSIONS": 0.20,
    "DATA_CONSISTENCY": 0.20,
    "DATA_LOSS": 0.20,
    "FINANCIAL": 0.20,
    "CONCURRENCY": 0.15,
    "IDEMPOTENCY": 0.15,
}

STAGE_DIMENSION_WEIGHTS: Final[dict[str, dict[str, float]]] = {
    "BUILD": {
        "requirement_fit": 0.20,
        "correctness": 0.20,
        "testability": 0.10,
        "robustness": 0.15,
        "maintainability": 0.15,
        "delivery_cost": 0.15,
        "rollbackability": 0.05,
    },
    "VERIFY": {
        "requirement_fit": 0.15,
        "correctness": 0.25,
        "testability": 0.15,
        "robustness": 0.20,
        "maintainability": 0.10,
        "delivery_cost": 0.10,
        "rollbackability": 0.05,
    },
    "RELEASE": {
        "requirement_fit": 0.10,
        "correctness": 0.25,
        "testability": 0.15,
        "robustness": 0.25,
        "maintainability": 0.05,
        "delivery_cost": 0.05,
        "rollbackability": 0.15,
    },
}

STAGE_COST_WEIGHT: Final[dict[str, float]] = {
    "BUILD": 0.20,
    "VERIFY": 0.12,
    "RELEASE": 0.05,
}


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-9:
        return default
    return numerator / denominator


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _upper(text: str) -> str:
    return text.strip().upper()


def _severity_score(severity: str) -> float:
    return SEVERITY_SCORES.get(_upper(severity), SEVERITY_SCORES["MEDIUM"])


def _stage_key(stage: str) -> str:
    key = _upper(stage)
    return key if key in STAGE_DIMENSION_WEIGHTS else "BUILD"


def _role_prior(issue_type: str, agent: str) -> float:
    issue_type_key = _upper(issue_type)
    agent_key = _upper(agent)

    dev_favored = {
        "FEASIBILITY",
        "TECHNICAL_FEASIBILITY",
        "ARCHITECTURE",
        "COMPLEXITY",
        "PERFORMANCE",
        "IMPLEMENTATION",
    }
    qa_favored = {
        "TESTABILITY",
        "REGRESSION_RISK",
        "RELEASE_BLOCKER",
        "RELEASE_RISK",
        "VERIFICATION",
    }

    if issue_type_key in dev_favored:
        return 1.10 if agent_key == "DEV" else 0.90
    if issue_type_key in qa_favored:
        return 1.10 if agent_key == "QA" else 0.90
    return 1.0


def compute_deadlock_index(history_rounds: list[DebateRound]) -> float:
    """Compute a normalized [0, 1] deadlock index across multi-round DEV/QA positions."""
    if len(history_rounds) < 2:
        return 0.0

    dev_same: list[float] = []
    qa_same: list[float] = []
    for previous, current in zip(history_rounds, history_rounds[1:], strict=False):
        dev_same.append(1.0 if _upper(previous.dev_position) == _upper(current.dev_position) else 0.0)
        qa_same.append(1.0 if _upper(previous.qa_position) == _upper(current.qa_position) else 0.0)

    position_stability = _avg(dev_same + qa_same)
    novelty_scores = [
        (round_item.dev_novelty + round_item.qa_novelty) / 2.0 for round_item in history_rounds
    ]
    resolution_scores = [
        _safe_div(round_item.resolved_objections, round_item.active_objections, 0.0)
        for round_item in history_rounds
    ]
    conflict_persistence = _avg(
        [
            1.0 if _upper(round_item.dev_position) != _upper(round_item.qa_position) else 0.0
            for round_item in history_rounds
        ]
    )

    return _clamp(
        0.35 * position_stability
        + 0.25 * (1.0 - _avg(novelty_scores))
        + 0.25 * conflict_persistence
        + 0.15 * (1.0 - _avg(resolution_scores))
    )


def is_deadlocked(arbitration_input: ArbitrationInput, deadlock_index: float) -> bool:
    """Return True when >=3 rounds show persistent low-novelty disagreement on >=Medium issues."""
    if len(arbitration_input.history_rounds) < 3:
        return False

    if _severity_score(arbitration_input.severity) < SEVERITY_SCORES["MEDIUM"]:
        return False

    recent_rounds = arbitration_input.history_rounds[-2:]
    no_new_info = all(
        round_item.dev_novelty < 0.20 and round_item.qa_novelty < 0.20
        for round_item in recent_rounds
    )
    still_not_ready = all(
        _upper(round_item.consensus_status) == "NOT READY" for round_item in recent_rounds
    )

    return deadlock_index >= 0.70 and no_new_info and still_not_ready


def score_claim(claim: Claim) -> float:
    """Evaluate a single atomic claim into a [0, 100] quality score."""
    m = claim.metrics
    base = (
        0.12 * m.traceability
        + 0.18 * m.evidence_strength
        + 0.10 * m.specificity
        + 0.12 * m.mechanism
        + 0.12 * m.verifiability
        + 0.10 * m.answer_quality
        + 0.08 * m.alternative_quality
        + 0.08 * m.impact_clarity
        + 0.05 * m.calibration
        + 0.05 * m.consistency
    )
    penalty = (
        0.20 * m.unsupported_assertion
        + 0.20 * m.contradiction
        + 0.15 * m.evasion
        + 0.10 * m.repetition
        + 0.10 * m.scope_drift
        + 0.15 * m.risk_blindness
        + 0.10 * m.hidden_assumptions
    )
    score = 100.0 * max(0.0, base - penalty)

    if m.evidence_strength < 0.30 and m.verifiability < 0.30:
        score = min(score, 40.0)
    if m.contradiction > 0.60:
        score = min(score, 25.0)

    return round(score, 4)


def claim_importance_weight(claim: Claim) -> float:
    return round(
        0.50 * _severity_score(claim.severity)
        + 0.30 * _clamp(claim.centrality)
        + 0.20 * _clamp(claim.breadth),
        4,
    )


def compute_claim_score_results(claims: list[Claim]) -> list[ClaimScoreResult]:
    """Score claims and discount defeated claims via the rebuttal argument graph."""
    raw_scores = {claim.claim_id: score_claim(claim) for claim in claims}

    rebuttal_map: dict[str, list[Claim]] = {}
    for claim in claims:
        for rebutted_id in claim.rebuts:
            rebuttal_map.setdefault(rebutted_id, []).append(claim)

    results: list[ClaimScoreResult] = []
    for claim in claims:
        defeated_by = None
        defeat_factor = 0.0
        rebuttals = rebuttal_map.get(claim.claim_id, [])
        if rebuttals:
            strongest = max(rebuttals, key=lambda item: raw_scores[item.claim_id])
            strongest_score = raw_scores[strongest.claim_id]
            if (
                strongest_score >= raw_scores[claim.claim_id] + 10.0
                and strongest.metrics.evidence_strength >= claim.metrics.evidence_strength
            ):
                defeated_by = strongest.claim_id
                defeat_factor = 1.0
            elif (
                strongest_score >= raw_scores[claim.claim_id] - 5.0
                and strongest.metrics.evidence_strength >= claim.metrics.evidence_strength
            ):
                defeated_by = strongest.claim_id
                defeat_factor = 0.5

        effective_score = round(raw_scores[claim.claim_id] * (1.0 - defeat_factor), 4)
        results.append(
            ClaimScoreResult(
                claim_id=claim.claim_id,
                author=claim.author,
                proposal=claim.proposal,
                raw_score=round(raw_scores[claim.claim_id], 4),
                effective_score=effective_score,
                weight=claim_importance_weight(claim),
                defeated_by=defeated_by,
                defeat_factor=defeat_factor,
            )
        )

    return results


def compute_agent_scores(
    arbitration_input: ArbitrationInput,
    claim_scores: list[ClaimScoreResult],
) -> dict[str, AgentScore]:
    grouped: dict[str, list[ClaimScoreResult]] = {"DEV": [], "QA": []}
    for item in claim_scores:
        grouped.setdefault(item.author, []).append(item)

    scores: dict[str, AgentScore] = {}
    for agent in ("DEV", "QA"):
        items = grouped.get(agent, [])
        total_weight = sum(item.weight for item in items)
        quality_score = _safe_div(
            sum(item.weight * item.effective_score for item in items),
            total_weight,
            0.0,
        )
        density = _safe_div(
            sum(1 for item in items if item.effective_score >= 60.0),
            len(items),
            0.0,
        )
        adjusted_quality_score = quality_score * (0.85 + 0.15 * density)
        brier = arbitration_input.history_brier_scores.get(agent)
        historical_reliability = 0.5 if brier is None else _clamp(1.0 - brier)
        credibility = _clamp(
            (0.8 * (adjusted_quality_score / 100.0) + 0.2 * historical_reliability)
            * _role_prior(arbitration_input.issue_type, agent)
        )

        scores[agent] = AgentScore(
            agent=agent,
            quality_score=round(quality_score, 4),
            density=round(density, 4),
            adjusted_quality_score=round(adjusted_quality_score, 4),
            historical_reliability=round(historical_reliability, 4),
            credibility=round(credibility, 4),
        )

    return scores


def compute_proposal_scores(
    arbitration_input: ArbitrationInput,
    claim_scores: list[ClaimScoreResult],
    agent_scores: dict[str, AgentScore],
) -> dict[str, ProposalScore]:
    claims_by_id = {claim.claim_id: claim for claim in arbitration_input.claims}
    claim_score_by_id = {item.claim_id: item for item in claim_scores}
    stage_key = _stage_key(arbitration_input.stage)
    dimension_weights = STAGE_DIMENSION_WEIGHTS[stage_key]

    proposal_scores: dict[str, ProposalScore] = {}
    for proposal in arbitration_input.proposals:
        dimension_scores: dict[str, float] = {}
        for dimension in dimension_weights:
            support = 0.0
            attack = 0.0
            for claim in arbitration_input.claims:
                if claim.proposal != proposal.proposal_id or dimension not in claim.dimensions:
                    continue
                claim_result = claim_score_by_id[claim.claim_id]
                agent_credibility = agent_scores[claim.author].credibility
                weighted_value = claim_result.weight * claim_result.effective_score * agent_credibility
                if claim.position == "SUPPORT":
                    support += weighted_value
                else:
                    attack += weighted_value

            ratio = 0.5 + 0.5 * _safe_div(support - attack, support + attack, 0.0)
            dimension_scores[dimension] = round(_clamp(ratio), 4)

        utility = round(
            sum(dimension_weights[dimension] * dimension_scores[dimension] for dimension in dimension_weights),
            4,
        )

        blocker_claims = [
            claims_by_id[item.claim_id]
            for item in claim_scores
            if item.proposal == proposal.proposal_id
            and claims_by_id[item.claim_id].position == "ATTACK"
            and _upper(claims_by_id[item.claim_id].severity) == "BLOCKER"
            and item.effective_score >= 75.0
            and claims_by_id[item.claim_id].metrics.verifiability >= 0.80
        ]
        strong_blocker = bool(blocker_claims)

        residual_risk = round(
            0.40 * (1.0 - dimension_scores.get("correctness", 0.5))
            + 0.25 * (1.0 - dimension_scores.get("robustness", 0.5))
            + 0.20 * (1.0 - dimension_scores.get("testability", 0.5))
            + 0.15 * (1.0 if strong_blocker else 0.0),
            4,
        )

        phi = (
            STAGE_RISK_BONUS[_stage_key(arbitration_input.stage)]
            + {"LOW": 0.00, "MEDIUM": 0.05, "HIGH": 0.10, "BLOCKER": 0.20}.get(
                _upper(arbitration_input.severity),
                0.05,
            )
            + ISSUE_TYPE_RISK_BONUS.get(_upper(arbitration_input.issue_type), 0.0)
        )

        owner_credibility = agent_scores[proposal.owner].credibility
        final_score = round((0.85 * utility) + (0.15 * owner_credibility) - (phi * residual_risk), 4)
        cost_proxy = (
            proposal.cost if proposal.cost is not None else 1.0 - dimension_scores.get("delivery_cost", 0.5)
        )
        regret = round(
            phi * residual_risk + STAGE_COST_WEIGHT[_stage_key(arbitration_input.stage)] * cost_proxy,
            4,
        )

        proposal_scores[proposal.proposal_id] = ProposalScore(
            proposal_id=proposal.proposal_id,
            owner=proposal.owner,
            utility=utility,
            residual_risk=residual_risk,
            final_score=final_score,
            regret=regret,
            cost=round(cost_proxy, 4),
            rollbackability=round(proposal.rollbackability, 4),
            strong_blocker=strong_blocker,
            dimension_scores=dimension_scores,
        )

    return proposal_scores


def compute_ambiguity_ratio(claims: list[Claim]) -> float:
    if not claims:
        return 0.0
    return round(
        sum(1 for claim in claims if claim.depends_on_undefined_requirement) / len(claims),
        4,
    )


def choose_minimal_experiment(
    experiments: list[DiscriminativeExperiment],
) -> ExperimentRecommendation | None:
    if not experiments:
        return None

    best_experiment = max(
        experiments,
        key=lambda item: _safe_div(
            abs(item.predicted_dev - item.predicted_qa) * item.criticality * item.executability,
            item.cost,
            0.0,
        ),
    )
    utility = _safe_div(
        abs(best_experiment.predicted_dev - best_experiment.predicted_qa)
        * best_experiment.criticality
        * best_experiment.executability,
        best_experiment.cost,
        0.0,
    )
    return ExperimentRecommendation(
        experiment_id=best_experiment.experiment_id,
        description=best_experiment.description,
        utility=round(utility, 4),
    )


def build_guardrails(
    winner_owner: str,
    losing_owner: str,
    claims: list[Claim],
    claim_scores: list[ClaimScoreResult],
) -> list[str]:
    del winner_owner  # retained for signature compatibility
    score_map = {item.claim_id: item for item in claim_scores}
    selected: list[str] = []
    for claim in claims:
        if claim.author != losing_owner or claim.position != "ATTACK":
            continue
        result = score_map[claim.claim_id]
        if result.effective_score < 65.0:
            continue
        message = f"{claim.claim_id}: {claim.text}"
        if claim.alternative:
            message += f" | Guardrail: {claim.alternative}"
        selected.append(message)
    return selected[:3]


def arbitrate_issue(arbitration_input: ArbitrationInput) -> ArbitrationResult:
    """Run quantitative claim, credibility, utility, and blocker-conservative arbitration."""
    deadlock_index = compute_deadlock_index(arbitration_input.history_rounds)
    arbitration_mode = is_deadlocked(arbitration_input, deadlock_index)
    ambiguity_ratio = compute_ambiguity_ratio(arbitration_input.claims)

    claim_scores = compute_claim_score_results(arbitration_input.claims)
    agent_scores_map = compute_agent_scores(arbitration_input, claim_scores)
    proposal_scores_map = compute_proposal_scores(arbitration_input, claim_scores, agent_scores_map)
    experiment_recommendation = choose_minimal_experiment(arbitration_input.experiments)

    proposals_ranked = sorted(
        proposal_scores_map.values(),
        key=lambda item: item.final_score,
        reverse=True,
    )
    top = proposals_ranked[0] if proposals_ranked else None
    second = proposals_ranked[1] if len(proposals_ranked) > 1 else None
    delta = (top.final_score - second.final_score) if top and second else 1.0

    reasons: list[str] = []
    decision = "RUN_MINIMAL_EXPERIMENT"
    chosen_proposal: str | None = None
    winner_agent: str | None = None
    guardrails: list[str] = []

    if ambiguity_ratio > 0.35:
        decision = "ESCALATE_TO_PM_AGENT"
        reasons.append("核心争议过度依赖未定义需求或验收标准，技术仲裁前应先补齐 PM 定义。")
    elif top is None:
        reasons.append("没有可评估的 proposal。")
    elif top.strong_blocker and not second:
        decision = "RUN_MINIMAL_EXPERIMENT"
        reasons.append("当前仅有一个方案且存在高质量未解除的 Blocker 攻击。")
    elif top.strong_blocker and second and not second.strong_blocker:
        decision = f"FOLLOW_{second.owner}"
        chosen_proposal = second.proposal_id
        winner_agent = second.owner
        reasons.append("高分方案仍带有未解除的高质量 Blocker，按保守原则转向另一方案。")
    elif top.final_score < 0.55 and (second is None or second.final_score < 0.55):
        decision = "RUN_MINIMAL_EXPERIMENT"
        reasons.append("双方方案分都不足以直接裁决，优先进入最小鉴别实验。")
    elif top.final_score >= 0.60 and delta >= 0.08:
        decision = f"FOLLOW_{top.owner}"
        chosen_proposal = top.proposal_id
        winner_agent = top.owner
        reasons.append("最高分方案形成了足够明显的综合优势。")
    elif second and delta < 0.05:
        lower_regret = min(proposals_ranked, key=lambda item: item.regret)
        if abs(proposals_ranked[0].regret - proposals_ranked[1].regret) < 0.03:
            decision = "RUN_MINIMAL_EXPERIMENT"
            reasons.append("双方最终分和后悔值都过于接近，实验优于拍板。")
        else:
            decision = f"FOLLOW_{lower_regret.owner}"
            chosen_proposal = lower_regret.proposal_id
            winner_agent = lower_regret.owner
            reasons.append("双方接近时采用最小后悔值原则。")
    else:
        decision = f"FOLLOW_{top.owner}"
        chosen_proposal = top.proposal_id
        winner_agent = top.owner
        reasons.append("虽然分差不大，但当前最高分方案仍是更稳妥的选择。")

    if chosen_proposal and winner_agent:
        losing_owner = "QA" if winner_agent == "DEV" else "DEV"
        guardrails = build_guardrails(winner_agent, losing_owner, arbitration_input.claims, claim_scores)
        if guardrails:
            reasons.append("已将败方高质量异议转为执行护栏，避免粗暴一刀切。")

    return ArbitrationResult(
        issue_id=arbitration_input.issue_id,
        question=arbitration_input.question,
        deadlock_index=round(deadlock_index, 4),
        ambiguity_ratio=ambiguity_ratio,
        arbitration_mode=arbitration_mode,
        decision=decision,
        chosen_proposal=chosen_proposal,
        winner_agent=winner_agent,
        reasons=reasons,
        guardrails=guardrails,
        agent_scores=list(agent_scores_map.values()),
        proposal_scores=list(proposal_scores_map.values()),
        claim_scores=claim_scores,
        experiment_recommendation=experiment_recommendation,
    )


def load_arbitration_input(file_path: str) -> ArbitrationInput:
    with open(file_path, encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    return ArbitrationInput.from_dict(payload)


def build_arbitration_template() -> dict[str, Any]:
    """Return a ready-to-run sample DEV/QA arbitration payload."""
    return {
        "issue": {
            "issue_id": "I_1",
            "question": "在 Verify 阶段，是否接受当前实现上线，还是要求补齐幂等保护后再继续？",
            "stage": "Verify",
            "issue_type": "regression_risk",
            "severity": "High",
            "dev_proposal_id": "P_DEV",
            "qa_proposal_id": "P_QA",
        },
        "history_brier_scores": {"DEV": 0.30, "QA": 0.18},
        "history_rounds": [
            {
                "dev_position": "可以上线",
                "qa_position": "必须阻塞",
                "dev_novelty": 0.25,
                "qa_novelty": 0.20,
                "resolved_objections": 0,
                "active_objections": 3,
                "consensus_status": "Not Ready",
            },
            {
                "dev_position": "可以上线",
                "qa_position": "必须阻塞",
                "dev_novelty": 0.10,
                "qa_novelty": 0.10,
                "resolved_objections": 0,
                "active_objections": 3,
                "consensus_status": "Not Ready",
            },
            {
                "dev_position": "可以上线",
                "qa_position": "必须阻塞",
                "dev_novelty": 0.05,
                "qa_novelty": 0.05,
                "resolved_objections": 0,
                "active_objections": 3,
                "consensus_status": "Not Ready",
            },
        ],
        "proposals": [
            {
                "proposal_id": "P_DEV",
                "owner": "DEV",
                "description": "接受当前实现并继续推进",
                "cost": 0.20,
                "rollbackability": 0.40,
            },
            {
                "proposal_id": "P_QA",
                "owner": "QA",
                "description": "先补齐幂等保护与回归测试再继续",
                "cost": 0.55,
                "rollbackability": 0.80,
            },
        ],
        "claims": [
            {
                "claim_id": "C_DEV_1",
                "author": "DEV",
                "proposal": "P_DEV",
                "position": "SUPPORT",
                "text": "当前改动局限在 webhook handler，爆炸半径可控。",
                "dimensions": ["maintainability", "delivery_cost", "rollbackability"],
                "severity": "Medium",
                "centrality": 0.7,
                "breadth": 0.5,
                "reference": "handler.py / webhook path",
                "confidence": 0.75,
                "metrics": {
                    "traceability": 0.8,
                    "evidence_strength": 0.6,
                    "specificity": 0.7,
                    "mechanism": 0.7,
                    "verifiability": 0.6,
                    "answer_quality": 0.6,
                    "alternative_quality": 0.3,
                    "impact_clarity": 0.6,
                    "calibration": 0.7,
                    "consistency": 0.8,
                },
            },
            {
                "claim_id": "C_QA_1",
                "author": "QA",
                "proposal": "P_DEV",
                "position": "ATTACK",
                "text": "当前实现缺少幂等保护，重试时可能重复写入，属于发布阻塞风险。",
                "dimensions": ["correctness", "robustness", "testability"],
                "severity": "Blocker",
                "centrality": 1.0,
                "breadth": 0.9,
                "reference": "AC3 / payment webhook / retry path",
                "rebuts": ["C_DEV_1"],
                "alternative": "加唯一键或幂等键，并补一条重放回归测试。",
                "confidence": 0.90,
                "metrics": {
                    "traceability": 1.0,
                    "evidence_strength": 0.9,
                    "specificity": 0.9,
                    "mechanism": 0.9,
                    "verifiability": 0.95,
                    "answer_quality": 0.8,
                    "alternative_quality": 0.9,
                    "impact_clarity": 0.95,
                    "calibration": 0.8,
                    "consistency": 0.9,
                },
            },
            {
                "claim_id": "C_QA_2",
                "author": "QA",
                "proposal": "P_QA",
                "position": "SUPPORT",
                "text": "先补齐幂等保护能显著降低回归和资金错误风险。",
                "dimensions": ["correctness", "robustness", "testability"],
                "severity": "High",
                "centrality": 0.9,
                "breadth": 0.8,
                "reference": "R7 / payment integrity",
                "confidence": 0.85,
                "metrics": {
                    "traceability": 0.9,
                    "evidence_strength": 0.8,
                    "specificity": 0.8,
                    "mechanism": 0.85,
                    "verifiability": 0.85,
                    "answer_quality": 0.7,
                    "alternative_quality": 0.8,
                    "impact_clarity": 0.9,
                    "calibration": 0.8,
                    "consistency": 0.9,
                },
            },
        ],
        "experiments": [
            {
                "experiment_id": "E_1",
                "description": "对同一 webhook payload 重放 200 次，检查是否产生重复写入。",
                "cost": 0.25,
                "executability": 0.90,
                "criticality": 1.00,
                "predicted_dev": 0.20,
                "predicted_qa": 0.90,
            }
        ],
    }


def _top_claims_for_judge(
    arbitration_input: ArbitrationInput,
    arbitration_result: ArbitrationResult,
    limit: int = 6,
) -> list[dict[str, Any]]:
    claim_map = {claim.claim_id: claim for claim in arbitration_input.claims}
    ranked_scores = sorted(
        arbitration_result.claim_scores,
        key=lambda item: item.effective_score,
        reverse=True,
    )[:limit]

    selected: list[dict[str, Any]] = []
    for score in ranked_scores:
        claim = claim_map[score.claim_id]
        selected.append(
            {
                "claim_id": claim.claim_id,
                "author": claim.author,
                "proposal": claim.proposal,
                "position": claim.position,
                "text": claim.text,
                "reference": claim.reference,
                "alternative": claim.alternative,
                "raw_score": score.raw_score,
                "effective_score": score.effective_score,
                "weight": score.weight,
                "defeated_by": score.defeated_by,
            }
        )
    return selected


def build_judge_explanation_payload(
    arbitration_input: ArbitrationInput,
    arbitration_result: ArbitrationResult,
) -> dict[str, Any]:
    return {
        "issue": {
            "issue_id": arbitration_input.issue_id,
            "question": arbitration_input.question,
            "stage": arbitration_input.stage,
            "issue_type": arbitration_input.issue_type,
            "severity": arbitration_input.severity,
        },
        "decision": {
            "decision": arbitration_result.decision,
            "chosen_proposal": arbitration_result.chosen_proposal,
            "winner_agent": arbitration_result.winner_agent,
            "deadlock_index": arbitration_result.deadlock_index,
            "ambiguity_ratio": arbitration_result.ambiguity_ratio,
            "arbitration_mode": arbitration_result.arbitration_mode,
            "reasons": arbitration_result.reasons,
            "guardrails": arbitration_result.guardrails,
        },
        "agent_scores": [asdict(item) for item in arbitration_result.agent_scores],
        "proposal_scores": [asdict(item) for item in arbitration_result.proposal_scores],
        "top_claims": _top_claims_for_judge(arbitration_input, arbitration_result),
        "experiment_recommendation": (
            asdict(arbitration_result.experiment_recommendation)
            if arbitration_result.experiment_recommendation
            else None
        ),
    }


def build_local_arbitration_explanation(
    arbitration_input: ArbitrationInput,
    arbitration_result: ArbitrationResult,
) -> str:
    """Render an offline deterministic Judge-Agent explanation from `ArbitrationResult`."""
    proposal_scores = {item.proposal_id: item for item in arbitration_result.proposal_scores}
    chosen_proposal = (
        proposal_scores.get(arbitration_result.chosen_proposal)
        if arbitration_result.chosen_proposal
        else None
    )
    top_claims = _top_claims_for_judge(arbitration_input, arbitration_result, limit=3)

    lines = [
        "ROLE: JUDGE",
        "MODE: Arbitration Explanation",
        "",
        "DECISION:",
        (
            f"- 当前裁决是 {arbitration_result.decision}。"
            f" winner_agent={arbitration_result.winner_agent or 'None'}，"
            f"chosen_proposal={arbitration_result.chosen_proposal or 'None'}。"
        ),
        (
            f"- 当前 issue 处于 {arbitration_input.stage} 阶段，"
            f"deadlock_index={arbitration_result.deadlock_index}，"
            f"ambiguity_ratio={arbitration_result.ambiguity_ratio}。"
        ),
        "",
        "SCORE_INTERPRETATION:",
    ]

    if chosen_proposal:
        lines.append(
            f"- 选中方案 {chosen_proposal.proposal_id} 的 final_score={chosen_proposal.final_score}，"
            f"utility={chosen_proposal.utility}，residual_risk={chosen_proposal.residual_risk}，"
            f"regret={chosen_proposal.regret}。"
        )

    for reason in arbitration_result.reasons:
        lines.append(f"- {reason}")

    lines.extend(["", "KEY_WINNING_CLAIMS:"])
    if top_claims:
        for claim in top_claims:
            lines.append(
                f"- {claim['claim_id']}({claim['author']}) score={claim['effective_score']}: {claim['text']}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "LOSING_SIDE_VALID_POINTS:"])
    if arbitration_result.guardrails:
        for guardrail in arbitration_result.guardrails:
            lines.append(f"- {guardrail}")
    else:
        lines.append("- 当前没有单独提取出的 guardrails。")

    lines.extend(["", "RISK_TRADEOFF:"])
    lines.append(
        "- 本次解释基于 proposal score、claim score、deadlock 指标与 residual risk 综合生成，"
        "目标是在当前阶段降低总体决策损失，而不是简单比较谁更会争论。"
    )

    lines.extend(["", "GUARDRAILS_OR_NEXT_STEP:"])
    if arbitration_result.decision == "RUN_MINIMAL_EXPERIMENT":
        if arbitration_result.experiment_recommendation:
            lines.append(
                f"- 优先执行实验 {arbitration_result.experiment_recommendation.experiment_id}: "
                f"{arbitration_result.experiment_recommendation.description}"
            )
        else:
            lines.append("- 当前建议进入最小鉴别实验，但输入中尚未提供 experiment。")
    elif arbitration_result.decision == "ESCALATE_TO_PM_AGENT":
        lines.append("- 当前建议升级给 PM-Agent，先补齐需求、验收口径或业务风险接受边界。")
    elif arbitration_result.guardrails:
        for guardrail in arbitration_result.guardrails:
            lines.append(f"- {guardrail}")
    else:
        lines.append("- 按当前裁决推进实现，并在下一轮验证中确认 residual risk 是否进一步下降。")

    lines.extend(["", "UNCERTAINTIES:"])
    if arbitration_result.experiment_recommendation:
        lines.append("- 仍存在可通过最小鉴别实验进一步压缩的不确定性。")
    else:
        lines.append("- 当前没有额外实验输入；若后续出现新证据，应重新计算 claim 与 proposal 分数。")

    return "\n".join(lines)


def generate_arbitration_explanation(
    arbitration_input: ArbitrationInput,
    arbitration_result: ArbitrationResult,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    llm_client: BaseLLMClient | None = None,
) -> str:
    judge_payload = build_judge_explanation_payload(arbitration_input, arbitration_result)
    user_message = (
        "请基于下面的仲裁摘要，输出一份给工程团队、测试团队和 PM 都能看懂的可读解释。"
        "不要重算新分，也不要虚构事实；以现有算法输出为准进行翻译和解释。\n\n"
        f"{json.dumps(judge_payload, ensure_ascii=False, indent=2)}"
    )
    if llm_client is not None:
        return llm_client.chat(
            JUDGE_SYSTEM_PROMPT,
            user_message,
            role="JUDGE",
            stage=arbitration_input.stage,
            stream=False,
        )
    return chat_with_agent(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_message=user_message,
        model=model,
        base_url=base_url,
        api_key=api_key,
        stream=False,
    )


# ---------------------------------------------------------------------------
# Envelope Deadlock Bridge & Quality Gate Verifier
# ---------------------------------------------------------------------------


def resolve_dev_qa_conflict(
    *,
    issue_id: str,
    stage: str,
    dev_envelope: TurnEnvelope,
    qa_envelope: TurnEnvelope,
    sandbox_report: SandboxReport | None = None,
    history_rounds: list[DebateRound] | None = None,
) -> ArbiterDecision:
    """Translate deadlocked DEV/QA TurnEnvelopes into an `ArbitrationInput` and resolve."""
    sandbox_failed = sandbox_report is not None and not sandbox_report.passed
    claims: list[Claim] = [
        Claim(
            claim_id="C_DEV_POS",
            author="DEV",
            proposal="P_DEV",
            position="SUPPORT",
            text=" ".join(dev_envelope.current_position) or "DEV advocates current implementation.",
            dimensions=["maintainability", "delivery_cost", "requirement_fit"],
            severity="Medium",
            metrics=ClaimMetrics(
                traceability=0.75,
                evidence_strength=0.45 if sandbox_failed else 0.80,
                specificity=0.75,
                mechanism=0.75,
                verifiability=0.70,
                answer_quality=0.70,
                alternative_quality=0.60,
                impact_clarity=0.70,
                calibration=0.70,
                consistency=0.80,
            ),
        )
    ]

    for idx, obj in enumerate(qa_envelope.objections, start=1):
        has_hard_evidence = sandbox_failed or obj.is_blocker
        claims.append(
            Claim(
                claim_id=f"C_QA_OBJ_{idx}",
                author="QA",
                proposal="P_DEV",
                position="ATTACK",
                text=obj.reason or f"QA objection on {obj.reference_id}",
                dimensions=["correctness", "robustness", "testability"],
                severity=obj.severity or "High",
                reference=obj.reference_id,
                rebuts=["C_DEV_POS"],
                alternative=obj.counterproposal,
                metrics=ClaimMetrics(
                    traceability=0.90,
                    evidence_strength=0.92 if has_hard_evidence else 0.65,
                    specificity=0.88,
                    mechanism=0.85,
                    verifiability=0.92 if has_hard_evidence else 0.70,
                    answer_quality=0.80,
                    alternative_quality=0.85 if obj.counterproposal else 0.50,
                    impact_clarity=0.90,
                    calibration=0.85,
                    consistency=0.90,
                ),
            )
        )

    claims.append(
        Claim(
            claim_id="C_QA_PROP",
            author="QA",
            proposal="P_QA",
            position="SUPPORT",
            text=" ".join(qa_envelope.current_position) or "QA advocates remediation before release.",
            dimensions=["correctness", "robustness", "testability"],
            severity="High" if qa_envelope.objections else "Medium",
            metrics=ClaimMetrics(
                traceability=0.85,
                evidence_strength=0.85 if sandbox_failed else 0.65,
                specificity=0.80,
                mechanism=0.80,
                verifiability=0.85,
                answer_quality=0.75,
                alternative_quality=0.80,
                impact_clarity=0.85,
                calibration=0.80,
                consistency=0.85,
            ),
        )
    )

    arb_input = ArbitrationInput(
        issue_id=issue_id,
        question=f"Resolve DEV vs QA conflict during {stage}",
        stage=stage,
        issue_type="verification" if stage.upper() == "VERIFY" else "correctness",
        severity="Blocker" if qa_envelope.blocker_objections else "High",
        dev_proposal_id="P_DEV",
        qa_proposal_id="P_QA",
        claims=claims,
        history_rounds=list(history_rounds or []),
        proposals=[
            ProposalContext(proposal_id="P_DEV", owner="DEV", cost=0.25, rollbackability=0.50),
            ProposalContext(proposal_id="P_QA", owner="QA", cost=0.50, rollbackability=0.85),
        ],
    )
    arb_result = arbitrate_issue(arb_input)
    arb_result.judge_explanation = build_local_arbitration_explanation(arb_input, arb_result)
    return ArbiterDecision.from_arbitration_result(stage=stage, result=arb_result)


def verify_quality_gate(
    sandbox_report: SandboxReport | None,
    qa_envelope: TurnEnvelope | None = None,
    *,
    stage: str = "Release",
) -> QualityGateVerdict:
    """Enforce deterministic quality guardrails; strictly refuse pass on timeout or test failure."""
    reasons: list[str] = []
    blocking_issues: list[str] = []

    if sandbox_report is None:
        blocking_issues.append("Missing sandbox verification report.")
        reasons.append("Quality Gate refused pass: code has not been verified in the local sandbox.")
        return QualityGateVerdict(
            passed=False,
            stage=stage,
            reasons=reasons,
            blocking_issues=blocking_issues,
            timed_out=False,
            exit_code=None,
        )

    if sandbox_report.timed_out:
        blocking_issues.append("Sandbox execution timed out.")
        reasons.append("Quality Gate refused pass: execution timed out (possible infinite loop or blocking I/O).")

    if (
        sandbox_report.exit_code != 0
        or not sandbox_report.passed
        or not run_result_indicates_success(sandbox_report.formatted_output)
    ):
        blocking_issues.append(
            f"Sandbox execution failed (exit_code={sandbox_report.exit_code})."
        )
        reasons.append("Quality Gate refused pass: automated tests or script execution failed.")

    if qa_envelope is not None:
        for obj in qa_envelope.blocker_objections:
            desc = f"[{obj.reference_id}] {obj.reason or 'Unresolved QA Blocker'}"
            blocking_issues.append(desc)
        if qa_envelope.blocker_objections:
            reasons.append(
                f"Quality Gate refused pass: {len(qa_envelope.blocker_objections)} unresolved QA Blocker objection(s)."
            )

    passed = len(blocking_issues) == 0
    if passed:
        reasons.append("All sandbox automated tests passed (exit_code=0) with zero QA blockers.")

    return QualityGateVerdict(
        passed=passed,
        stage=stage,
        reasons=reasons,
        blocking_issues=blocking_issues,
        timed_out=sandbox_report.timed_out,
        exit_code=sandbox_report.exit_code,
    )
