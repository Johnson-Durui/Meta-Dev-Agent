"""Typed domain models for the Meta-Dev multi-agent software factory."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


def _clamp_val(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class AgentRole(str, Enum):
    """Canonical roles in the Meta-Dev multi-agent software factory."""

    PM = "PM"
    DEV = "DEV"
    QA = "QA"
    ARBITER = "ARBITER"
    JUDGE = "JUDGE"


class Stage(str, Enum):
    """Five-stage software factory lifecycle."""

    CLARIFY = "Clarify"
    DESIGN = "Design"
    BUILD = "Build"
    VERIFY = "Verify"
    RELEASE = "Release"


@dataclass
class Objection:
    """Structured cross-agent objection raised during PM/DEV/QA collaboration."""

    target: str
    reference_id: str
    objection_type: str
    severity: str
    reason: str = ""
    risk: str = ""
    counterproposal: str = ""
    accept_condition: str = ""

    @property
    def is_blocker(self) -> bool:
        return self.severity.strip().upper() == "BLOCKER"


@dataclass
class TurnEnvelope:
    """Parsed structured envelope from an agent response."""

    role: str
    stage: str
    raw_text: str
    current_position: list[str] = field(default_factory=list)
    questions: dict[str, list[str]] = field(default_factory=dict)
    objections: list[Objection] = field(default_factory=list)
    responses_to_objections: list[str] = field(default_factory=list)
    proposal: str = ""
    agreements: list[str] = field(default_factory=list)
    open_issues: list[str] = field(default_factory=list)
    consensus_status: str = "Not Ready"
    consensus_reason: str = ""
    extracted_files: dict[str, str] = field(default_factory=dict)

    @property
    def blocker_objections(self) -> list[Objection]:
        return [obj for obj in self.objections if obj.is_blocker]

    @property
    def has_reached_consensus(self) -> bool:
        return self.consensus_status.strip().upper() == "REACHED"


@dataclass
class ASTGuardReport:
    """Static AST analysis report for generated Python source code."""

    has_blocking_input: bool = False
    blocking_calls: list[str] = field(default_factory=list)
    dangerous_calls: list[str] = field(default_factory=list)
    syntax_valid: bool = True
    syntax_error: str | None = None


@dataclass
class SandboxReport:
    """Isolated subprocess execution result from the local verification sandbox."""

    command: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: float
    route_reason: str
    passed: bool
    formatted_output: str
    ast_warnings: list[str] = field(default_factory=list)


@dataclass
class ClaimMetrics:
    """Ten positive quality metrics and seven penalty signals for atomic claims."""

    traceability: float = 0.0
    evidence_strength: float = 0.0
    specificity: float = 0.0
    mechanism: float = 0.0
    verifiability: float = 0.0
    answer_quality: float = 0.0
    alternative_quality: float = 0.0
    impact_clarity: float = 0.0
    calibration: float = 0.0
    consistency: float = 0.0
    unsupported_assertion: float = 0.0
    contradiction: float = 0.0
    evasion: float = 0.0
    repetition: float = 0.0
    scope_drift: float = 0.0
    risk_blindness: float = 0.0
    hidden_assumptions: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClaimMetrics:
        return cls(
            traceability=_clamp_val(float(data.get("traceability", 0.0))),
            evidence_strength=_clamp_val(float(data.get("evidence_strength", 0.0))),
            specificity=_clamp_val(float(data.get("specificity", 0.0))),
            mechanism=_clamp_val(float(data.get("mechanism", 0.0))),
            verifiability=_clamp_val(float(data.get("verifiability", 0.0))),
            answer_quality=_clamp_val(float(data.get("answer_quality", 0.0))),
            alternative_quality=_clamp_val(float(data.get("alternative_quality", 0.0))),
            impact_clarity=_clamp_val(float(data.get("impact_clarity", 0.0))),
            calibration=_clamp_val(float(data.get("calibration", 0.0))),
            consistency=_clamp_val(float(data.get("consistency", 0.0))),
            unsupported_assertion=_clamp_val(float(data.get("unsupported_assertion", 0.0))),
            contradiction=_clamp_val(float(data.get("contradiction", 0.0))),
            evasion=_clamp_val(float(data.get("evasion", 0.0))),
            repetition=_clamp_val(float(data.get("repetition", 0.0))),
            scope_drift=_clamp_val(float(data.get("scope_drift", 0.0))),
            risk_blindness=_clamp_val(float(data.get("risk_blindness", 0.0))),
            hidden_assumptions=_clamp_val(float(data.get("hidden_assumptions", 0.0))),
        )


@dataclass
class DebateRound:
    """Historical DEV/QA debate round snapshot used for deadlock computation."""

    dev_position: str
    qa_position: str
    dev_novelty: float
    qa_novelty: float
    resolved_objections: int
    active_objections: int
    consensus_status: str = "Not Ready"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DebateRound:
        return cls(
            dev_position=str(data.get("dev_position", "")),
            qa_position=str(data.get("qa_position", "")),
            dev_novelty=_clamp_val(float(data.get("dev_novelty", 0.0))),
            qa_novelty=_clamp_val(float(data.get("qa_novelty", 0.0))),
            resolved_objections=int(data.get("resolved_objections", 0)),
            active_objections=int(data.get("active_objections", 0)),
            consensus_status=str(data.get("consensus_status", "Not Ready")),
        )


@dataclass
class Claim:
    """Atomic argument claim supporting or attacking a candidate proposal."""

    claim_id: str
    author: str
    proposal: str
    position: str
    text: str
    dimensions: list[str]
    severity: str = "Medium"
    centrality: float = 1.0
    breadth: float = 0.5
    reference: str = ""
    rebuts: list[str] = field(default_factory=list)
    alternative: str = ""
    confidence: float = 0.5
    depends_on_undefined_requirement: bool = False
    metrics: ClaimMetrics = field(default_factory=ClaimMetrics)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Claim:
        return cls(
            claim_id=str(data["claim_id"]),
            author=str(data["author"]).strip().upper(),
            proposal=str(data["proposal"]),
            position=str(data.get("position", "SUPPORT")).strip().upper(),
            text=str(data.get("text", "")),
            dimensions=[str(item) for item in data.get("dimensions", [])] or ["correctness"],
            severity=str(data.get("severity", "Medium")),
            centrality=_clamp_val(float(data.get("centrality", 1.0))),
            breadth=_clamp_val(float(data.get("breadth", 0.5))),
            reference=str(data.get("reference", "")),
            rebuts=[str(item) for item in data.get("rebuts", [])],
            alternative=str(data.get("alternative", "")),
            confidence=_clamp_val(float(data.get("confidence", 0.5))),
            depends_on_undefined_requirement=bool(data.get("depends_on_undefined_requirement", False)),
            metrics=ClaimMetrics.from_dict(data.get("metrics", {})),
        )


@dataclass
class ProposalContext:
    """Candidate technical or verification proposal under arbitration."""

    proposal_id: str
    owner: str
    description: str = ""
    cost: float = 0.5
    rollbackability: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProposalContext:
        return cls(
            proposal_id=str(data["proposal_id"]),
            owner=str(data["owner"]).strip().upper(),
            description=str(data.get("description", "")),
            cost=_clamp_val(float(data.get("cost", 0.5))),
            rollbackability=_clamp_val(float(data.get("rollbackability", 0.5))),
        )


@dataclass
class DiscriminativeExperiment:
    """Minimal discriminating experiment candidate when verbal claims tie."""

    experiment_id: str
    description: str
    cost: float
    executability: float
    criticality: float
    predicted_dev: float
    predicted_qa: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscriminativeExperiment:
        return cls(
            experiment_id=str(data["experiment_id"]),
            description=str(data.get("description", "")),
            cost=max(float(data.get("cost", 1.0)), 0.01),
            executability=_clamp_val(float(data.get("executability", 1.0))),
            criticality=_clamp_val(float(data.get("criticality", 1.0))),
            predicted_dev=_clamp_val(float(data.get("predicted_dev", 0.5))),
            predicted_qa=_clamp_val(float(data.get("predicted_qa", 0.5))),
        )


@dataclass
class ArbitrationInput:
    """Complete structured input to the DEV/QA conflict arbiter."""

    issue_id: str
    question: str
    stage: str
    issue_type: str
    severity: str
    dev_proposal_id: str
    qa_proposal_id: str
    claims: list[Claim]
    history_rounds: list[DebateRound] = field(default_factory=list)
    proposals: list[ProposalContext] = field(default_factory=list)
    experiments: list[DiscriminativeExperiment] = field(default_factory=list)
    history_brier_scores: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArbitrationInput:
        issue = data.get("issue", {})
        proposals = [ProposalContext.from_dict(item) for item in data.get("proposals", [])]
        if not proposals:
            dev_id = str(issue.get("dev_proposal_id", "P_DEV"))
            qa_id = str(issue.get("qa_proposal_id", "P_QA"))
            proposals = [
                ProposalContext(proposal_id=dev_id, owner="DEV", description="DEV proposal"),
                ProposalContext(proposal_id=qa_id, owner="QA", description="QA proposal"),
            ]

        return cls(
            issue_id=str(issue.get("issue_id", "I_1")),
            question=str(issue.get("question", "")),
            stage=str(issue.get("stage", "Build")),
            issue_type=str(issue.get("issue_type", "correctness")),
            severity=str(issue.get("severity", "Medium")),
            dev_proposal_id=str(issue.get("dev_proposal_id", proposals[0].proposal_id)),
            qa_proposal_id=str(
                issue.get(
                    "qa_proposal_id",
                    proposals[1].proposal_id if len(proposals) > 1 else "P_QA",
                )
            ),
            claims=[Claim.from_dict(item) for item in data.get("claims", [])],
            history_rounds=[DebateRound.from_dict(item) for item in data.get("history_rounds", [])],
            proposals=proposals,
            experiments=[DiscriminativeExperiment.from_dict(item) for item in data.get("experiments", [])],
            history_brier_scores={
                str(key).strip().upper(): _clamp_val(float(value))
                for key, value in data.get("history_brier_scores", {}).items()
            },
        )


@dataclass
class ClaimScoreResult:
    """Evaluated claim score after argument-graph rebuttal resolution."""

    claim_id: str
    author: str
    proposal: str
    raw_score: float
    effective_score: float
    weight: float
    defeated_by: str | None
    defeat_factor: float


@dataclass
class AgentScore:
    """Credibility and argument density summary for DEV or QA."""

    agent: str
    quality_score: float
    density: float
    adjusted_quality_score: float
    historical_reliability: float
    credibility: float


@dataclass
class ProposalScore:
    """Multi-dimensional utility, residual risk, and regret score for a proposal."""

    proposal_id: str
    owner: str
    utility: float
    residual_risk: float
    final_score: float
    regret: float
    cost: float
    rollbackability: float
    strong_blocker: bool
    dimension_scores: dict[str, float]


@dataclass
class ExperimentRecommendation:
    """Highest utility discriminating experiment recommendation."""

    experiment_id: str
    description: str
    utility: float


@dataclass
class ArbitrationResult:
    """Full quantitative output produced by `arbitrate_issue`."""

    issue_id: str
    question: str
    deadlock_index: float
    ambiguity_ratio: float
    arbitration_mode: bool
    decision: str
    chosen_proposal: str | None
    winner_agent: str | None
    reasons: list[str]
    guardrails: list[str]
    agent_scores: list[AgentScore]
    proposal_scores: list[ProposalScore]
    claim_scores: list[ClaimScoreResult]
    experiment_recommendation: ExperimentRecommendation | None = None
    judge_explanation: str | None = None


@dataclass
class ArbiterDecision:
    """High-level conflict resolution record stored in pipeline run manifests."""

    issue_id: str
    stage: str
    deadlocked: bool
    deadlock_index: float
    ambiguity_ratio: float
    decision: str
    winner_agent: str | None
    chosen_proposal: str | None
    reasons: list[str] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    judge_explanation: str | None = None

    @classmethod
    def from_arbitration_result(
        cls,
        stage: str,
        result: ArbitrationResult,
    ) -> ArbiterDecision:
        return cls(
            issue_id=result.issue_id,
            stage=stage,
            deadlocked=result.arbitration_mode,
            deadlock_index=result.deadlock_index,
            ambiguity_ratio=result.ambiguity_ratio,
            decision=result.decision,
            winner_agent=result.winner_agent,
            chosen_proposal=result.chosen_proposal,
            reasons=list(result.reasons),
            guardrails=list(result.guardrails),
            judge_explanation=result.judge_explanation,
        )


@dataclass
class QualityGateVerdict:
    """Deterministic quality guardrail decision before Release / Git commit."""

    passed: bool
    stage: str
    reasons: list[str] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    timed_out: bool = False
    exit_code: int | None = None


@dataclass
class HealIterationRecord:
    """Audit record for a single Verify -> Self-Heal loop iteration."""

    round_index: int
    sandbox_passed: bool
    exit_code: int | None
    timed_out: bool
    qa_summary: str
    files_updated: list[str] = field(default_factory=list)


@dataclass
class PipelineRunSummary:
    """End-to-end execution manifest for a 5-stage Software Factory run."""

    run_id: str
    requirement: str
    workspace: str
    model: str
    dry_run: bool
    stages_completed: list[str]
    heal_rounds_used: int
    max_heal_rounds: int
    saved_files: list[str]
    final_sandbox_report: SandboxReport | None
    quality_gate: QualityGateVerdict
    arbiter_decisions: list[ArbiterDecision] = field(default_factory=list)
    heal_history: list[HealIterationRecord] = field(default_factory=list)
    release_decision: str = "NO-GO"
    commit_hash: str | None = None
    commit_message: str | None = None
    manifest_path: str | None = None
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
