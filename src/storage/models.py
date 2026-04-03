"""Pydantic models for AgentTrust data."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TargetType(str, Enum):
    MCP_SERVER = "mcp_server"
    AGENT = "agent"
    SKILL = "skill"


class EvalStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvalLevel(int, Enum):
    MANIFEST = 1
    FUNCTIONAL = 2
    DOMAIN_EXPERT = 3


class QualityTier(str, Enum):
    EXPERT = "expert"
    PROFICIENT = "proficient"
    BASIC = "basic"
    FAILED = "failed"


class ConnectionStrategy(str, Enum):
    SSE = "sse"
    DOCKER = "docker"
    SELF_REPORT = "self_report"
    A2A = "a2a"


class EvalMode(str, Enum):
    VERIFIED = "verified"      # DV — spot check (~30s)
    CERTIFIED = "certified"    # OV — full test suite (~90s)
    AUDITED = "audited"        # EV — comprehensive audit (~3min)


def normalize_eval_mode(raw: Optional[str]) -> Optional[str]:
    """Map old eval_mode values (quick/standard/full) to new names."""
    if raw is None:
        return None
    _COMPAT = {"quick": "verified", "standard": "certified", "full": "audited"}
    return _COMPAT.get(raw, raw)


# Request models
class EvaluateRequest(BaseModel):
    target_url: str
    target_type: TargetType = TargetType.MCP_SERVER
    level: EvalLevel = EvalLevel.FUNCTIONAL
    domains: List[str] = []
    eval_mode: EvalMode = EvalMode.CERTIFIED
    webhook_url: Optional[str] = None
    callback_secret: Optional[str] = None
    erc8004_agent_id: Optional[int] = None  # ERC-8004 agent token ID for on-chain reputation


# Response models
class EvaluateResponse(BaseModel):
    evaluation_id: str
    status: EvalStatus = EvalStatus.PENDING
    estimated_time_seconds: int = 60
    poll_url: str = ""
    message: str = ""


class ToolScore(BaseModel):
    score: int
    tests_passed: int
    tests_total: int


class ScoreResponse(BaseModel):
    target_id: str
    target_type: TargetType
    score: int = 0
    tier: QualityTier = QualityTier.FAILED
    confidence: float = 0.0
    domains: List[str] = []
    tool_scores: Dict[str, ToolScore] = {}
    evaluation_count: int = 0
    evaluation_version: Optional[str] = None
    last_evaluated_at: Optional[datetime] = None
    attestation_url: Optional[str] = None
    last_eval_mode: Optional[str] = None
    manifest_hash: Optional[str] = None


class EvaluationStatus(BaseModel):
    evaluation_id: str
    status: EvalStatus
    progress_pct: int = 0
    score: Optional[int] = None
    tier: Optional[str] = None
    eval_mode: Optional[str] = None
    evaluation_version: Optional[str] = None
    report: Optional[Dict[str, Any]] = None
    scores: Optional[Dict[str, Any]] = None
    attestation_jwt: Optional[str] = None
    badge_url: Optional[str] = None
    result: Optional[ScoreResponse] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None  # wall-clock eval time
    gaming_risk: Optional[str] = None  # none/low/medium/high
    timing_anomaly: Optional[bool] = None
    irt_theta: Optional[float] = None
    irt_se: Optional[float] = None
    confidence_interval: Optional[Dict[str, float]] = None
    token_usage: Optional[Dict[str, Any]] = None  # per-eval token tracking
    cost_usd: Optional[float] = None  # total cost in USD
    cost_summary: Optional[Dict[str, Any]] = None  # reshaped cost overview


# Webhook payload model
class WebhookPayload(BaseModel):
    event: str = "evaluation.completed"
    evaluation_id: str
    target_id: str
    score: int
    tier: str
    report_url: str
    badge_url: str
    attestation_url: Optional[str] = None
    signature: Optional[str] = None


# Agent card enrichment
class EnrichAgentCardRequest(BaseModel):
    agent_card: Dict[str, Any]


class EnrichAgentCardResponse(BaseModel):
    enriched_card: Dict[str, Any]
    quality_data: Optional[Dict[str, Any]] = None
    evaluate_url: Optional[str] = None


# Database document models
class EvaluationDoc(BaseModel):
    target_id: str
    target_type: TargetType
    target_url: str
    target_manifest: Optional[dict] = None
    status: EvalStatus = EvalStatus.PENDING
    level: EvalLevel = EvalLevel.FUNCTIONAL
    connection_strategy: ConnectionStrategy = ConnectionStrategy.SSE
    evaluation_version: str = "v1.0"
    questions_asked: int = 0
    questions_answered: int = 0
    scores: Optional[dict] = None
    report: Optional[Dict[str, Any]] = None
    llm_judge_model: Optional[str] = None
    llm_judge_responses: List[dict] = []
    webhook_url: Optional[str] = None
    callback_secret: Optional[str] = None
    attestation_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None


class ScoreDoc(BaseModel):
    target_id: str
    target_type: TargetType
    current_score: int = 0
    tier: QualityTier = QualityTier.FAILED
    confidence: float = 0.0
    evaluation_count: int = 0
    evaluation_version: Optional[str] = None
    domain_scores: Dict[str, dict] = {}
    tool_scores: Dict[str, dict] = {}
    first_evaluated_at: Optional[datetime] = None
    last_evaluated_at: Optional[datetime] = None
    next_evaluation_at: Optional[datetime] = None
    badge_url: Optional[str] = None
    last_eval_mode: Optional[str] = None


class ScoreHistoryDoc(BaseModel):
    target_id: str
    target_type: TargetType
    evaluation_id: str
    score: int
    tier: str
    confidence: float
    evaluation_version: str = "v1.0"
    domain_scores: Dict[str, int] = {}
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
    delta_from_previous: Optional[int] = None


# ── Production Feedback ──────────────────────────────────────────────────────

class FeedbackOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class FeedbackRequest(BaseModel):
    target_id: str
    outcome: FeedbackOutcome
    outcome_score: int = Field(ge=0, le=100)
    context: Optional[str] = None
    session_id: Optional[str] = None
    details: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    target_id: str
    message: str = "Feedback recorded"


class CorrelationResponse(BaseModel):
    target_id: str
    eval_score: int
    production_score: int
    correlation: Optional[float] = None
    feedback_count: int
    alignment: str
    confidence_adjustment: float
    sandbagging_risk: str
    outcome_breakdown: Dict[str, int] = {}


class FeedbackDoc(BaseModel):
    target_id: str
    outcome: FeedbackOutcome
    outcome_score: int = 0
    context: Optional[str] = None
    session_id: Optional[str] = None
    details: Optional[str] = None
    submitted_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Battle Arena ─────────────────────────────────────────────────────────────

class BattleStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BattleRequest(BaseModel):
    agent_a_url: str
    agent_b_url: str
    domain: Optional[str] = None
    challenge_count: int = Field(default=5, ge=3, le=15)
    eval_mode: EvalMode = EvalMode.VERIFIED
    blind: bool = True  # hide agent identities from judge


class BattleParticipant(BaseModel):
    target_id: str
    target_url: str
    name: str = ""
    eval_id: Optional[str] = None
    scores: Dict[str, float] = {}  # 6-axis scores
    overall_score: int = 0
    rating_before: Optional[Dict[str, float]] = None  # {mu, sigma}
    rating_after: Optional[Dict[str, float]] = None


class QuestionResponse(BaseModel):
    """Per-question response data for IRT calibration."""
    question_id: str = ""
    question_hash: str = ""
    domain: str = ""
    difficulty_tag: str = ""
    agent_a_correct: bool = False
    agent_b_correct: bool = False
    agent_a_score: float = 0.0
    agent_b_score: float = 0.0
    agent_a_latency_ms: int = 0
    agent_b_latency_ms: int = 0
    battle_discrimination: float = 0.0  # how well this Q separates the agents


class BattleIntegrity(BaseModel):
    """Arena integrity metadata (QO-009)."""
    blind_enforced: bool = True
    position_swapped: bool = False
    style_controlled: bool = False
    consistency: str = "not_checked"  # consistent | tie_forced | not_checked
    style_penalties: Dict[str, float] = Field(default_factory=lambda: {"agent_a": 0.0, "agent_b": 0.0})
    integrity_version: str = "1.0"


class BattleResult(BaseModel):
    battle_id: str
    agent_a: BattleParticipant
    agent_b: BattleParticipant
    winner: Optional[str] = None  # "a", "b", or None (draw)
    margin: int = 0
    photo_finish: bool = False  # margin < 5
    match_quality: float = 0.0
    domain: Optional[str] = None
    challenge_count: int = 5
    eval_mode: str = "verified"
    match_type: str = "manual"  # manual, ladder, swiss, queue
    duration_ms: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: BattleStatus = BattleStatus.PENDING
    question_responses: List[QuestionResponse] = []
    rating_deltas: Optional[Dict[str, Any]] = None  # {agent_a: {axes}, agent_b: {axes}}
    integrity: Optional[BattleIntegrity] = None
    error: Optional[str] = None


class MatchPrediction(BaseModel):
    agent_a_id: str
    agent_b_id: str
    win_probability_a: float = 0.5
    win_probability_b: float = 0.5
    match_quality: float = 0.0
    recommendation: str = "unknown"  # good_match, one_sided, too_unbalanced


# ── Divisions & Rankings ─────────────────────────────────────────────────────

class Division(str, Enum):
    CHALLENGER = "challenger"
    DIAMOND = "diamond"
    PLATINUM = "platinum"
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    UNRANKED = "unranked"


DIVISION_CONFIG = {
    Division.CHALLENGER: {"label": "Challenger", "color": "#FF4500", "icon": "crown", "min_mu": 40.0},
    Division.DIAMOND: {"label": "Diamond", "color": "#B9F2FF", "icon": "gem", "min_mu": 35.0},
    Division.PLATINUM: {"label": "Platinum", "color": "#E5E4E2", "icon": "shield", "min_mu": 30.0},
    Division.GOLD: {"label": "Gold", "color": "#FFD700", "icon": "medal", "min_mu": 27.0},
    Division.SILVER: {"label": "Silver", "color": "#C0C0C0", "icon": "star", "min_mu": 24.0},
    Division.BRONZE: {"label": "Bronze", "color": "#CD7F32", "icon": "circle", "min_mu": 20.0},
    Division.UNRANKED: {"label": "Unranked", "color": "#808080", "icon": "minus", "min_mu": 0.0},
}


def compute_division(mu: float, sigma: float, battles: int, is_top3: bool = False) -> str:
    """Compute division from rating stats. Top-3 override to Challenger."""
    if battles < 3:
        return Division.UNRANKED
    if is_top3 and mu >= DIVISION_CONFIG[Division.DIAMOND]["min_mu"]:
        return Division.CHALLENGER
    # High uncertainty keeps you lower
    effective = mu - sigma * 0.5
    for div in [Division.DIAMOND, Division.PLATINUM, Division.GOLD, Division.SILVER, Division.BRONZE]:
        if effective >= DIVISION_CONFIG[div]["min_mu"]:
            return div
    return Division.UNRANKED


class RankingEntry(BaseModel):
    target_id: str
    name: str = ""
    bt_rating: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    division: str = Division.UNRANKED
    division_config: Dict[str, Any] = {}
    battle_record: Dict[str, int] = Field(default_factory=lambda: {"wins": 0, "losses": 0, "draws": 0})
    openskill_mu: float = 25.0
    position: int = 0
    domain: Optional[str] = None


class AgentProfile(BaseModel):
    target_id: str
    name: str = ""
    bt_rating: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    division: str = Division.UNRANKED
    division_config: Dict[str, Any] = {}
    openskill_mu: float = 25.0
    openskill_sigma: float = 8.333
    battle_record: Dict[str, int] = Field(default_factory=lambda: {"wins": 0, "losses": 0, "draws": 0})
    total_battles: int = 0
    win_rate: float = 0.0
    current_streak: int = 0  # positive = win streak, negative = loss streak
    best_streak: int = 0
    per_axis_scores: Dict[str, float] = {}
    rating_history: List[Dict[str, Any]] = []
    recent_battles: List[Dict[str, Any]] = []
    position: int = 0
    domain: Optional[str] = None


class LadderEntry(BaseModel):
    target_id: str
    domain: Optional[str] = None
    position: int = 0
    target_url: str = ""
    name: str = ""
    overall_score: int = 0
    openskill_mu: float = 25.0
    openskill_sigma: float = 8.333
    battle_record: Dict[str, int] = Field(default_factory=lambda: {"wins": 0, "losses": 0, "draws": 0})
    last_challenge_at: Optional[datetime] = None
    seeded_at: datetime = Field(default_factory=datetime.utcnow)
    defenses: int = 0
