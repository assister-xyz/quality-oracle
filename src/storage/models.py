"""Pydantic models for AgentTrust data."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TargetType(str, Enum):
    # ── Original triplet (pre-QO-058) ──────────────────────────────────────
    MCP_SERVER = "mcp_server"
    AGENT = "agent"  # legacy generic-agent slot — pre-058 dispatch did nothing
    SKILL = "skill"
    # ── QO-058: generic agent protocols ────────────────────────────────────
    # A2A_AGENT  — Google Agent2Agent v1.0 (12 Mar 2026), agent-card.json
    # REST_CHAT  — manifest-less generic REST/JSON chat agents (long tail).
    #              Capped at Verified tier unless OpenAPI doc supplied or
    #              calibration n=10 + correlation feedback ≥10 escalates
    #              `inference_confidence` from medium → high (spec §"Path to
    #              Certified for manifest-less").
    # OPENAPI_AGENT — Generic OpenAPI/Swagger-described agent. Manifest is
    #                 the OpenAPI doc → full 6-axis weights apply.
    # UNKNOWN     — placeholder pre-resolution; the discovery cascade rewrites
    #               it to a concrete type before evaluator dispatch.
    #
    # Migration note: legacy rows with target_type='agent' need NO migration —
    # the dispatcher routes them to the not-implemented branch in 053-C and
    # 058 leaves that path alone (fail-fast). New evaluations submit one of
    # the four new values explicitly.
    A2A_AGENT = "a2a_agent"
    REST_CHAT = "rest_chat"
    OPENAPI_AGENT = "openapi_agent"
    UNKNOWN = "unknown"


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


# ── QO-053-C (CB3): public-name → (EvalLevel, EvalMode) mapping ─────────────
#
# The codebase enum at ``EvalLevel`` is ``MANIFEST=1, FUNCTIONAL=2,
# DOMAIN_EXPERT=3``. The public copy in landing-page tier names ("L1
# functional", "L2 certified", "L3 stress / audited") used in QO-053 specs
# is *display only*. This helper maps a freeform public-name string into
# the actual ``(EvalLevel, EvalMode)`` tuple consumed by the dispatcher.

_PUBLIC_LEVEL_MAP: Dict[str, "tuple[EvalLevel, EvalMode]"] = {
    # L1 — manifest validation only.
    "l1": (EvalLevel.MANIFEST, EvalMode.VERIFIED),
    "l1 functional": (EvalLevel.MANIFEST, EvalMode.VERIFIED),
    "l1-functional": (EvalLevel.MANIFEST, EvalMode.VERIFIED),
    # L2 — functional / certified.
    "l2": (EvalLevel.FUNCTIONAL, EvalMode.CERTIFIED),
    "l2 certified": (EvalLevel.FUNCTIONAL, EvalMode.CERTIFIED),
    "l2-certified": (EvalLevel.FUNCTIONAL, EvalMode.CERTIFIED),
    # L3 — domain expert / stress / audited.
    "l3": (EvalLevel.DOMAIN_EXPERT, EvalMode.AUDITED),
    "l3 stress": (EvalLevel.DOMAIN_EXPERT, EvalMode.AUDITED),
    "l3 audited": (EvalLevel.DOMAIN_EXPERT, EvalMode.AUDITED),
    "l3-stress": (EvalLevel.DOMAIN_EXPERT, EvalMode.AUDITED),
    "l3-audited": (EvalLevel.DOMAIN_EXPERT, EvalMode.AUDITED),
}


def level_for_skill_eval(public_name: str) -> "tuple[EvalLevel, EvalMode]":
    """Resolve a spec/landing-page public level name into the codebase enums.

    >>> level_for_skill_eval("L2 certified")
    (<EvalLevel.FUNCTIONAL: 2>, <EvalMode.CERTIFIED: 'certified'>)

    Unknown names raise ``ValueError`` rather than silently picking a default
    so a typo in a question-pack manifest fails loudly during ingest.
    """
    key = (public_name or "").strip().lower()
    if key in _PUBLIC_LEVEL_MAP:
        return _PUBLIC_LEVEL_MAP[key]
    raise ValueError(
        f"Unknown skill-eval public level {public_name!r}. "
        f"Valid: {sorted(set(_PUBLIC_LEVEL_MAP))}"
    )


# ── Sybil Defense (QO-044) ───────────────────────────────────────────────────


class OperatorStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    BANNED = "banned"


class CloneSuspectStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED_CLONE = "confirmed_clone"
    CLEARED = "cleared"


class Operator(BaseModel):
    operator_id: str
    display_name: str
    email: Optional[str] = None
    auth_provider: str = "email"  # email (legacy) | github (verified)

    # GitHub OAuth fields (QO-046) — present when auth_provider == "github"
    github_user_id: Optional[int] = None
    github_username: Optional[str] = None
    github_avatar_url: Optional[str] = None
    github_account_age_days: Optional[int] = None
    github_public_repos: Optional[int] = None
    github_followers: Optional[int] = None

    # Verified flag: True only when auth_provider == "github" AND passed anti-abuse checks
    verified: bool = False

    agent_target_ids: List[str] = Field(default_factory=list)
    max_agents: int = 5
    max_battles_per_day: int = 15
    status: OperatorStatus = OperatorStatus.ACTIVE
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None


class OperatorRegisterRequest(BaseModel):
    display_name: str
    email: str


class OperatorRegisterResponse(BaseModel):
    operator_id: str
    display_name: str
    email: str
    max_agents: int
    max_battles_per_day: int
    created_at: datetime


class CloneSuspect(BaseModel):
    agent_a_id: str
    agent_b_id: str
    similarity_score: float  # Jaccard 0.0-1.0
    matched_questions: int
    total_questions: int
    status: CloneSuspectStatus = CloneSuspectStatus.PENDING
    detected_at: datetime
    notes: Optional[str] = None


class AuthMeResponse(BaseModel):
    """Current verified operator info (GET /v1/auth/me)."""
    operator_id: str
    display_name: str
    github_username: str
    github_avatar_url: Optional[str] = None
    email: Optional[str] = None
    verified: bool = True
    agent_count: int
    max_agents: int
    max_battles_per_day: int
    created_at: datetime
    last_login_at: Optional[datetime] = None


# ── Cost per Correct Response (QO-051) ──────────────────────────────────────


class CPCRScores(BaseModel):
    """Three CPCR variants published per evaluation.

    binary: cost_usd / correct_count (LayerLens/PinchBench-compatible).
    weighted: cost_usd / sum(score/100) — honours partial credit.
    shadow: shadow_cost_usd / correct_count — always published at public
            market rates so free-tier evals remain comparable on the
            leaderboard and in the AQVC credential.
    """
    correct_threshold: int = 70
    correct_count: int = 0
    total_responses: int = 0
    cpcr: Optional[float] = None
    weighted_cpcr: Optional[float] = None
    shadow_cpcr: Optional[float] = None


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
    # QO-051: cost + CPCR on the public score endpoint. shadow_cpcr is the
    # canonical public value because free-tier evals would otherwise appear
    # as $0/correct and be meaningless on the leaderboard.
    cost_usd: Optional[float] = None
    shadow_cost_usd: Optional[float] = None
    cpcr: Optional[CPCRScores] = None


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

    # ── QO-053-C (CB3 + AC9): per-target dispatch + differential audit fields
    # All optional → existing MCP eval documents remain valid without
    # migration. Migration script ``dev/migrate_legacy_evaluations.py`` fills
    # ``target_type_dispatched`` for any pre-053-C row.
    target_type_dispatched: Optional[TargetType] = None
    subject_uri: Optional[str] = None
    axis_weights_used: Optional[Dict[str, float]] = None
    delta_vs_baseline: Optional[float] = None
    baseline_score: Optional[float] = None
    baseline_status: Optional[str] = None  # "ok" | "failed" | None


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
    # ── QO-061: eval-score snapshot for honest correlation ──────────────
    # Captured at write time so the correlation engine pairs feedback against
    # the eval score that was current WHEN the user reported the outcome —
    # not the eval score whenever the report is queried.
    eval_score_at_time: Optional[int] = None
    # KYA tier of the reporter at time of submission. Weights the row in the
    # weighted Pearson correlation: free=1.0, builder=2.0, team=3.0.
    reporter_kya_tier: int = 1
    # Set to "legacy_kya_unknown" by the QO-061 backfill migration on rows
    # that pre-date the schema change. Downstream readers must surface this.
    data_quality_warning: Optional[str] = None


class FeedbackSnapshot(BaseModel):
    """Paired (eval_score, outcome_score) row for the correlation engine.

    QO-061: replaces the old `(target_id, eval_score, feedback_items)` tuple
    that was passed to `compute_correlation_report`. The whole point of this
    model is to make the eval-score snapshot a first-class field — the OLD
    code computed Pearson on `(feedback_index, outcome)` (a drift detector,
    not anti-sandbagging).
    """
    target_id: str
    eval_score_at_time: float
    feedback_outcome: float          # outcome_score 0-100
    reporter_kya_tier: int = 1       # 1=free, 2=builder, 3=team
    weight: float = 1.0
    timestamp: Optional[datetime] = None
    data_quality_warning: Optional[str] = None


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


# ── Skill Parser & Spec Compliance (QO-053-A) ────────────────────────────────


class Severity(str, Enum):
    """Severity tier for spec violations and anti-pattern findings."""
    HIGH = "high"
    MED = "med"
    LOW = "low"


class ParsedSkill(BaseModel):
    """Parsed Anthropic Agent-Skills SKILL.md (R1 mirror).

    Fields populated by `core.skill_parser.parse_skill_md`. Extra YAML keys land
    in `frontmatter_raw` so callers can inspect off-spec input transparently.
    """
    model_config = ConfigDict(populate_by_name=True)

    name: str  # NFKC-normalized
    description: str
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)
    allowed_tools: List[str] = Field(default_factory=list, alias="allowed-tools")
    body: str = ""
    body_size_bytes: int = 0
    body_lines: int = 0
    body_tokens: Optional[int] = None
    spec_dirs: Dict[str, bool] = Field(default_factory=dict)
    convention_dirs: Dict[str, bool] = Field(default_factory=dict)
    extra_dirs: List[str] = Field(default_factory=list)
    folder_name: str = ""
    folder_name_nfkc: str = ""
    frontmatter_raw: Dict[str, Any] = Field(default_factory=dict)
    git_sha: Optional[str] = None
    parse_warnings: List[str] = Field(default_factory=list)


class Violation(BaseModel):
    """Single spec-compliance violation produced by `validate_skill`."""
    rule: str  # AP1, AP2, ..., AP18 (or AP5_LONG warning)
    severity: Severity
    field: Optional[str] = None
    line: Optional[int] = None
    message: str
    suggestion: str = ""
    score_deduction: int = 0  # 0–20


class SpecCompliance(BaseModel):
    """Aggregate compliance result over a single ParsedSkill."""
    score: int = 100  # 0–100
    violations: List[Violation] = Field(default_factory=list)
    passed_hard_fails: bool = True  # False if any HIGH violation


class AntiPattern(BaseModel):
    """One anti-pattern finding produced by `detect_anti_patterns`."""
    id: str  # AP1..AP18
    severity: Severity
    field: Optional[str] = None
    line: Optional[int] = None
    regex_match: Optional[str] = None
    message: str
    suggestion: str = ""


# ── Skill Activation Adapter (QO-053-B) ──────────────────────────────────────


class ToolCall(BaseModel):
    """One tool invocation recorded during an L2/L3 activation turn.

    The activator's MockFileSystem (and, later, the QO-059 Docker harness)
    record every tool the activated agent attempts to call. Downstream probes
    in QO-053-D/E inspect this log to detect script poisoning, fee-payer
    hijack, and other tool-trace anomalies.
    """
    tool: str  # Read | Bash | Glob | Grep | Edit | Write
    args: Dict[str, Any] = Field(default_factory=dict)
    returned: Optional[str] = None
    error: Optional[str] = None
    blocked: bool = False  # True when path-escape / network access denied
    duration_ms: int = 0


class UsageSummary(BaseModel):
    """Aggregate token + cost summary across an activation session.

    Mirrors the structure consumed by QO-051 CPCR aggregator. ``dollars_spent``
    is computed at the activator using ``calculate_cost`` so that free-tier
    runs honestly report ``0.0`` and the shadow-cost path is computed by
    callers from raw token counts.
    """
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    dollars_spent: float = 0.0
    model: str = ""
    provider: str = ""
    request_ids: List[str] = Field(default_factory=list)
    n_calls: int = 0


class ActivationResponse(BaseModel):
    """Single ``respond()`` call return value for any SkillActivatedAgent.

    Mirrors the public contract used by the evaluator dispatch in
    QO-053-C; ``parse_warnings`` surfaces both bash-preprocessor strips and
    cache-disabled-below-min-tokens (AC9) so audit can see why the per-call
    cost spiked.
    """
    text: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str
    provider: str = ""
    request_id: str = ""
    latency_ms: int = 0
    parse_warnings: List[str] = Field(default_factory=list)


class ActivationFailure(Exception):
    """Raised when an activation provider exhausts retries.

    Attributes
    ----------
    last_request_id:
        Provider-side request ID from the final attempt — propagated to the
        DLQ entry so engineers can grep provider logs without replaying the
        whole eval.
    provider:
        ``cerebras`` / ``groq`` / ``anthropic``.
    attempt_count:
        Total attempts made (1 + retries).
    """

    def __init__(
        self,
        message: str,
        last_request_id: str = "",
        provider: str = "",
        attempt_count: int = 0,
    ):
        super().__init__(message)
        self.last_request_id = last_request_id
        self.provider = provider
        self.attempt_count = attempt_count


class ModelVersionRecord(BaseModel):
    """Persisted (alias, dated_snapshot, resolved_at) row for AQVC reproducibility.

    See AC4. Stored in ``quality__model_versions``; one document per provider
    (``alias`` is unique per provider so a re-resolve overwrites in place).
    """
    provider: str
    alias: str  # e.g. ``claude-sonnet-4-5`` or ``llama3.1-8b``
    dated_snapshot: str  # e.g. ``claude-sonnet-4-5-20250929``
    resolved_at: datetime = Field(default_factory=datetime.utcnow)
    source: str = "list_models"  # ``list_models`` (Anthropic) or ``fixed`` (Cerebras/Groq)


class ActivationDLQEntry(BaseModel):
    """Dead-letter queue entry written when a call fails permanently (AC8)."""
    skill_id: Optional[str] = None
    question_id: Optional[str] = None
    last_request_id: str = ""
    provider: str = ""
    error_class: str = ""
    error_message: str = ""
    attempt_count: int = 0
    ts: datetime = Field(default_factory=datetime.utcnow)
