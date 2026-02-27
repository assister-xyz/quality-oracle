"""Pydantic models for Quality Oracle data."""
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


# Request models
class EvaluateRequest(BaseModel):
    target_url: str
    target_type: TargetType = TargetType.MCP_SERVER
    level: EvalLevel = EvalLevel.FUNCTIONAL
    domains: List[str] = []
    webhook_url: Optional[str] = None
    callback_secret: Optional[str] = None


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


class EvaluationStatus(BaseModel):
    evaluation_id: str
    status: EvalStatus
    progress_pct: int = 0
    score: Optional[int] = None
    tier: Optional[str] = None
    evaluation_version: Optional[str] = None
    report: Optional[Dict[str, Any]] = None
    attestation_jwt: Optional[str] = None
    badge_url: Optional[str] = None
    result: Optional[ScoreResponse] = None
    error: Optional[str] = None


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
