"""
Data models used by the semiconductor intelligence agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TechEvent:
    """Input event for the agent."""

    title: str
    content: str
    source_type: str
    source: str
    published_at: str
    company_hint: str = ""
    url: str = ""
    source_id: str = ""
    year: int | None = None


@dataclass
class TechContext:
    """RAG context gathered for an event."""

    event: TechEvent
    as_of: str = ""
    related_chunks: list[dict] = field(default_factory=list)
    context_text: str = ""
    analysis_inputs: dict = field(default_factory=dict)


@dataclass
class TechEvaluation:
    """Technical evaluation result."""

    innovation_score: int
    trl: int
    reproducibility: str
    prior_art_exists: bool
    key_claims: list[str] = field(default_factory=list)
    summary: str = ""
    as_of: str = ""


@dataclass
class CompetitiveMap:
    """Competitive mapping result."""

    beneficiaries: list[dict] = field(default_factory=list)
    threats: list[dict] = field(default_factory=list)
    ripple_effects: list[str] = field(default_factory=list)
    competitive_summary: str = ""
    as_of: str = ""


@dataclass
class ExpectedValue:
    """Expected value assessment."""

    p_realization: float
    p_benefit: float
    impact_magnitude: str
    time_horizon: str
    ev_score: str
    rationale: str = ""
    as_of: str = ""


@dataclass
class MarketCheck:
    """Market reaction check."""

    already_priced_in: bool | None
    price_change_pct: float | None
    signal: str
    note: str = ""
    time_horizon: str = ""
    company_moves: list[dict] = field(default_factory=list)
    thesis: str = ""
    as_of: str = ""
    analysis_status: str = "unknown"
    limitations: list[str] = field(default_factory=list)


@dataclass
class IntelligenceReport:
    """Final output report."""

    event: TechEvent
    context: TechContext
    evaluation: TechEvaluation
    competitive: CompetitiveMap
    ev: ExpectedValue
    market: MarketCheck
    as_of: str = ""
    headline: str = ""
    full_report: str = ""


@dataclass
class TechnologyPotentialAssessment:
    """Structured technology potential assessment."""

    topic: str
    as_of: str = ""
    company_hint: str = ""
    domain_hint: str = ""
    catalyst_imminence: dict = field(default_factory=dict)
    bottleneck: dict = field(default_factory=dict)
    company_impact: list[dict] = field(default_factory=list)
    evidence_quality: dict = field(default_factory=dict)
    novelty: dict = field(default_factory=dict)
    revenue_linkage: dict = field(default_factory=dict)
    market_transmission_speed: dict = field(default_factory=dict)
    longevity: dict = field(default_factory=dict)
    overall_thesis: str = ""
    red_flags: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)
    recommendation: str = ""
    confidence: float = 0.0
    reasoning_confidence: float = 0.0
    reasoning_breakdown: dict = field(default_factory=dict)
    supporting_evidence: list[dict] = field(default_factory=list)
    evidence_pack: dict = field(default_factory=dict)
