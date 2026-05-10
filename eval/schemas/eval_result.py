"""Pydantic schemas for evaluation results."""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DimensionScore:
    """A single dimension score for one option."""
    dimension: str
    key: str
    label: str
    score: int


@dataclass
class OptionResult:
    """Evaluation result for a single option (A or B)."""
    option: str
    scores: dict[str, DimensionScore] = field(default_factory=dict)
    total: int = 0

    def get_score(self, dim_id: str) -> int | None:
        d = self.scores.get(dim_id)
        return d.score if d else None


@dataclass
class EvalResult:
    """Complete pairwise evaluation result after aggregation."""

    query_id: str
    scores_a: dict[str, int] = field(default_factory=dict)
    scores_b: dict[str, int] = field(default_factory=dict)
    overall_preference: str = "tie"
    decisive_factor: str = ""
    a_strengths: str = ""
    a_weaknesses: str = ""
    b_strengths: str = ""
    b_weaknesses: str = ""
    hong_kong_specific_flags: list[str] = field(default_factory=list)
    failure_mode_flags: list[str] = field(default_factory=list)
    low_confidence_flags: list[str] = field(default_factory=list)
    d8_note: str | None = None
    veto_triggered: bool = False
    veto_reason: str = ""

    @property
    def total_a(self) -> int:
        return sum(self.scores_a.values())

    @property
    def total_b(self) -> int:
        return sum(self.scores_b.values())

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "scores": {
                "A": {**self.scores_a, "total": self.total_a},
                "B": {**self.scores_b, "total": self.total_b},
            },
            "overall_preference": self.overall_preference,
            "reasoning": {
                "decisive_factor": self.decisive_factor,
                "A_strengths": self.a_strengths,
                "A_weaknesses": self.a_weaknesses,
                "B_strengths": self.b_strengths,
                "B_weaknesses": self.b_weaknesses,
            },
            "flags": {
                "hong_kong_specific": self.hong_kong_specific_flags,
                "failure_mode": self.failure_mode_flags,
                "low_confidence": self.low_confidence_flags,
            },
            "d8_note": self.d8_note,
            "veto": {
                "triggered": self.veto_triggered,
                "reason": self.veto_reason,
            },
        }
