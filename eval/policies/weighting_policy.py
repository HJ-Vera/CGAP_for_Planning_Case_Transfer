"""Weighting policy for adjusting dimension weights.

Currently uses equal weighting (all 8 dimensions have equal weight in total score).
Future extensions: objective/subjective weighting, query-type-specific weighting, etc.
"""

from dataclasses import dataclass, field


@dataclass
class WeightConfig:
    """Configuration for dimension weights."""
    dimension: str
    weight: float = 1.0


DEFAULT_WEIGHTS: dict[str, float] = {
    "D1": 1.0,
    "D2": 1.0,
    "D3": 1.0,
    "D4": 1.0,
    "D5": 1.0,
    "D6": 1.0,
    "D7": 1.0,
    "D8": 1.0,
}


def apply_weights(scores: dict[str, int], weights: dict[str, float] | None = None) -> dict[str, float]:
    """Apply optional weights to dimension scores.

    Args:
        scores: Raw dimension scores (e.g. {"D1": 4, "D2": 3, ...})
        weights: Optional per-dimension weight mapping. Defaults to equal weighting.

    Returns:
        Weighted scores dict with same keys.
    """
    w = weights or DEFAULT_WEIGHTS
    return {dim: scores.get(dim, 0) * w.get(dim, 1.0) for dim in scores}


def weighted_total(scores: dict[str, int], weights: dict[str, float] | None = None) -> float:
    """Calculate weighted total score."""
    weighted = apply_weights(scores, weights)
    return sum(weighted.values())
