"""Aggregation policy for combining objective and subjective skill results.

Priority chain: D5 > D6 > D7 > D8 > D1 > D2 > D3 > D4

When D5 is tied, D6 decides; when D6 is tied, D7 decides; etc.
"""

from dataclasses import dataclass, field
from .veto_policy import apply_d5_veto, check_d8_no_override_d5


@dataclass
class AggregationResult:
    overall_preference: str
    decisive_factor: str
    a_strengths: list[str] = field(default_factory=list)
    a_weaknesses: list[str] = field(default_factory=list)
    b_strengths: list[str] = field(default_factory=list)
    b_weaknesses: list[str] = field(default_factory=list)
    hong_kong_specific_flags: list[str] = field(default_factory=list)
    failure_mode_flags: list[str] = field(default_factory=list)
    low_confidence_flags: list[str] = field(default_factory=list)
    d8_note: str | None = None
    veto_triggered: bool = False
    veto_reason: str = ""


DIMENSION_LABELS = {
    "D1": "检索精准性",
    "D2": "场景匹配度",
    "D3": "来源可靠性",
    "D4": "时效性",
    "D5": "法规适配性",
    "D6": "本地语境深度",
    "D7": "可迁移性",
    "D8": "启发性价值",
}

# score_key → D-ID mapping (e.g. "D1_precision" → "D1")
SCORE_KEY_TO_DIM = {
    "D1_precision": "D1", "D2_scenario": "D2",
    "D3_source": "D3", "D4_timeliness": "D4",
    "D5_regulatory": "D5", "D6_local": "D6",
    "D7_transfer": "D7", "D8_inspiration": "D8",
}

PRIORITY_CHAIN = ["D5", "D6", "D7", "D8", "D1", "D2", "D3", "D4"]


def _fmt_score(score_key: str, value: int) -> str:
    """Format a score key+value to human-readable label, e.g. D1_precision=5 → 检索精准性(5分)"""
    dim_id = SCORE_KEY_TO_DIM.get(score_key, score_key)
    label = DIMENSION_LABELS.get(dim_id, dim_id)
    return f"{label}({value}分)"


def aggregate(
    obj_scores_a: dict[str, int],
    obj_scores_b: dict[str, int],
    subj_scores_a: dict[str, int],
    subj_scores_b: dict[str, int],
    hk_flags_obj: list[str] | None = None,
    hk_flags_subj: list[str] | None = None,
    failure_flags_obj: list[str] | None = None,
    failure_flags_subj: list[str] | None = None,
    low_conf_obj: list[str] | None = None,
    low_conf_subj: list[str] | None = None,
    subj_d8_note: str | None = None,
) -> AggregationResult:
    """Aggregate objective and subjective skill results into a final preference.

    Args:
        obj_scores_a: Objective skill scores for Option A (keyed by D1_precision, etc.)
        obj_scores_b: Objective skill scores for Option B
        subj_scores_a: Subjective skill scores for Option A (keyed by D5_regulatory, etc.)
        subj_scores_b: Subjective skill scores for Option B
        hk_flags_obj: Hong Kong specific flags from objective skill
        hk_flags_subj: Hong Kong specific flags from subjective skill
        failure_flags_obj: Failure mode flags from objective skill
        failure_flags_subj: Failure mode flags from subjective skill
        low_conf_obj: Low confidence flags from objective skill
        low_conf_subj: Low confidence flags from subjective skill
        subj_d8_note: D8 note from subjective skill

    Returns:
        AggregationResult with final overall_preference and reasoning.
    """

    def _extract_dim(scores: dict, dim_id: str) -> int:
        key_map = {
            "D1": "D1_precision", "D2": "D2_scenario", "D3": "D3_source",
            "D4": "D4_timeliness", "D5": "D5_regulatory", "D6": "D6_local",
            "D7": "D7_transfer", "D8": "D8_inspiration",
        }
        return scores.get(key_map.get(dim_id, ""), 0)

    all_a = {**obj_scores_a, **subj_scores_a}
    all_b = {**obj_scores_b, **subj_scores_b}

    d5_a = _extract_dim(all_a, "D5")
    d5_b = _extract_dim(all_b, "D5")
    total_a = sum(all_a.get(k, 0) for k in all_a)
    total_b = sum(all_b.get(k, 0) for k in all_b)

    veto = apply_d5_veto(d5_a, d5_b)
    if veto.triggered:
        d8_a = _extract_dim(all_a, "D8")
        d8_b = _extract_dim(all_b, "D8")
        d8_notes = []
        if check_d8_no_override_d5(d8_a, d5_a):
            d8_notes.append(subj_d8_note or "方案A高启发性案例，但存在D5法规障碍，建议仅作概念参考")
        if check_d8_no_override_d5(d8_b, d5_b):
            d8_notes.append(subj_d8_note or "方案B高启发性案例，但存在D5法规障碍，建议仅作概念参考")

        return AggregationResult(
            overall_preference=veto.forced_preference or "tie",
            decisive_factor=veto.reason,
            veto_triggered=True,
            veto_reason=veto.reason,
            d8_note="; ".join(d8_notes) if d8_notes else None,
            hong_kong_specific_flags=(hk_flags_obj or []) + (hk_flags_subj or []),
            failure_mode_flags=(failure_flags_obj or []) + (failure_flags_subj or []),
            low_confidence_flags=(low_conf_obj or []) + (low_conf_subj or []),
        )

    preference = "tie"
    decisive_dim = ""
    for dim_id in PRIORITY_CHAIN:
        score_a = _extract_dim(all_a, dim_id)
        score_b = _extract_dim(all_b, dim_id)
        if score_a > score_b:
            preference = "A"
            decisive_dim = dim_id
            break
        elif score_b > score_a:
            preference = "B"
            decisive_dim = dim_id
            break

    label = DIMENSION_LABELS.get(decisive_dim, decisive_dim)
    decisive_factor = (
        f"按维度优先级规则(D5>D6>D7>D8>D1>D2>D3>D4)，"
        f"{label}({decisive_dim})成为决定性维度"
    )
    if preference == "A":
        decisive_factor += (
            f"：方案A({_extract_dim(all_a, decisive_dim)}分) "
            f"高于方案B({_extract_dim(all_b, decisive_dim)}分)。"
        )
    elif preference == "B":
        decisive_factor += (
            f"：方案B({_extract_dim(all_b, decisive_dim)}分) "
            f"高于方案A({_extract_dim(all_a, decisive_dim)}分)。"
        )
    else:
        decisive_factor = "所有维度分数均相同，无法辨别明确方向。"

    strengths_a = [_fmt_score(k, v) for k, v in all_a.items() if v >= 4]
    weaknesses_a = [_fmt_score(k, v) for k, v in all_a.items() if v <= 2]
    strengths_b = [_fmt_score(k, v) for k, v in all_b.items() if v >= 4]
    weaknesses_b = [_fmt_score(k, v) for k, v in all_b.items() if v <= 2]

    return AggregationResult(
        overall_preference=preference,
        decisive_factor=decisive_factor,
        a_strengths=strengths_a,
        a_weaknesses=weaknesses_a,
        b_strengths=strengths_b,
        b_weaknesses=weaknesses_b,
        hong_kong_specific_flags=(hk_flags_obj or []) + (hk_flags_subj or []),
        failure_mode_flags=(failure_flags_obj or []) + (failure_flags_subj or []),
        low_confidence_flags=(low_conf_obj or []) + (low_conf_subj or []),
        d8_note=subj_d8_note,
    )
