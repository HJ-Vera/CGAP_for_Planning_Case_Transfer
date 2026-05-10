"""D5 one-vote veto policy.

D5 veto rule: If one option's D5 score is ≤ 2 and the other's is > 3,
the overall_preference MUST be the option with the higher D5 score,
regardless of total scores.
"""

from dataclasses import dataclass


@dataclass
class VetoResult:
    triggered: bool
    forced_preference: str | None = None
    reason: str = ""


def apply_d5_veto(d5_a: int, d5_b: int) -> VetoResult:
    """Apply D5 one-vote veto rule.

    Args:
        d5_a: D5 score for Option A
        d5_b: D5 score for Option B

    Returns:
        VetoResult with triggered flag and forced preference if applicable.
    """
    if d5_a <= 2 and d5_b > 3:
        return VetoResult(
            triggered=True,
            forced_preference="B",
            reason=(
                f"D5法规适配性障碍：方案A(D5={d5_a})核心机制在香港无法定对应路径，"
                f"方案B(D5={d5_b})有法定对应。根据D5一票否决规则，overall_preference必须选择B。"
            ),
        )
    if d5_b <= 2 and d5_a > 3:
        return VetoResult(
            triggered=True,
            forced_preference="A",
            reason=(
                f"D5法规适配性障碍：方案B(D5={d5_b})核心机制在香港无法定对应路径，"
                f"方案A(D5={d5_a})有法定对应。根据D5一票否决规则，overall_preference必须选择A。"
            ),
        )
    return VetoResult(triggered=False)


def check_d8_no_override_d5(d8: int, d5: int) -> bool:
    """Check if D8 > D5 rule is triggered (D8 ≥ 4 but D5 ≤ 2).

    Returns True if d8_note must be added and overall_preference must not
    select this option (unless the other option also has D5 ≤ 2).
    """
    return d8 >= 4 and d5 <= 2
