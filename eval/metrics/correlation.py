"""Correlation and visualization utilities for evaluation results.

Provides per-dimension correlation analysis and scatter plot generation.
"""

import numpy as np
import pandas as pd
from scipy import stats


DIMENSION_LABELS = {
    "D1": "D1 精准度",
    "D2": "D2 场景匹配",
    "D3": "D3 来源可靠",
    "D4": "D4 时效性",
    "D5": "D5 法规兼容",
    "D6": "D6 本地约束",
    "D7": "D7 可迁移性",
    "D8": "D8 启发创新",
}


def dimension_correlation_table(diff: pd.DataFrame) -> pd.DataFrame:
    """Create a per-dimension Spearman + agreement table.

    Args:
        diff: DataFrame from agreement.compute_diffs()

    Returns:
        DataFrame indexed by dimension with spearman_rho, p_value, agreement_rate.
    """
    rows = []
    for dim in sorted(diff["dimension"].unique()):
        sub = diff[diff["dimension"] == dim]
        if len(sub) < 3:
            continue
        rho, p = stats.spearmanr(sub["llm_diff"], sub["human_diff"])
        sign_llm = np.sign(sub["llm_diff"])
        sign_human = np.sign(sub["human_diff"])
        agree = (sign_llm == sign_human).mean()
        rows.append({
            "dimension": dim,
            "label": DIMENSION_LABELS.get(dim, dim),
            "spearman_rho": round(rho, 4),
            "p_value": round(p, 4),
            "agreement_rate": round(agree, 4),
            "n": len(sub),
        })
    return pd.DataFrame(rows).set_index("dimension")


def plot_scatter(
    diff: pd.DataFrame,
    output_path: str | None = None,
    title: str | None = None,
) -> object:
    """Generate a scatter plot of LLM vs Human score differences.

    Args:
        diff: DataFrame with llm_diff and human_diff columns.
        output_path: Optional path to save the figure.
        title: Optional plot title.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    rho, p = stats.spearmanr(diff["llm_diff"], diff["human_diff"])
    agreement = (np.sign(diff["llm_diff"]) == np.sign(diff["human_diff"])).mean()

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.scatterplot(
        data=diff, x="human_diff", y="llm_diff",
        hue="dimension", alpha=0.8, palette="tab10", ax=ax,
    )

    lims = [
        min(diff[["llm_diff", "human_diff"]].min().min(), -0.5),
        max(diff[["llm_diff", "human_diff"]].max().max(), 0.5),
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y = x (完美一致)")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("人类评分差异 (workflow - 对照)")
    ax.set_ylabel("LLM 评分差异 (workflow - 对照)")
    ax.set_title(
        title
        or f"LLM vs Human 评分差异对比\n"
        f"整体 Spearman rho = {rho:.3f}, 方向一致率 = {agreement:.2%}"
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_radar_comparison(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    output_path: str | None = None,
    title: str = "方案A vs 方案B 核心能力雷达图",
) -> object:
    """Generate a radar chart comparing two options across 8 dimensions.

    Args:
        scores_a: Dict of D1-D8 scores for Option A.
        scores_b: Dict of D1-D8 scores for Option B.
        output_path: Optional path to save the figure.
        title: Plot title.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    labels = [DIMENSION_LABELS.get(f"D{i}", f"D{i}") for i in range(1, 9)]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    values_a = [scores_a.get(f"D{i}", 0) for i in range(1, 9)]
    values_a += values_a[:1]
    ax.plot(angles, values_a, "o-", color="#4C72B0", linewidth=2, label="方案 A")
    ax.fill(angles, values_a, color="#4C72B0", alpha=0.25)

    values_b = [scores_b.get(f"D{i}", 0) for i in range(1, 9)]
    values_b += values_b[:1]
    ax.plot(angles, values_b, "s--", color="#DD8452", linewidth=2, label="方案 B")
    ax.fill(angles, values_b, color="#DD8452", alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], color="grey", size=10)

    ax.set_title(title, size=16, pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig
