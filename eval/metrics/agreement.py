"""Agreement metrics: Spearman, ICC, and direction agreement between LLM and human scores.

Input data format (Excel/CSV):
    case | group | dimension | human_score

The LLM scores are expected in a similar format or as a parallel DataFrame.
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_diffs(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame | None = None,
    human_score_col: str = "human_score",
    llm_score_col: str = "llm_score",
) -> pd.DataFrame:
    """Compute pairwise differences between workflow and baseline for both human and LLM scores.

    Args:
        human_df: DataFrame with columns [case, group, dimension, human_score]
        llm_df: Optional DataFrame with columns [case, group, dimension, llm_score].
                If None, llm_score_col must exist in human_df.

    Returns:
        DataFrame with columns [group, dimension, case_b, llm_diff, human_diff]
    """
    if llm_df is None:
        if llm_score_col not in human_df.columns:
            raise ValueError(f"Column '{llm_score_col}' not found in dataframe")
        df = human_df.copy()
    else:
        df = human_df.merge(
            llm_df, on=["case", "group", "dimension"], how="inner"
        )
        if "human_score" not in df.columns:
            df.rename(columns={human_score_col + "_x": "human_score"}, inplace=True)

    diff_list = []
    for group_name, group_df in df.groupby("group"):
        cases = group_df["case"].unique()
        if len(cases) != 2:
            raise ValueError(f"Group '{group_name}' has {len(cases)} cases (expected 2)")
        if "workflow" not in cases:
            raise ValueError(f"Group '{group_name}' missing 'workflow' case")

        case_a = "workflow"
        case_b = [c for c in cases if c != "workflow"][0]

        df_a = group_df[group_df["case"] == case_a].set_index("dimension")
        df_b = group_df[group_df["case"] == case_b].set_index("dimension")

        if len(df_a) != 8 or len(df_b) != 8:
            raise ValueError(f"Group '{group_name}' missing dimensions (found {len(df_a)} and {len(df_b)}, expected 8)")

        for dim in df_a.index:
            diff_list.append({
                "group": group_name,
                "dimension": dim,
                "case_b": case_b,
                "llm_diff": df_a.loc[dim, llm_score_col] - df_b.loc[dim, llm_score_col],
                "human_diff": df_a.loc[dim, human_score_col] - df_b.loc[dim, human_score_col],
            })

    return pd.DataFrame(diff_list)


def compute_spearman(diff: pd.DataFrame) -> dict:
    """Compute Spearman rank correlation between LLM and human differences.

    Args:
        diff: DataFrame from compute_diffs() with llm_diff and human_diff columns.

    Returns:
        dict with keys: rho, p_value, significant
    """
    rho, p = stats.spearmanr(diff["llm_diff"], diff["human_diff"])
    return {"rho": rho, "p_value": p, "significant": p < 0.05}


def compute_agreement(diff: pd.DataFrame) -> dict:
    """Compute direction agreement rate between LLM and human score differences.

    The agreement rate measures what proportion of cases where LLM and human
    agree on which option scores higher.

    Args:
        diff: DataFrame from compute_diffs()

    Returns:
        dict with keys: agreement_rate, n_total
    """
    sign_llm = np.sign(diff["llm_diff"])
    sign_human = np.sign(diff["human_diff"])
    agreement = (sign_llm == sign_human).mean()
    return {"agreement_rate": agreement, "n_total": len(diff)}


def compute_icc(diff: pd.DataFrame) -> dict:
    """Compute Intraclass Correlation Coefficient (ICC(A,k)) between LLM and human differences.

    Args:
        diff: DataFrame from compute_diffs()

    Returns:
        dict with keys: icc, p_value, icc_type
    """
    import pingouin as pg

    icc_long = diff.reset_index().melt(
        id_vars="index",
        value_vars=["llm_diff", "human_diff"],
        var_name="Rater",
        value_name="Score",
    )

    icc_results = pg.intraclass_corr(
        data=icc_long,
        targets="index",
        raters="Rater",
        ratings="Score",
    )

    icc_row = icc_results[icc_results["Type"] == "ICC(A,k)"]
    if not icc_row.empty:
        return {
            "icc": icc_row["ICC"].values[0],
            "p_value": icc_row["pval"].values[0],
            "icc_type": "ICC(A,k)",
        }
    return {"icc": float("nan"), "p_value": float("nan"), "icc_type": "ICC(A,k)"}


def compute_per_dimension(diff: pd.DataFrame) -> pd.DataFrame:
    """Compute per-dimension Spearman rho and direction agreement.

    Args:
        diff: DataFrame from compute_diffs()

    Returns:
        DataFrame with columns [dimension, spearman_rho, spearman_p, agreement_rate]
    """
    results = []
    for dim in sorted(diff["dimension"].unique()):
        sub = diff[diff["dimension"] == dim]
        if len(sub) < 3:
            continue
        rho, p = stats.spearmanr(sub["llm_diff"], sub["human_diff"])
        sign_llm = np.sign(sub["llm_diff"])
        sign_human = np.sign(sub["human_diff"])
        agree = (sign_llm == sign_human).mean()
        results.append({
            "dimension": dim,
            "spearman_rho": rho,
            "spearman_p": p,
            "agreement_rate": agree,
        })
    return pd.DataFrame(results)
