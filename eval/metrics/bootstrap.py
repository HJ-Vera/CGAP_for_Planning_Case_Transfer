"""Bootstrap confidence intervals for Spearman, Agreement, and ICC metrics."""

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_spearman(diff: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict:
    """Bootstrap 95% CI for Spearman rho.

    Args:
        diff: DataFrame with llm_diff and human_diff columns.
        n_boot: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        dict with keys: mean, ci_lower, ci_upper
    """
    np.random.seed(seed)
    rhos = []
    data = diff[["llm_diff", "human_diff"]].values
    n = len(data)

    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        sample = data[idx]
        rho, _ = stats.spearmanr(sample[:, 0], sample[:, 1])
        rhos.append(rho)

    return {
        "mean": np.mean(rhos),
        "ci_lower": np.percentile(rhos, 2.5),
        "ci_upper": np.percentile(rhos, 97.5),
        "n_boot": n_boot,
    }


def bootstrap_agreement(diff: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict:
    """Bootstrap 95% CI for direction agreement rate.

    Handles zero-difference cases by excluding them from valid comparison.

    Args:
        diff: DataFrame from compute_diffs().
        n_boot: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        dict with keys: mean, ci_lower, ci_upper
    """
    np.random.seed(seed)
    agreements = []
    n = len(diff)

    for _ in range(n_boot):
        sample = diff.sample(n, replace=True)

        sign_llm = np.sign(sample["llm_diff"])
        sign_human = np.sign(sample["human_diff"])

        both_zero = (sample["llm_diff"] == 0) & (sample["human_diff"] == 0)
        valid = ~both_zero

        if valid.sum() == 0:
            continue

        agree = (sign_llm[valid] == sign_human[valid]).mean()
        agreements.append(agree)

    if not agreements:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "n_boot": n_boot}

    return {
        "mean": np.mean(agreements),
        "ci_lower": np.percentile(agreements, 2.5),
        "ci_upper": np.percentile(agreements, 97.5),
        "n_boot": n_boot,
    }


def bootstrap_icc(diff: pd.DataFrame, n_boot: int = 500, seed: int = 42) -> dict:
    """Bootstrap 95% CI for ICC(A,k).

    Uses unique target IDs for each bootstrap iteration to handle
    resampled duplicate rows correctly.

    Args:
        diff: DataFrame from compute_diffs().
        n_boot: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        dict with keys: mean, ci_lower, ci_upper
    """
    import pingouin as pg

    np.random.seed(seed)
    iccs = []

    for _ in range(n_boot):
        sample = diff.sample(len(diff), replace=True).reset_index(drop=True)
        sample["target_id"] = sample.index

        icc_long = sample.melt(
            id_vars="target_id",
            value_vars=["llm_diff", "human_diff"],
            var_name="Rater",
            value_name="Score",
        )

        try:
            icc_res = pg.intraclass_corr(
                data=icc_long,
                targets="target_id",
                raters="Rater",
                ratings="Score",
            )

            icc_row = icc_res[icc_res["Type"] == "ICC(A,k)"]
            if not icc_row.empty:
                iccs.append(icc_row["ICC"].values[0])
        except Exception:
            continue

    if not iccs:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "n_boot": n_boot}

    return {
        "mean": np.mean(iccs),
        "ci_lower": np.percentile(iccs, 2.5),
        "ci_upper": np.percentile(iccs, 97.5),
        "n_boot": n_boot,
    }
