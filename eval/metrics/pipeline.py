"""
完整评估管线：
  1. 加载人类打分 Excel  →  case | group | dimension | human_score
  2. 加载 LLM eval_output JSON → 提取每组的 D1-D8 分数
  3. 合并 → 计算 diff  →  Spearman / ICC / 方向一致率 / Bootstrap CI
  4. 输出结果 + 图表

用法：
    # 单次运行
    python -m eval.metrics.pipeline \
        --human "data/human_scores.xlsx" \
        --eval-dir "eval/eval_output/workflow_full_VS_baseline_single_llm"

    # 代码调用
    from eval.metrics.pipeline import MetricsPipeline
    pipe = MetricsPipeline()
    pipe.load_human("data/human_scores.xlsx")
    pipe.load_llm("eval/eval_output/workflow_full_VS_baseline_single_llm")
    pipe.run()
    pipe.report()
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .agreement import compute_spearman, compute_agreement, compute_icc, compute_per_dimension
from .bootstrap import bootstrap_spearman, bootstrap_agreement, bootstrap_icc
from .correlation import dimension_correlation_table, plot_scatter, plot_radar_comparison


class MetricsPipeline:
    """端到端评估指标管线"""

    def __init__(self, output_dir: str | None = None):
        self.human_df: pd.DataFrame | None = None
        self.llm_df: pd.DataFrame | None = None
        self.diff: pd.DataFrame | None = None
        self.results: dict = {}
        self.output_dir = Path(output_dir) if output_dir else None

    # ── 数据加载 ──────────────────────────────────────────────

    def load_human(self, excel_path: str) -> pd.DataFrame:
        """加载人类打分 Excel，格式: case | group | dimension | human_score"""
        self.human_df = pd.read_excel(excel_path)

        required = {"case", "group", "dimension", "human_score"}
        missing = required - set(self.human_df.columns)
        if missing:
            raise ValueError(f"Excel 缺少列: {missing}")

        # 确保 group 列的值规整为 workflow / baseline
        group_map = {}
        for g in self.human_df["group"].unique():
            gl = str(g).lower()
            if "workflow" in gl:
                group_map[g] = "workflow"
            elif "baseline" in gl or "base" in gl:
                group_map[g] = "baseline"
        if group_map:
            self.human_df["group"] = self.human_df["group"].map(group_map).fillna(self.human_df["group"])

        print(f"✅ 加载人类数据: {len(self.human_df)} 行, "
              f"{self.human_df['case'].nunique()} 个 case, "
              f"{self.human_df['dimension'].nunique()} 个维度")
        return self.human_df

    def load_llm(self, eval_dir: str) -> pd.DataFrame:
        """从 eval_output 目录加载 LLM 打分结果，提取为 llm_score 列
        
        评估 JSON 的 scores 结构:
            {"A": {"D1_precision": 5, ...}, "B": {"D2_scenario": 3, ...}}
        
        A = workflow 组, B = baseline 组（假设 A=workflow, B=baseline）
        展开为: case | group | dimension | llm_score
        """
        eval_path = Path(eval_dir)
        if not eval_path.exists():
            raise FileNotFoundError(f"评估目录不存在: {eval_dir}")

        rows = []
        for json_file in sorted(eval_path.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            query_id = data.get("query_id", json_file.stem[:4])
            scores = data.get("scores", {})

            # D1-D8 的 key 映射
            dim_keys = {
                "D1": "D1_precision", "D2": "D2_scenario",
                "D3": "D3_source", "D4": "D4_timeliness",
                "D5": "D5_regulatory", "D6": "D6_local",
                "D7": "D7_transfer", "D8": "D8_inspiration",
            }

            for group_name, option_key in [("workflow", "A"), ("baseline", "B")]:
                opt_scores = scores.get(option_key, {})
                for dim_id, key in dim_keys.items():
                    score = opt_scores.get(key)
                    if score is not None:
                        rows.append({
                            "case": query_id,
                            "group": group_name,
                            "dimension": dim_id,
                            "llm_score": score,
                        })

        self.llm_df = pd.DataFrame(rows)
        print(f"✅ 加载 LLM 数据: {len(self.llm_df)} 行, "
              f"{self.llm_df['case'].nunique()} 个 case")
        return self.llm_df

    # ── 合并 & 计算 diff ─────────────────────────────────────

    def merge_and_compute_diff(self) -> pd.DataFrame:
        """合并人类和 LLM 数据，计算 workflow - baseline 的逐对差异"""
        if self.human_df is None or self.llm_df is None:
            raise ValueError("请先调用 load_human() 和 load_llm()")

        # 对齐 case 名称
        self._align_case_names()

        # 合并
        merged = self.human_df.merge(
            self.llm_df, on=["case", "group", "dimension"], how="inner"
        )
        print(f"📊 合并后: {len(merged)} 行 (人类 {len(self.human_df)}, LLM {len(self.llm_df)})")

        if merged.empty:
            print("⚠ 合并后为空，请检查 case/group/dimension 的对齐情况")
            print("人类 cases:", sorted(self.human_df["case"].unique()))
            print("LLM cases:", sorted(self.llm_df["case"].unique()))
            return pd.DataFrame()

        # 计算 diff: workflow - baseline
        diff_list = []
        for group_name, group_df in merged.groupby("group"):
            cases = group_df["case"].unique()
            if len(cases) != 2:
                raise ValueError(f"group '{group_name}' 有 {len(cases)} 个 case (需要 2)")

            # 找 workflow 和 baseline
            wf_cases = [c for c in cases if c in merged[merged["group"] == group_name]["case"].unique()]
            # 实际上 merged 已按 group 分组，这里 group_name 已经是 workflow/baseline 这类名
            # 不对，这里 group 列的值是 workflow/baseline，不是 case
            pass

        # 正确逻辑：按每个原始 case（如 Q001）分组，计算该 case 的 workflow vs baseline diff
        diff_rows = []
        for case_name, case_df in merged.groupby("case"):
            wf = case_df[case_df["group"] == "workflow"].set_index("dimension")
            bl = case_df[case_df["group"] == "baseline"].set_index("dimension")

            if len(wf) != 8 or len(bl) != 8:
                print(f"  ⚠ {case_name}: workflow={len(wf)}维, baseline={len(bl)}维 → 跳过")
                continue

            case_b_name = "baseline"  # 对照组始终是 baseline
            for dim in wf.index:
                diff_rows.append({
                    "case": case_name,
                    "dimension": dim,
                    "llm_diff": wf.loc[dim, "llm_score"] - bl.loc[dim, "llm_score"],
                    "human_diff": wf.loc[dim, "human_score"] - bl.loc[dim, "human_score"],
                })

        self.diff = pd.DataFrame(diff_rows)

        if self.diff.empty:
            print("⚠ 差异数据为空")
        else:
            print(f"📊 差异数据: {len(self.diff)} 行 ({self.diff['case'].nunique()} 个 case)")
        return self.diff

    def _align_case_names(self):
        """尝试对齐人类和 LLM 数据的 case 名称"""
        if self.human_df is None or self.llm_df is None:
            return

        human_cases = set(self.human_df["case"].unique())
        llm_cases = set(self.llm_df["case"].unique())

        common = human_cases & llm_cases
        if common:
            return  # 已有交集，无需对齐

        human_only = human_cases - llm_cases
        llm_only = llm_cases - human_cases

        if human_only and llm_only:
            print(f"  🔄 尝试对齐 case 名: human={human_only}, llm={llm_only}")
            # 如果双方数量一致，尝试按字母序一一对应
            if len(human_only) == len(llm_only):
                mapping = dict(zip(sorted(llm_only), sorted(human_only)))
                self.llm_df["case"] = self.llm_df["case"].map(mapping).fillna(self.llm_df["case"])
                print(f"  对齐映射: {mapping}")

    # ── 运行全流程 ──────────────────────────────────────────

    def run(self) -> dict:
        """运行完整评估管线"""
        diff = self.merge_and_compute_diff()
        if diff is None or diff.empty:
            return {"error": "差异数据为空"}

        print("\n" + "=" * 60)
        print("📊 评估指标计算")
        print("=" * 60)

        # 1. Spearman
        spr = compute_spearman(diff)
        print(f"\nSpearman rho = {spr['rho']:.4f} (p = {spr['p_value']:.4f}) {'✅' if spr['significant'] else '❌'}")

        # 2. 方向一致率
        agr = compute_agreement(diff)
        print(f"方向一致率 = {agr['agreement_rate']:.2%} ({agr['n_total']} 样本)")

        # 3. ICC
        icc = compute_icc(diff)
        print(f"ICC(A,k) = {icc['icc']:.4f} (p = {icc['p_value']:.4f})")

        # 4. 按维度
        per_dim = compute_per_dimension(diff)
        print(f"\n--- 按维度 ---")
        for _, row in per_dim.iterrows():
            print(f"  {row['dimension']}: rho={row['spearman_rho']:.3f} (p={row['spearman_p']:.4f}), 一致率={row['agreement_rate']:.2%}")

        # 5. Bootstrap CI
        bs_spr = bootstrap_spearman(diff)
        print(f"\nSpearman Bootstrap: {bs_spr['mean']:.3f} [95% CI: {bs_spr['ci_lower']:.3f}, {bs_spr['ci_upper']:.3f}]")

        bs_agr = bootstrap_agreement(diff)
        print(f"Agreement Bootstrap: {bs_agr['mean']:.2%} [95% CI: {bs_agr['ci_lower']:.2%}, {bs_agr['ci_upper']:.2%}]")

        bs_icc = bootstrap_icc(diff)
        print(f"ICC Bootstrap: {bs_icc['mean']:.3f} [95% CI: {bs_icc['ci_lower']:.3f}, {bs_icc['ci_upper']:.3f}]")

        self.results = {
            "spearman": spr,
            "agreement": agr,
            "icc": icc,
            "per_dimension": per_dim.to_dict("records"),
            "bootstrap_spearman": bs_spr,
            "bootstrap_agreement": bs_agr,
            "bootstrap_icc": bs_icc,
        }

        # 生成图表
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._save_plots(diff)

        return self.results

    def _save_plots(self, diff: pd.DataFrame):
        """保存可视化图表"""
        try:
            scatter_path = self.output_dir / "scatter_llm_vs_human.png"
            plot_scatter(diff, str(scatter_path))
            print(f"\n📊 散点图: {scatter_path}")
        except Exception as e:
            print(f"  ⚠ 散点图失败: {e}")

    def report(self) -> str:
        """生成文本报告"""
        if not self.results:
            return "请先调用 run()"

        r = self.results
        lines = [
            "=" * 50,
            "评估指标报告",
            "=" * 50,
            f"Spearman rho: {r['spearman']['rho']:.4f} (p={r['spearman']['p_value']:.4f}) {'✅显著' if r['spearman']['significant'] else '❌不显著'}",
            f"方向一致率:  {r['agreement']['agreement_rate']:.2%} (n={r['agreement']['n_total']})",
            f"ICC(A,k):     {r['icc']['icc']:.4f} (p={r['icc']['p_value']:.4f})",
            "",
            "Bootstrap 95% CI:",
            f"  Spearman:   {r['bootstrap_spearman']['mean']:.3f} [{r['bootstrap_spearman']['ci_lower']:.3f}, {r['bootstrap_spearman']['ci_upper']:.3f}]",
            f"  Agreement:  {r['bootstrap_agreement']['mean']:.2%} [{r['bootstrap_agreement']['ci_lower']:.2%}, {r['bootstrap_agreement']['ci_upper']:.2%}]",
            f"  ICC:        {r['bootstrap_icc']['mean']:.3f} [{r['bootstrap_icc']['ci_lower']:.3f}, {r['bootstrap_icc']['ci_upper']:.3f}]",
            "",
            "按维度:",
        ]
        for dim_row in r["per_dimension"]:
            lines.append(
                f"  {dim_row['dimension']}: rho={dim_row['spearman_rho']:.3f} "
                f"(p={dim_row['spearman_p']:.4f}), 一致率={dim_row['agreement_rate']:.2%}"
            )
        return "\n".join(lines)

    def save_results(self, path: str | None = None):
        """保存结果 JSON"""
        p = path or (self.output_dir / "metrics_results.json" if self.output_dir else "metrics_results.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        print(f"💾 结果已保存: {p}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="运行完整评估管线")
    parser.add_argument("--human", type=str, required=True, help="人类打分 Excel 路径")
    parser.add_argument("--eval-dir", type=str, required=True, help="LLM 评估结果目录")
    parser.add_argument("--output-dir", type=str, default="eval/eval_output/metrics", help="输出目录")
    args = parser.parse_args()

    pipe = MetricsPipeline(output_dir=args.output_dir)
    pipe.load_human(args.human)
    pipe.load_llm(args.eval_dir)
    pipe.run()
    print("\n" + pipe.report())
    pipe.save_results()
