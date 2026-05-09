"""
实验结果分析 — 生成论文用的表格和对比图
"""

import os
import json
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUT_DIR

RESULTS_DIR = os.path.join(OUTPUT_DIR, "experiments")


def load_results():
    """从标准目录结构加载所有实验结果"""
    results = []
    if not os.path.exists(RESULTS_DIR):
        return results

    for exp_name in sorted(os.listdir(RESULTS_DIR)):
        exp_dir = os.path.join(RESULTS_DIR, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        for query_dir_name in sorted(os.listdir(exp_dir)):
            query_dir = os.path.join(exp_dir, query_dir_name)
            meta_path = os.path.join(query_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
    return results


def generate_comparison_table(results):
    """生成主实验对比表"""
    rows = []
    for r in results:
        if r["status"] != "success":
            continue
        m = r.get("metrics", {})
        rows.append({
            "实验组": r["exp_name"],
            "查询": r["query"][:20],
            "模式": r["config"]["mode"],
            "耗时(s)": round(m.get("elapsed_seconds", 0)),
            "报告长度": m.get("report_length", 0),
            "评估分数": m.get("evaluation_score", "N/A"),
        })

    df = pd.DataFrame(rows)
    print("\n📊 实验对比表:")
    print(df.to_string(index=False))

    # 按实验组汇总平均值
    summary = df.groupby("实验组").agg({
        "耗时(s)": "mean",
        "报告长度": "mean",
        "评估分数": lambda x: pd.to_numeric(x, errors="coerce").mean(),
    }).round(1)

    print("\n📊 各实验组平均指标:")
    print(summary.to_string())

    # 保存为 CSV
    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    summary.to_csv(csv_path)
    print(f"\n💾 已保存: {csv_path}")

    return summary


def generate_ablation_table(results):
    """生成消融实验表"""
    ablation_exps = [r for r in results
                     if r["exp_name"].startswith("ablation") or r["exp_name"] == "workflow_full"]

    if not ablation_exps:
        print("没有消融实验数据")
        return

    rows = []
    for r in ablation_exps:
        if r["status"] != "success":
            continue
        m = r.get("metrics", {})
        cfg = r["config"]
        rows.append({
            "配置": r["exp_name"],
            "混合检索": "✓" if cfg.get("use_hybrid_retrieval", True) else "✗",
            "深度研究": "✓" if cfg.get("use_deep_research", True) else "✗",
            "差异分析": "✓" if cfg.get("use_gap_analysis", True) else "✗",
            "耗时(s)": round(m.get("elapsed_seconds", 0)),
            "报告长度": m.get("report_length", 0),
            "评估分数": m.get("evaluation_score", "N/A"),
        })

    df = pd.DataFrame(rows)
    summary = df.groupby(["配置", "混合检索", "深度研究", "差异分析"]).agg({
        "耗时(s)": "mean",
        "报告长度": "mean",
        "评估分数": lambda x: pd.to_numeric(x, errors="coerce").mean(),
    }).round(1)

    print("\n📊 消融实验表:")
    print(summary.to_string())

    csv_path = os.path.join(RESULTS_DIR, "ablation_table.csv")
    summary.to_csv(csv_path)
    print(f"\n💾 已保存: {csv_path}")


if __name__ == "__main__":
    results = load_results()
    print(f"加载了 {len(results)} 条实验结果")

    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] != "success")
    print(f"成功: {success} | 失败: {failed}")

    generate_comparison_table(results)
    generate_ablation_table(results)