"""
评估运行脚本 — 用新 eval 框架对比两个实验组的输出质量

用法:
    # 所有查询，workflow_full vs baseline_single_llm
    python -m eval.run_eval --exp-a workflow_full --exp-b baseline_single_llm --model glm-5.1 --judge glm --query-index 0
    
    python -m eval.run_eval --exp-a workflow_full --exp-b baseline_single_llm

    # 指定模型和单个查询
    python -m eval.run_eval --exp-a workflow_full --exp-b no_rag --model glm-5.1 --judge glm --query-index 0

    # 只跑前5个查询
    python -m eval.run_eval --exp-a workflow_full --exp-b plan_execute_full --limit 5
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from experiments.exp_config import TEST_QUERIES

# 实验输出目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = (_PROJECT_ROOT / "outputtt" / "experiments").resolve()
EVAL_OUTPUT_DIR = Path(__file__).resolve().parent / "eval_output"


def load_case_content(exp_name: str, query: str, query_index: int) -> str | None:
    """加载某个实验组的最终报告（仅最终报告，不含分阶段中间产物）

    文件命名规则（按优先级）:
      1. query_{i}_{suffix}.md   (如 query_1_workflow.md / query_1_baseline.md)
      2. final_report.md         (fallback)
    """
    safe_query = re.sub(r'[\\/:*?"<>|]', '_', query[:20])
    run_dir = EXPERIMENTS_DIR / exp_name / f"query_{query_index + 1}_{safe_query}"

    if not run_dir.exists():
        print(f"  ⚠ 目录不存在: {run_dir}")
        return None

    # 优先找 query_*_*.md，找不到则 fallback 到 final_report.md
    query_files = sorted(run_dir.glob(f"query_{query_index + 1}_*.md"))
    if query_files:
        report_path = query_files[0]
    else:
        report_path = run_dir / "final_report.md"

    if not report_path.exists():
        print(f"  ⚠ 未找到报告: {run_dir} (query_*.md 或 final_report.md)")
        return None

    content = report_path.read_text(encoding="utf-8")
    print(f"  📄 [{exp_name}] {report_path.name} ({len(content)} chars)")
    return content


def get_judge(model: str, judge_type: str):
    """根据类型创建 Judge 实例"""
    if judge_type == "deepseek":
        from eval.judges.deepseek_judge import DeepSeekJudge
        api_key = config.DEEPSEEK_API_KEY
        base_url = config.DEEPSEEK_API_BASE
        return DeepSeekJudge(model=model, api_key=api_key, base_url=base_url)
    elif judge_type == "glm":
        from eval.judges.glm_judge import GLMJudge
        api_key = config.GLM_API_KEY
        base_url = config.GLM_BASE_URL
        return GLMJudge(model=model, api_key=api_key, base_url=base_url)
    else:
        raise ValueError(f"不支持的 judge 类型: {judge_type}")


def run_pairwise_evaluation(
    exp_a: str,
    exp_b: str,
    judge_type: str = "deepseek",
    model: str = "deepseek-chat",
    query_indexes: list[int] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """运行两个实验组的逐对评估"""

    judge = get_judge(model, judge_type)
    from eval.runners.pairwise_runner import PairwiseRunner
    runner = PairwiseRunner(judge)

    comparison_name = f"{exp_a}_VS_{exp_b}"
    output_dir = EVAL_OUTPUT_DIR / comparison_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if query_indexes is None:
        query_indexes = list(range(len(TEST_QUERIES)))
    if limit is not None:
        query_indexes = query_indexes[:limit]

    # 校验 query_index 越界
    max_idx = len(TEST_QUERIES) - 1
    for qi in query_indexes:
        if qi < 0 or qi > max_idx:
            raise ValueError(f"query-index {qi} 越界，有效范围: 0-{max_idx}")

    results = []
    total = len(query_indexes)

    for idx, qi in enumerate(query_indexes):
        query = TEST_QUERIES[qi]
        query_id = f"Q{qi + 1:03d}"

        print(f"\n[{idx + 1}/{total}] {query_id} | {query[:40]}...")

        content_a = load_case_content(exp_a, query, qi)
        content_b = load_case_content(exp_b, query, qi)

        if not content_a or not content_b:
            print(f"  ❌ 跳过：缺少报告内容 (A={bool(content_a)}, B={bool(content_b)})")
            continue

        start = time.time()
        try:
            result = runner.run(
                query_id=query_id,
                query=query,
                option_a_content=content_a,
                option_b_content=content_b,
            )
            elapsed = time.time() - start

            # result 已是扁平格式，直接追加元数据
            eval_record = {
                "query_id": query_id,
                "query": query,
                "exp_a": exp_a,
                "exp_b": exp_b,
                "scores": result.get("scores", {}),
                "reasoning_chain": result.get("reasoning_chain", {}),
                "overall_preference": result.get("overall_preference", "N/A"),
                "reasoning": result.get("reasoning", {}),
                "hong_kong_specific_flags": result.get("hong_kong_specific_flags", []),
                "failure_mode_flags": result.get("failure_mode_flags", []),
                "low_confidence_flags": result.get("low_confidence_flags", []),
                "d8_note": result.get("d8_note"),
                "content_verification": result.get("content_verification", {}),
                "elapsed_seconds": round(elapsed, 1),
            }

            output_path = output_dir / f"{query_id}_OPTION_A_{exp_a}_VS_OPTION_B_{exp_b}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(eval_record, f, ensure_ascii=False, indent=2)

            results.append(eval_record)

            pref = result.get("overall_preference", "N/A")
            total_a = result.get("scores", {}).get("A", {}).get("total", "?")
            total_b = result.get("scores", {}).get("B", {}).get("total", "?")
            d8_note = result.get("d8_note")
            veto_tag = "🚫VETO" if d8_note and pref != "tie" else ""
            print(f"  ✅ 偏好={pref} {veto_tag} | A总分={total_a} B总分={total_b} | {elapsed:.0f}s")
            if d8_note:
                print(f"     d8_note: {d8_note}")

        except Exception as e:
            elapsed = time.time() - start
            print(f"  ❌ 评估失败 ({elapsed:.0f}s): {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'═' * 50}")
    print(f"完成: {len(results)}/{total} 个查询评估成功")
    print(f"结果目录: {output_dir}")

    # 统计偏好分布
    if results:
        prefs = [r["overall_preference"] for r in results]
        veto_count = sum(1 for r in results if r.get("d8_note"))
        print(f"偏好分布: A={prefs.count('A')} B={prefs.count('B')} tie={prefs.count('tie')}")
        if veto_count:
            print(f"触发D5否决: {veto_count} 次")

    return results


def main():
    parser = argparse.ArgumentParser(description="评估两个实验组的输出质量")
    parser.add_argument("--exp-a", type=str, required=True, help="实验组 A 名称")
    parser.add_argument("--exp-b", type=str, required=True, help="实验组 B 名称")
    parser.add_argument("--judge", type=str, default="deepseek",
                        choices=["deepseek", "glm"], help="使用的 LLM")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                        help="模型名称 (deepseek-chat / glm-4-flash)")
    parser.add_argument("--query-index", type=int, default=None,
                        help="只评估指定查询 (0-based)")
    parser.add_argument("--limit", type=int, default=None,
                        help="最多评估几个查询")
    args = parser.parse_args()

    if args.query_index is not None and args.limit is not None:
        parser.error("--query-index 和 --limit 不能同时使用")

    query_indexes = [args.query_index] if args.query_index is not None else None

    run_pairwise_evaluation(
        exp_a=args.exp_a,
        exp_b=args.exp_b,
        judge_type=args.judge,
        model=args.model,
        query_indexes=query_indexes,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
