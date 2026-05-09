"""
实验运行器 — 按配置批量跑实验，记录结果和指标
"""

import os
import re
import sys
import json
import time
import traceback

# 把项目根目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from llm import get_llm
from experiments.exp_config import TEST_QUERIES, EXPERIMENTS

RESULTS_DIR = os.path.join(config.OUTPUT_DIR, "experiments")


def run_single_experiment(exp_name: str, exp_cfg: dict, query: str, query_index: int) -> dict:
    """跑一组实验的一个查询，按标准目录结构保存所有产出物"""

    mode = exp_cfg.get("mode", "workflow")

    # 创建输出目录
    safe_query = re.sub(r'[\\/:*?"<>|]', '_', query[:20])
    run_dir = os.path.join(RESULTS_DIR, exp_name, f"query_{query_index+1}_{safe_query}")
    cases_dir = os.path.join(run_dir, "cases")
    os.makedirs(cases_dir, exist_ok=True)

    result = {
        "exp_name": exp_name,
        "query": query,
        "query_index": query_index,
        "mode": mode,
        "config": exp_cfg,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": run_dir,
    }

    try:
        if mode == "single_llm":
            metrics, report, state = _run_baseline(query)
        elif mode == "plan_execute":
            metrics, report, state = _run_plan_execute(query, exp_cfg)
        else:
            metrics, report, state = _run_workflow(query, exp_cfg)

        result["metrics"] = metrics
        result["status"] = "success"

        # ── 保存所有产出物 ──
        _save_outputs(run_dir, cases_dir, report, state, result)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    result["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # 保存 meta.json
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in result.items() if k != "run_dir"},
                  f, ensure_ascii=False, indent=2, default=str)

    return result


def _save_outputs(run_dir, cases_dir, report, state, result):
    """把一次运行的所有产出物存入标准目录结构"""

    # 1. 最终报告
    if report:
        with open(os.path.join(run_dir, "final_report.md"), "w", encoding="utf-8") as f:
            f.write(report)

    if state is None:
        return

    # 2. 本地情境
    local_context = state.get("local_context", {})
    if local_context:
        context_data = {
            "matched_area": local_context.get("matched_area", ""),
            "matched_areas": local_context.get("matched_areas", []),
            "core_problems": state.get("core_problems", []),
            "rewritten_problems": state.get("rewritten_problems", []),
            "context_summary": local_context.get("context_summary", ""),
            "data_analysis": local_context.get("data_analysis", "")[:5000],
            "local_data": local_context.get("local_data", {}),
        }
        with open(os.path.join(run_dir, "context.json"), "w", encoding="utf-8") as f:
            json.dump(context_data, f, ensure_ascii=False, indent=2, default=str)

    # 3. 差异分析
    adaptation = state.get("adaptation_plan", "")
    if adaptation:
        with open(os.path.join(run_dir, "gap_analysis.md"), "w", encoding="utf-8") as f:
            f.write(adaptation)

    # 4. 案例（核心：每个案例单独存 json + md）
    case_counter = 0
    case_results = state.get("case_results", {})

    # workflow 模式：case_results = {"problem_1": [...], "problem_2": [...]}
    if isinstance(case_results, dict):
        for problem_key, cases in case_results.items():
            if not isinstance(cases, list):
                continue
            for case in cases:
                if not isinstance(case, dict):
                    continue
                case_counter += 1
                _save_single_case(cases_dir, case_counter, problem_key, case)

    # plan-execute 模式：从 completed_results 里提取 deep_research 结果
    completed = state.get("completed_results", {})
    if completed and case_counter == 0:
        for tid, res in completed.items():
            if isinstance(res, dict) and res.get("final_report"):
                case_counter += 1
                _save_single_case(cases_dir, case_counter, tid, res)

    result["metrics"]["cases_saved"] = case_counter
    print(f"  💾 保存了 {case_counter} 个案例到 {cases_dir}")


def _save_single_case(cases_dir, index, problem_key, case):
    """保存单个案例的结构化数据和完整报告"""

    # 结构化 JSON（给评估 skill 读的）
    structured = {
        "case_index": index,
        "problem_key": problem_key,
        "title": case.get("title", ""),
        "url": case.get("url", ""),
        "source": case.get("source", ""),
        "language": case.get("language", ""),
        "city_country": case.get("city_country", ""),
        "time": case.get("time", ""),
        "core_problem": case.get("core_problem", ""),
        "solution": case.get("solution", ""),
        "key_results": case.get("key_results", ""),
        "preconditions": case.get("preconditions", ""),
        "downsides": case.get("downsides", ""),
        "hybrid_score": case.get("hybrid_score", 0),
        "bm25_norm": case.get("bm25_norm", 0),
        "sbert_sim": case.get("sbert_sim", 0),
        "has_supplement": case.get("has_supplement", False),
        "loop_count": case.get("loop_count", 0),
        "missing_fields": case.get("missing_fields", []),
    }

    # 提取 extraction 里的字段（plan-execute 模式的结构不同）
    extraction = case.get("extraction", {})
    if extraction and isinstance(extraction, dict):
        for field in ["city_country", "time", "core_problem", "solution",
                       "key_results", "preconditions", "downsides"]:
            if not structured[field] and extraction.get(field):
                structured[field] = extraction[field]

    json_path = os.path.join(cases_dir, f"case_{index}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2, default=str)

    # 完整分析报告 MD
    report = case.get("comprehensive_analysis", "") or case.get("final_report", "")
    if report:
        md_path = os.path.join(cases_dir, f"case_{index}_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {structured['title']}\n\n")
            f.write(f"**来源**: {structured['source']} | **语言**: {structured['language']}\n")
            f.write(f"**URL**: {structured['url']}\n\n---\n\n")
            f.write(report)


def _run_baseline(query: str) -> tuple:
    """Baseline: 单个 LLM 直接生成报告"""
    from langchain_core.messages import SystemMessage, HumanMessage
    from tools.data_loader import read_md_file
    from config import PLANNING_KNOWLEDGE_MD

    llm = get_llm(max_tokens=30000)
    md_content = read_md_file(PLANNING_KNOWLEDGE_MD)

    start = time.time()

    prompt = f"""你是资深城市规划专家。请为以下问题撰写一份完整的城市规划案例研究报告。

**问题**: {query}
**目标城市**: 香港

请包含：项目背景、核心问题、全球案例研究、差异分析、规划方案、风险评估。
用中文，Markdown 格式。"""

    response = llm.invoke([
        SystemMessage(content="你是城市规划报告专家"),
        HumanMessage(content=prompt),
    ])

    elapsed = time.time() - start
    report = response.content

    metrics = {
        "llm_calls": 1,
        "elapsed_seconds": elapsed,
        "report_length": len(report),
        "search_results_count": 0,
        "cases_found": 0,
        "deep_research_loops": 0,
    }

    return metrics, report, None


def _run_workflow(query: str, exp_cfg: dict) -> tuple:
    """Workflow 模式，通过 exp_cfg 开关控制消融"""
    # 把开关写入一个全局可读的地方，让 agent 内部读取
    import experiments.exp_flags as flags
    flags.USE_LOCAL_ANALYSIS = exp_cfg.get("use_local_analysis", True)
    flags.USE_WEB_SEARCH = exp_cfg.get("use_web_search", True)
    flags.USE_HYBRID = exp_cfg.get("use_hybrid_retrieval", True)
    flags.USE_LLM_SELECTION = exp_cfg.get("use_llm_selection", True)
    flags.USE_DEEP_RESEARCH = exp_cfg.get("use_deep_research", True)
    flags.USE_GAP_ANALYSIS = exp_cfg.get("use_gap_analysis", True)


    start = time.time()

    from main import main as main_workflow
    result = main_workflow(query)

    elapsed = time.time() - start

    if result is None:
        return {"elapsed_seconds": elapsed, "error": "workflow returned None"}, ""

    # 收集指标
    metrics = {
        "elapsed_seconds": elapsed,
        "report_length": len(result.get("final_report", "")),
        "evaluation_score": result.get("evaluation_scores", {}).get("total_score", 0),
        "core_problems_count": len(result.get("core_problems", [])),
        "cases_per_problem": {
            k: len(v) for k, v in result.get("case_results", {}).items()
        },
    }

    return metrics, result.get("final_report", ""), result


def _run_plan_execute(query: str, exp_cfg: dict) -> tuple:
    """Plan-Execute 模式"""
    start = time.time()

    from main_pe import main as main_pe
    result = main_pe(query)

    elapsed = time.time() - start

    if result is None:
        return {"elapsed_seconds": elapsed, "error": "pe returned None"}, "", None

    plan = result.get("plan", [])
    metrics = {
        "elapsed_seconds": elapsed,
        "report_length": len(result.get("final_report", "")),
        "evaluation_score": result.get("evaluation", {}).get("score", 0),
        "plan_version": result.get("plan_version", 1),
        "replan_count": result.get("replan_count", 0),
        "total_tasks": len(plan),
        "tasks_done": sum(1 for t in plan if t["status"] == "done"),
        "tasks_failed": sum(1 for t in plan if t["status"] == "failed"),
    }

    return metrics, result.get("final_report", ""), result


def run_all():
    """跑所有实验"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    total = len(EXPERIMENTS) * len(TEST_QUERIES)
    current = 0

    all_results = []

    for exp_name, exp_cfg in EXPERIMENTS.items():
        print(f"\n{'═' * 70}")
        print(f"实验组: {exp_name} — {exp_cfg['description']}")
        print(f"{'═' * 70}")

        for qi, query in enumerate(TEST_QUERIES):
            current += 1
            print(f"\n[{current}/{total}] {exp_name} | {query[:30]}...")

            result = run_single_experiment(exp_name, exp_cfg, query, qi)

            all_results.append(result)

            # 每跑完一个就保存，防止中途崩溃丢数据
            save_path = os.path.join(RESULTS_DIR, f"{exp_name}_{current}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)

            # 同时保存报告
            if result.get("report"):
                report_path = os.path.join(RESULTS_DIR,
                    f"report_{exp_name}_{query[:15].replace(' ', '_')}.md")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(result["report"])

            print(f"  → 状态: {result['status']}")
            if result.get("metrics"):
                m = result["metrics"]
                print(f"  → 耗时: {m.get('elapsed_seconds', 0):.0f}s"
                      f" | 报告长度: {m.get('report_length', 0)}")

    # 保存汇总
    summary_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n✅ 所有实验完成！结果保存在: {RESULTS_DIR}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None,
                        help="只跑指定实验组，如: ablation_no_hybrid")
    parser.add_argument("--query-index", type=int, default=None,
                        help="只跑指定查询，如: 0")
    args = parser.parse_args()

    if args.exp and args.query_index is not None:
        # 跑单个实验
        cfg = EXPERIMENTS[args.exp]
        q = TEST_QUERIES[args.query_index]
        result = run_single_experiment(args.exp, cfg, q, args.query_index)
        print(json.dumps(result.get("metrics", {}), indent=2, ensure_ascii=False))
    elif args.exp:
        # 跑一组实验的所有查询
        cfg = EXPERIMENTS[args.exp]
        for qi, q in enumerate(TEST_QUERIES):
            result = run_single_experiment(args.exp, cfg, q, qi)
            print(f"{q[:20]}... → {result['status']}")
    else:
        # 跑全部
        run_all()