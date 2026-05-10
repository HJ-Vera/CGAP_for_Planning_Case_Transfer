"""
Plan-Execute 模式的报告生成器
"""

import os
import json
import time

from langchain_core.messages import SystemMessage, HumanMessage

from llm import get_llm
from config import OUTPUT_DIR, PLANNING_KNOWLEDGE_MD
from prompts import load_prompt
from tools.data_loader import read_md_file
from plan_execute.pe_state import PlanExecuteState


def generate_report(state: PlanExecuteState) -> PlanExecuteState:
    """根据所有子任务结果生成最终报告，并在附录中附上每个案例的完整分析"""
    print("\n" + "=" * 60)
    print("📄 生成最终报告")
    print("=" * 60)

    llm = get_llm(max_tokens=30000)

    completed = state.get("completed_results", {})
    local_context = state.get("local_context", {})
    evaluation = state.get("evaluation", {})

    # ── 收集案例的 comprehensive analysis（用于附录）──
    case_analyses = []
    for task in state["plan"]:
        tid = task["task_id"]
        result = completed.get(tid)
        if task["task_type"] != "deep_research_case":
            continue
        if not isinstance(result, dict):
            continue

        title = result.get("title", "未知案例")
        report = result.get("final_report", "")
        note = result.get("note", "")
        extraction = result.get("extraction", {})

        if report and len(report) > 100:
            case_analyses.append({
                "task_id": tid,
                "title": title,
                "report": report,
                "note": note,
                "extraction": extraction,
            })

    print(f"  📎 收集到 {len(case_analyses)} 个案例的完整分析（将附在附录中）")

    # ── 组织所有结果用于主报告 ──
    results_sections = []
    for task in state["plan"]:
        tid = task["task_id"]
        result = completed.get(tid)
        if result is None:
            continue

        section = f"### {task['description']} [{task['task_type']}]\n"

        if isinstance(result, str):
            section += result
        elif isinstance(result, list):
            section += f"共 {len(result)} 条结果:\n"
            for i, r in enumerate(result[:10]):
                if isinstance(r, dict):
                    title = str(r.get('title') or '')
                    snippet = r.get('snippet')
                    abstract = r.get('abstract')
                    text = str(snippet or abstract or '')[:150]
                    section += f"  {i+1}. {title} - {text}\n"
                else:
                    section += f"  {i+1}. {str(r)[:150]}\n"
        elif isinstance(result, dict):
            if result.get("final_report"):
                section += result["final_report"]
            elif result.get("analysis_text"):
                section += result["analysis_text"]
            else:
                section += json.dumps(result, ensure_ascii=False, indent=2)[:2000]

        results_sections.append(section)

    md_content = read_md_file(PLANNING_KNOWLEDGE_MD)

    # ── 告诉 LLM 附录由系统自动生成，不需要它写 ──
    prompt = load_prompt(
        "plan_execute/pe_report", "01_report_prompt",
        user_query=state["user_query"],
        target_city=state["target_city"],
        matched_area=str(state.get("matched_area", "未确定")),
        core_problems=str(local_context.get("core_problems", [])),
        plan_version=str(state.get("plan_version", 1)),
        md_content=md_content,
        results_sections=chr(10).join(results_sections),
        evaluation_score=str(evaluation.get("score", "N/A")),
        strengths=str(evaluation.get("strengths", [])),
        weaknesses=str(evaluation.get("weaknesses", [])),
    )

    try:
        response = llm.invoke([
            SystemMessage(content="你是资深城市规划报告专家"),
            HumanMessage(content=prompt),
        ])

        main_report = response.content
        print(f"  ✅ 主报告生成完成: {len(main_report)} 字符")

        # ── 拼接附录 ──
        appendix_parts = []
        appendix_parts.append("\n\n---\n")
        appendix_parts.append("# 附录：案例完整分析报告\n")
        appendix_parts.append(f"> 以下为系统通过 Gap-Driven Tree Search 自动生成的 "
                              f"{len(case_analyses)} 个案例的完整结构化分析。\n")

        if case_analyses:
            for i, ca in enumerate(case_analyses, 1):
                appendix_parts.append(f"\n## 附录 {i}：{ca['title']}\n")
                appendix_parts.append(f"**任务编号**: {ca['task_id']}\n")

                if ca.get("note"):
                    appendix_parts.append(f"**备注**: {ca['note']}\n")

                # 如果有结构化提取字段，先列摘要
                ext = ca.get("extraction", {})
                if ext and isinstance(ext, dict):
                    fields = [
                        ("city_country", "城市/国家"),
                        ("time", "时间"),
                        ("core_problem", "核心问题"),
                        ("solution", "解决方案"),
                        ("key_results", "关键成果"),
                        ("preconditions", "前置条件"),
                        ("downsides", "潜在代价"),
                    ]
                    has_any = any(ext.get(k) for k, _ in fields)
                    if has_any:
                        appendix_parts.append("\n### 结构化提取摘要\n")
                        appendix_parts.append("| 字段 | 内容 |\n|---|---|\n")
                        for key, label in fields:
                            val = ext.get(key, "")
                            if val and val.strip():
                                # 表格里不能有换行，替换掉
                                val_clean = val.replace("\n", " ").replace("|", "\\|")[:300]
                                appendix_parts.append(f"| {label} | {val_clean} |\n")
                        appendix_parts.append("\n")

                # 完整报告
                appendix_parts.append("### 完整分析报告\n\n")
                appendix_parts.append(ca["report"])
                appendix_parts.append("\n")
        else:
            appendix_parts.append("\n*（本次运行未生成有效的案例深度分析报告）*\n")

        appendix = "".join(appendix_parts)
        print(f"  📎 附录生成完成: {len(appendix)} 字符，含 {len(case_analyses)} 个案例")

        # ── 合并最终报告 ──
        final_report = main_report + appendix
        state["final_report"] = final_report
        state["is_complete"] = True

        print(f"  ✅ 最终报告总长度: {len(final_report)} 字符"
              f"（主报告 {len(main_report)} + 附录 {len(appendix)}）")

        # 保存
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        report_path = os.path.join(OUTPUT_DIR, f"pe_report_{int(time.time())}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"  💾 报告已保存: {report_path}")

    except Exception as e:
        print(f"  ⚠️ 报告生成失败: {e}")
        state["final_report"] = f"# 报告生成失败\n\n错误: {str(e)}"
        state["is_complete"] = True

    return state
