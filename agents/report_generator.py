"""
报告生成模块 — 生成最终城市规划方案报告
"""

import os
import re
import time

from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from config import PLANNING_KNOWLEDGE_MD, OUTPUT_DIR
from llm import get_llm
from prompts import load_prompt
from tools.data_loader import read_md_file

import experiments.exp_flags as flags



def extract_content(response):
    """Extract text from LangChain AIMessage response, handling both string and list content."""
    if hasattr(response, 'content'):
        content = response.content
    else:
        content = response

    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if hasattr(block, 'text'):
                parts.append(block.text)
            elif isinstance(block, dict) and 'text' in block:
                parts.append(block['text'])
            else:
                parts.append(str(block))
        return ' '.join(parts)
    else:
        return str(content)


async def generate_final_report(state: AgentState) -> AgentState:
    """生成最终报告（整合所有案例分析）"""
    print("\n" + "="*60)
    print("📄 生成最终报告 - 整合所有案例分析")
    print("="*60)

    # 根据实际模型调整token限制
    llm = get_llm(max_tokens=100000)

    # ========== 第一步：整合所有案例分析 ==========
    print("\n📊 整合案例分析数据...")

    # 按问题组织案例分析
    problem_case_analyses = {}

    for problem_idx in range(1, 4):  # problem_1, problem_2, problem_3
        problem_key = f"problem_{problem_idx}"
        cases = state["case_results"].get(problem_key, [])

        if cases:
            print(f"  问题{problem_idx}: 有 {len(cases)} 个案例")

            # 为每个问题创建案例分析总结
            case_summaries = []
            for case_idx, case in enumerate(cases):
                # 提取关键信息
                title = case.get("title", "未知标题")
                source = case.get("source", "未知来源")
                language = "国际" if case.get("language") == "en" else "本地"

                # 获取详细分析
                analysis = case.get("comprehensive_analysis", "")

                if analysis and len(analysis) > 100:  # 确保有足够内容
                    # 创建案例摘要
                    case_summary = f"""
                                   案例{case_idx+1}: {title} ({language}, 来源: {source})
                                   ---
                                   {analysis}{'...' if len(analysis) > 5000 else ''}
                                   """
                    case_summaries.append(case_summary)
                else:
                    # 如果没有详细分析，使用初始提取
                    initial_extraction = case.get("initial_extraction", "")
                    if initial_extraction:
                        case_summary = f"""
                                       案例{case_idx+1}: {title} ({language}, 来源: {source})
                                       ---
                                       {initial_extraction}{'...' if len(initial_extraction) > 8000 else ''}
                                   """
                        case_summaries.append(case_summary)

            # 合并所有案例摘要
            if case_summaries:
                problem_case_analyses[problem_idx] = "\n".join(case_summaries)
                print(f"  → 整合了 {len(case_summaries)} 个案例分析")
            else:
                problem_case_analyses[problem_idx] = f"问题{problem_idx}: 暂无详细案例分析"
                print(f"  → 无有效案例分析")
        else:
            problem_case_analyses[problem_idx] = f"问题{problem_idx}: 暂无相关案例"
            print(f"  问题{problem_idx}: 无案例")

    # ========== 第二步：准备案例统计信息 ==========
    case_stats = {
        "total": 0,
        "global": 0,
        "local": 0,
        "by_problem": {}  # 每个问题的案例数
    }

    for problem_idx in range(1, 4):
        problem_key = f"problem_{problem_idx}"
        cases = state["case_results"].get(problem_key, [])
        problem_total = len(cases)
        problem_global = sum(1 for c in cases if c.get("language") == "en")
        problem_local = sum(1 for c in cases if c.get("language") == "zh")

        case_stats["total"] += problem_total
        case_stats["global"] += problem_global
        case_stats["local"] += problem_local
        case_stats["by_problem"][problem_idx] = {
            "total": problem_total,
            "global": problem_global,
            "local": problem_local
        }

    # ========== 第三步：构建详细提示词 ==========
    print("\n📝 构建报告提示词...")

    # 创建问题与案例的映射关系
    problem_case_mapping = ""
    for problem_idx in range(1, 4):
        problem_stats = case_stats["by_problem"].get(problem_idx, {"total": 0, "global": 0, "local": 0})
        problem_case_mapping += f"""
问题{problem_idx}: {state["core_problems"][problem_idx-1] if problem_idx-1 < len(state["core_problems"]) else "未知"}
- 案例总数: {problem_stats['total']} 个
- 国际案例: {problem_stats['global']} 个
- 本地案例: {problem_stats['local']} 个
"""
    md_content = read_md_file(PLANNING_KNOWLEDGE_MD)
    # 构建完整的提示词
    prompta = load_prompt(
        "agents/report_generator", "01_prompta_full",
        user_query=state['user_query'],
        matched_area=state['local_context'].get('matched_area', 'Unknown'),
        data_analysis=state['local_context'].get('data_analysis', ''),
        context_summary=state['local_context'].get('context_summary', ''),
        full_response=state['local_context'].get('full_response', ''),
        md_content=md_content,
        core_problems_list=chr(10).join([f"{i+1}. {p}" for i, p in enumerate(state['core_problems'])]),
        case_stats_total=str(case_stats['total']),
        case_stats_global=str(case_stats['global']),
        case_stats_local=str(case_stats['local']),
        problem_case_mapping=problem_case_mapping,
        core_problem_1=state["core_problems"][0] if len(state["core_problems"]) > 0 else "未知",
        core_problem_2=state["core_problems"][1] if len(state["core_problems"]) > 1 else "未知",
        core_problem_3=state["core_problems"][2] if len(state["core_problems"]) > 2 else "未知",
        problem_1_case_analyses=problem_case_analyses.get(1, "暂无案例分析"),
        problem_2_case_analyses=problem_case_analyses.get(2, "暂无案例分析"),
        problem_3_case_analyses=problem_case_analyses.get(3, "暂无案例分析"),
        gap_analysis_directions=chr(10).join([f"方向{i+1}: {v.get('problem', '')}\n{v.get('analysis', '')}" for i, v in enumerate(state['gap_analysis'].values())]),
        adaptation_plan=state['adaptation_plan'],
        total_score=str(state['evaluation_scores'].get('total_score', 'N/A')),
        scores_detail=chr(10).join([f"- {k}: {v}/100" for k, v in state['evaluation_scores'].get('scores', {}).items()]),
        rewritten_problem_1=state["rewritten_problems"][0] if len(state["rewritten_problems"]) > 0 else "",
        rewritten_problem_2=state["rewritten_problems"][1] if len(state["rewritten_problems"]) > 1 else "",
        rewritten_problem_3=state["rewritten_problems"][2] if len(state["rewritten_problems"]) > 2 else "",
    )
    promptb = load_prompt(
        "agents/report_generator", "02_promptb_no_local",
        user_query=state['user_query'],
        matched_area=state['local_context'].get('matched_area', 'Unknown'),
        case_stats_total=str(case_stats['total']),
        case_stats_global=str(case_stats['global']),
        case_stats_local=str(case_stats['local']),
        core_problem_1=state["core_problems"][0] if len(state["core_problems"]) > 0 else "未知",
        core_problem_2=state["core_problems"][1] if len(state["core_problems"]) > 1 else "未知",
        core_problem_3=state["core_problems"][2] if len(state["core_problems"]) > 2 else "未知",
        problem_1_case_analyses=problem_case_analyses.get(1, "暂无案例分析"),
        problem_2_case_analyses=problem_case_analyses.get(2, "暂无案例分析"),
        problem_3_case_analyses=problem_case_analyses.get(3, "暂无案例分析"),
        gap_analysis_directions=chr(10).join([f"方向{i+1}: {v.get('problem', '')}\n{v.get('analysis', '')}" for i, v in enumerate(state['gap_analysis'].values())]),
        adaptation_plan=state['adaptation_plan'],
        total_score=str(state['evaluation_scores'].get('total_score', 'N/A')),
        scores_detail=chr(10).join([f"- {k}: {v}/100" for k, v in state['evaluation_scores'].get('scores', {}).items()]),
        rewritten_problem_1=state["rewritten_problems"][0] if len(state["rewritten_problems"]) > 0 else "",
        rewritten_problem_2=state["rewritten_problems"][1] if len(state["rewritten_problems"]) > 1 else "",
        rewritten_problem_3=state["rewritten_problems"][2] if len(state["rewritten_problems"]) > 2 else "",
    )
    promptc = load_prompt(
        "agents/report_generator", "03_promptc_no_gap",
        user_query=state['user_query'],
        matched_area=state['local_context'].get('matched_area', 'Unknown'),
        data_analysis=state['local_context'].get('data_analysis', ''),
        context_summary=state['local_context'].get('context_summary', ''),
        full_response=state['local_context'].get('full_response', ''),
        md_content=md_content,
        core_problems_list=chr(10).join([f"{i+1}. {p}" for i, p in enumerate(state['core_problems'])]),
        case_stats_total=str(case_stats['total']),
        case_stats_global=str(case_stats['global']),
        case_stats_local=str(case_stats['local']),
        problem_case_mapping=problem_case_mapping,
        core_problem_1=state["core_problems"][0] if len(state["core_problems"]) > 0 else "未知",
        core_problem_2=state["core_problems"][1] if len(state["core_problems"]) > 1 else "未知",
        core_problem_3=state["core_problems"][2] if len(state["core_problems"]) > 2 else "未知",
        problem_1_case_analyses=problem_case_analyses.get(1, "暂无案例分析"),
        problem_2_case_analyses=problem_case_analyses.get(2, "暂无案例分析"),
        problem_3_case_analyses=problem_case_analyses.get(3, "暂无案例分析"),
        rewritten_problem_1=state["rewritten_problems"][0] if len(state["rewritten_problems"]) > 0 else "",
        rewritten_problem_2=state["rewritten_problems"][1] if len(state["rewritten_problems"]) > 1 else "",
        rewritten_problem_3=state["rewritten_problems"][2] if len(state["rewritten_problems"]) > 2 else "",
    )
    try:
        from experiments.exp_flags import USE_LOCAL_ANALYSIS
    except ImportError:
        USE_LOCAL_ANALYSIS = True
    
    try:
        from experiments.exp_flags import USE_GAP_ANALYSIS
    except ImportError:
        USE_GAP_ANALYSIS = True

    if USE_LOCAL_ANALYSIS == False:
        prompt = promptb
    elif USE_GAP_ANALYSIS == False:
        prompt = promptc
    else:
        prompt = prompta

    print("✅ 提示词构建完成")
    print(f"📊 提示词长度: {len(prompt)} 字符")
    print(f"📄 包含案例分析: {case_stats['total']} 个")

    # ========== 第四步：调用LLM生成报告 ==========
    print("\n🧠 调用LLM生成报告...")

    try:
        messages = [
            SystemMessage(content="""你是资深城市规划报告撰写专家，具有丰富的国际经验。
你的任务是根据提供的详细案例分析，撰写一份专业、全面的城市规划方案报告。
                          
特别注意：
（直接输出报告内容，不要包含任何解释或前置文本，比如，好的，作为资深城市规划报告撰写专家，我将基于您提供的全部信息，为您生成一份专业、全面的报告。）                          
 """),
            HumanMessage(content=prompt)
        ]

        response = await llm.ainvoke(messages)

        if response and extract_content(response):
            state["final_report"] = extract_content(response)

            print("\n✅ 报告生成完成")
            print(f"📊 报告长度: {len(extract_content(response))} 字符")

            # 分析报告内容
            sections = extract_content(response).count("##")
            references = sum(1 for line in extract_content(response).split('\n') if "案例" in line)
            print(f"📄 报告章节: {sections} 个")
            print(f"🔗 案例引用: {references} 处")

            # 保存报告到文件
            safe_query = re.sub(r'[\\/:*?"<>|]', '_', str(state['user_query'][:50]))
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            report_filename = os.path.join(OUTPUT_DIR, f"final_report_{safe_query}.md")

            with open(report_filename, "w", encoding="utf-8") as f:
                f.write(extract_content(response))

            # 收集所有案例的详细分析并追加到文件
            print("📋 收集案例详细分析...")
            all_case_analyses = []
            for problem_idx in range(1, 4):
                problem_key = f"problem_{problem_idx}"
                cases = state["case_results"].get(problem_key, [])
                for case_idx, case in enumerate(cases):
                    analysis = case.get("comprehensive_analysis", "")
                    if analysis and len(analysis.strip()) > 0:
                        title = case.get("title", f"案例{case_idx+1}")
                        source = case.get("source", "未知来源")
                        language = "国际" if case.get("language") == "en" else "本地"
                        url = case.get("url", "")
                        url_display = f"[链接]({url})" if url and url not in ("", "N/A") else "无URL"
                        separator = f"\n\n{'='*80}\n"
                        case_header = f"## 案例详情: {title}\n\n ### **所属问题**:\n {problem_idx} - {state['core_problems'][problem_idx-1] if problem_idx-1 < len(state['core_problems']) else '未知'}\n **URL**: {url_display}\n\n"
                        all_case_analyses.append(separator + case_header + analysis)

            if all_case_analyses:
                print(f"📄 找到 {len(all_case_analyses)} 个案例详细分析，追加到报告文件")
                # 构建附录内容字符串
                appendix_content = ""
                appendix_content += "# 附录: 详细案例分析\n"
                appendix_content += "以下为每个案例的完整详细分析报告，供深度参考:\n\n"
                for case_analysis in all_case_analyses:
                    appendix_content += case_analysis

                # 将附录追加到文件
                with open(report_filename, "a", encoding="utf-8") as f:
                    f.write(appendix_content)

                # 将附录追加到state中的报告内容
                state["final_report"] += appendix_content
                print(f"✅ 案例详细分析已追加到报告文件和state")
            else:
                print("ℹ️ 未找到案例详细分析内容")

            print(f"💾 报告已保存: {report_filename}")

            # 保存案例分析摘要（便于调试）
            summary_filename = os.path.join(OUTPUT_DIR, f"case_summary_{safe_query}.txt")
            with open(summary_filename, "w", encoding="utf-8") as f:
                f.write(f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"原始问题: {state['user_query']}\n\n")
                f.write("案例分析统计:\n")
                for problem_idx in range(1, 4):
                    f.write(f"问题{problem_idx}: {case_stats['by_problem'][problem_idx]['total']} 个案例\n")

        else:
            print("⚠️ 报告生成失败：无响应内容")
            state["final_report"] = "# 报告生成失败\n\nLLM未返回有效内容。"

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"⚠️ 报告生成异常: {str(e)}")
        print(f"详细堆栈跟踪:\n{error_details}")
        state["final_report"] = f"# 报告生成失败\n\n错误信息: {str(e)}\n\n详细跟踪: {error_details[:1000]}"

    return state