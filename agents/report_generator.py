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


def generate_final_report(state: AgentState) -> AgentState:
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
    prompt = ""
    # 构建完整的提示词
    prompta = f"""
请基于以下内容生成一份完整的国际化城市规划方案报告:

# 一、项目概况
**原始问题**: {state['user_query']}
**目标区域**: {state['local_context'].get('matched_area', 'Unknown')}
**本地数据**: {state['local_context'].get('data_analysis', '')}
**本地情境**: {state['local_context'].get('context_summary', '')}
**本地存在问题**: {state['local_context'].get('full_response', '')}
**相关政策法规**: {md_content}

# 二、核心问题识别
{chr(10).join([f"{i+1}. {p}" for i, p in enumerate(state['core_problems'])])}

# 三、全球案例研究概况
共收集 {case_stats['total']} 个相关案例，其中：
- 国际案例: {case_stats['global']} 个
- 本地案例: {case_stats['local']} 个

{problem_case_mapping}

# 四、详细案例分析
以下是每个问题对应的详细案例分析，请仔细参考这些案例内容：

## 问题1: {state["core_problems"][0] if len(state["core_problems"]) > 0 else "未知"}
{problem_case_analyses.get(1, "暂无案例分析")}

## 问题2: {state["core_problems"][1] if len(state["core_problems"]) > 1 else "未知"}
{problem_case_analyses.get(2, "暂无案例分析")}

## 问题3: {state["core_problems"][2] if len(state["core_problems"]) > 2 else "未知"}
{problem_case_analyses.get(3, "暂无案例分析")}

# 五、差异分析
{chr(10).join([f"方向{i+1}: {v.get('problem', '')}\n{v.get('analysis', '')}" for i, v in enumerate(state['gap_analysis'].values())])}

# 六、综合规划方案
{state['adaptation_plan']}

# 七、评估结果
总分: {state['evaluation_scores'].get('total_score', 'N/A')}/400
{chr(10).join([f"- {k}: {v}/100" for k, v in state['evaluation_scores'].get('scores', {}).items()])}

# 八、报告撰写要求
请基于以上所有信息，生成一份详细、专业的 Markdown 格式城市规划方案报告。

## 报告结构要求：
1. **执行摘要** - 简要概括报告核心内容
2. **项目背景与本地情境分析** - 详细分析目标区域的现状
3. **核心问题识别** - 重述并分析三个核心问题
4. **全球案例研究**
   - 4.1 {state["rewritten_problems"][0]}案例分析（重点参考问题1的案例）
   - 4.2 {state["rewritten_problems"][1]}案例分析（重点参考问题2的案例）
   - 4.3 {state["rewritten_problems"][2]}案例分析（重点参考问题3的案例）
   - 4.4 关键成功要素总结
5. **差异分析与适应性改造**
   - 5.1 前置条件对比分析
   - 5.2 实施障碍识别
   - 5.3 本地化调整策略
6. **综合规划方案**
   - 6.1 总体定位与目标
   - 6.2 三大核心策略（对应三个问题）
   - 6.3 实施路径与时序安排
7. **风险评估与应对**
   - 7.1 潜在代价分析
   - 7.2 负面影响预警
   - 7.3 风险管控措施
8. **预期成果与评估**
9. **实施建议与下一步行动**
10. **附录：详细案例清单**

## 撰写注意事项：
1. **充分利用案例分析**：报告中必须引用具体案例，说明哪些国际/本地经验可借鉴
2. **保持专业性和逻辑性**：报告应具有学术和实践价值
3. **突出本地适应性**：强调如何将国际经验本地化
4. **数据可视化建议**：在适当位置建议添加表格、图表等可视化元素
5. **语言风格**：专业但不晦涩，适合决策者阅读
6. **内容详实**：每个部分都要有充分的分析和细节支持
7.**不能编造内容或者案例**：所有内容必须基于提供的信息和分析结果，不能凭空捏造数据或案例。
8. 每个策略部分都要有案例支撑,保持报告的专业性和可操作性
9. 报告总篇幅不少于15000字，请确保内容详实、分析深入、细节充分。

请开始撰写完整报告（使用中文，Markdown格式）。
"""
    promptb = f"""
请基于以下内容生成一份完整的国际化城市规划方案报告:

# 一、项目概况
**原始问题**: {state['user_query']}
**目标区域**: {state['local_context'].get('matched_area', 'Unknown')}

# 二、全球案例研究概况
共收集 {case_stats['total']} 个相关案例，其中：
- 国际案例: {case_stats['global']} 个
- 本地案例: {case_stats['local']} 个

# 三、详细案例分析
以下是每个问题对应的详细案例分析，请仔细参考这些案例内容：

## 问题1: {state["core_problems"][0] if len(state["core_problems"]) > 0 else "未知"}
{problem_case_analyses.get(1, "暂无案例分析")}

## 问题2: {state["core_problems"][1] if len(state["core_problems"]) > 1 else "未知"}
{problem_case_analyses.get(2, "暂无案例分析")}

## 问题3: {state["core_problems"][2] if len(state["core_problems"]) > 2 else "未知"}
{problem_case_analyses.get(3, "暂无案例分析")}

# 五、差异分析
{chr(10).join([f"方向{i+1}: {v.get('problem', '')}\n{v.get('analysis', '')}" for i, v in enumerate(state['gap_analysis'].values())])}

# 六、综合规划方案
{state['adaptation_plan']}

# 七、评估结果
总分: {state['evaluation_scores'].get('total_score', 'N/A')}/400
{chr(10).join([f"- {k}: {v}/100" for k, v in state['evaluation_scores'].get('scores', {}).items()])}

# 八、报告撰写要求
请基于以上所有信息，生成一份详细、专业的 Markdown 格式城市规划方案报告。

## 报告结构要求：
1. **执行摘要** - 简要概括报告核心内容，不要对查询区域做出深入分析，聚焦案例总结
2. **全球案例研究**
   - 2.1 {state["rewritten_problems"][0]}案例分析（重点参考问题1的案例）
   - 2.2 {state["rewritten_problems"][1]}案例分析（重点参考问题2的案例）
   - 2.3 {state["rewritten_problems"][2]}案例分析（重点参考问题3的案例）
   - 2.4 关键成功要素总结
3. **差异分析与适应性改造**
   - 3.1 前置条件对比分析
   - 3.2 实施障碍识别
   - 3.3 本地化调整策略
4. **综合规划方案**
   - 4.1 总体定位与目标
   - 4.2 核心策略
   - 4.3 实施路径与时序安排
5. **风险评估与应对**
   - 5.1 潜在代价分析
   - 5.2 负面影响预警
   - 5.3 风险管控措施
6. **预期成果与评估**
7. **实施建议与下一步行动**
8. **附录：详细案例清单**

## 撰写注意事项：
1. **充分利用案例分析**：报告中必须引用具体案例，说明哪些国际/本地经验可借鉴
2. **保持专业性和逻辑性**：报告应具有学术和实践价值
3. **突出本地适应性**：强调如何将国际经验本地化
4. **数据可视化建议**：在适当位置建议添加表格、图表等可视化元素
5. **语言风格**：专业但不晦涩，适合决策者阅读
6. **内容详实**：每个部分都要有充分的分析和细节支持
7.**不能编造内容或者案例**：所有内容必须基于提供的信息和分析结果，不能凭空捏造数据或案例。
8. 每个策略部分都要有案例支撑,保持报告的专业性和可操作性
9. 报告总篇幅不少于15000字，请确保内容详实、分析深入、细节充分。
9. 严格按照目录组织内容，不能新增或删除章节标题，确保内容完整覆盖每个部分要求。

请开始撰写完整报告（使用中文，Markdown格式）。
"""
    promptc = f"""
请基于以下内容生成一份完整的国际化城市规划方案报告:

# 一、项目概况
**原始问题**: {state['user_query']}
**目标区域**: {state['local_context'].get('matched_area', 'Unknown')}
**本地数据**: {state['local_context'].get('data_analysis', '')}
**本地情境**: {state['local_context'].get('context_summary', '')}
**本地存在问题**: {state['local_context'].get('full_response', '')}
**相关政策法规**: {md_content}

# 二、核心问题识别
{chr(10).join([f"{i+1}. {p}" for i, p in enumerate(state['core_problems'])])}

# 三、全球案例研究概况
共收集 {case_stats['total']} 个相关案例，其中：
- 国际案例: {case_stats['global']} 个
- 本地案例: {case_stats['local']} 个

{problem_case_mapping}

# 四、详细案例分析
以下是每个问题对应的详细案例分析，请仔细参考这些案例内容：

## 问题1: {state["core_problems"][0] if len(state["core_problems"]) > 0 else "未知"}
{problem_case_analyses.get(1, "暂无案例分析")}

## 问题2: {state["core_problems"][1] if len(state["core_problems"]) > 1 else "未知"}
{problem_case_analyses.get(2, "暂无案例分析")}

## 问题3: {state["core_problems"][2] if len(state["core_problems"]) > 2 else "未知"}
{problem_case_analyses.get(3, "暂无案例分析")}

# 五、报告撰写要求
请基于以上所有信息，生成一份详细、专业的 Markdown 格式城市规划方案报告。

## 报告结构要求：
1. **执行摘要** - 简要概括报告核心内容
2. **项目背景与本地情境分析** - 详细分析目标区域的现状
3. **核心问题识别** - 重述并分析三个核心问题
4. **全球案例研究**
   - 4.1 {state["rewritten_problems"][0]}案例分析（重点参考问题1的案例）
   - 4.2 {state["rewritten_problems"][1]}案例分析（重点参考问题2的案例）
   - 4.3 {state["rewritten_problems"][2]}案例分析（重点参考问题3的案例）
   - 4.4 关键成功要素总结
5. **附录：详细案例清单**

## 撰写注意事项：
1. **充分利用案例分析**：报告中必须引用具体案例，说明哪些国际/本地经验可借鉴
2. **保持专业性和逻辑性**：报告应具有学术和实践价值
3. **数据可视化建议**：在适当位置建议添加表格、图表等可视化元素
4. **语言风格**：专业但不晦涩，适合决策者阅读
5. **内容详实**：每个部分都要有充分的分析和细节支持
6.**不能编造内容或者案例**：所有内容必须基于提供的信息和分析结果，不能凭空捏造数据或案例。
7. 每个策略部分都要有案例支撑,保持报告的专业性和可操作性
8. 报告总篇幅不少于15000字，请确保内容详实、分析深入、细节充分。
9. 严格按照目录组织内容，不能新增或删除章节标题，确保内容完整覆盖每个部分要求。

请开始撰写完整报告（使用中文，Markdown格式）。
"""
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

        response = llm.invoke(messages)

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