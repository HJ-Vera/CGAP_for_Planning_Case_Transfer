"""
智能体 3: 差异分析与改造智能体
- 对比案例与本地情境
- 识别阻碍点
- 生成改造方案
"""

import json

from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from config import TOKEN_LIMITS, PLANNING_KNOWLEDGE_MD, OUTPUT_DIR
from llm import get_llm
from tools.data_loader import read_md_file
import experiments.exp_flags as flags


async def gap_analysis_agent(state: AgentState) -> AgentState:
    """
    差异分析与改造智能体
    - 对比案例与本地情境
    - 识别阻碍点
    - 生成改造方案
    """
    print("\n" + "="*60)
    print("🔬 差异分析与改造智能体启动")
    print("="*60)

    llm = get_llm(max_tokens=TOKEN_LIMITS["gap_analysis_agent"])

    case_results = state["case_results"]
    local_context = state["local_context"]

    gap_analysis_results = {}

    try:
        from experiments.exp_flags import USE_GAP_ANALYSIS
    except ImportError:
        USE_GAP_ANALYSIS= True

    if USE_GAP_ANALYSIS == False:
        print("⚠️ 差异分析已关闭，跳过此步骤")
        state["gap_analysis"] = {}
        state["adaptation_plan"] = "差异分析已关闭，未生成改造方案。"
        return state

    for i, problem in enumerate(state["core_problems"]):
        print(f"\n📊 分析问题 {i+1}: {problem}")

        cases = case_results.get(f"problem_{i+1}", [])

        # 提取案例的综合分析
        cases_analysis = "\n\n".join([
            f"【案例 {c.get('case_number', j+1)}】\n"
            f"标题: {c.get('title', 'Unknown')}\n"
            f"来源: {c.get('source', 'Unknown')} ({c.get('language', 'unknown')})\n"
            f"\n{c.get('comprehensive_analysis', c.get('extracted_info', ''))}\n"
            for j, c in enumerate(cases[:5])  # 使用前5个案例
        ])

        # 获取全球趋势总结
        global_summary = cases[0].get('global_summary', '') if cases else ''
        md_content = read_md_file(PLANNING_KNOWLEDGE_MD)

        prompt = f"""
作为城市规划专家,请分析全球案例与本地情境的差异:

**本地区域**: {local_context.get('matched_area', 'Unknown')}
**本地数据**: {json.dumps(local_context.get('data_analysis', {}), ensure_ascii=False)}
**本地相关法规与政策**: {md_content}
**核心问题**: {problem}

**全球趋势总结**:
{global_summary}

**详细案例分析**:
{cases_analysis}

请完成以下分析（用简明文本回答，每部分3-5句话）:

【对比分析】
全球案例的主要做法与本地情境的相似点和差异点:

【关键差异 (Gaps)】
哪些条件在全球案例中存在但本地缺失（制度、经济、技术、社会条件等）:

【适应性改造方案】
如何调整全球经验以适应本地情境（具体措施）:

【实施风险预警】
借鉴这些方案可能面临的挑战和代价:

直接输出分析文本。
"""

        messages = [SystemMessage(content="你是国际城市规划专家"), HumanMessage(content=prompt)]
        response = await llm.ainvoke(messages)

        analysis_text = response.content.strip()

        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, f"analysis_text_problem_{i+1}.md"), "w", encoding="utf-8") as f:
            f.write(analysis_text)

        gap_analysis_results[f"problem_{i+1}"] = {
            "problem": problem,
            "cases_count": len(cases),
            "global_cases": sum(1 for c in cases if c.get('language') == 'en'),
            "local_cases": sum(1 for c in cases if c.get('language') == 'zh'),
            "analysis": analysis_text,
            "global_summary": global_summary
        }

        print(f"  ✅ 分析完成")
        print(f"  📊 参考案例: {len(cases)} 个 (国际: {gap_analysis_results[f'problem_{i+1}']['global_cases']}, 本地: {gap_analysis_results[f'problem_{i+1}']['local_cases']})")

    # 整合方案
    print("\n🔗 整合总体规划方案...")

    integration_prompt = f"""
请将以下三个方向的全球经验分析整合成一个连贯的城市规划方案:

**区域**: {local_context.get('matched_area', 'Unknown')}
**本地数据**: {json.dumps(local_context.get('data_analysis', {}), ensure_ascii=False)}
**本地分析**: {json.dumps(local_context.get('context_summary', {}), ensure_ascii=False)}
**本地存在问题**: {json.dumps(local_context.get('full_response', {}), ensure_ascii=False)}
**本地相关法规与政策**: {md_content}

{chr(10).join([
    f"**方向{i+1}: {v['problem']}**\n"
    f"全球趋势: {v['global_summary']}\n"
    f"分析结果: {v['analysis']}\n"
    for i, v in enumerate(gap_analysis_results.values())
])}

请撰写一份综合规划方案，内容详实，细节充分，分析深入，内容包括:

【一、总体定位与目标】
基于本地情境和全球经验的综合定位

【二、三大核心策略】
方向1: [具体措施、国际经验借鉴、本地化调整]
方向2: [具体措施、国际经验借鉴、本地化调整]
方向3: [具体措施、国际经验借鉴、本地化调整]

【三、前置条件与准备】
需要具备的制度、经济、技术、社会条件

【四、实施路径与时序】
短期、中期、长期的实施步骤

【五、风险管控】
潜在代价、负面影响及应对措施

【六、预期成果】
定量和定性的预期效果

请确保方案具有国际视野但符合本地实际。
"""

    messages = [SystemMessage(content="你是国际城市规划专家"), HumanMessage(content=integration_prompt)]
    integration_response = await llm.ainvoke(messages)

    state["gap_analysis"] = gap_analysis_results
    state["adaptation_plan"] = integration_response.content.strip()

    print("\n✅ 差异分析完成")
    print(f"📄 规划方案长度: {len(state['adaptation_plan'])} 字符（Markdown 结果将通过独立气泡渲染）")

    # 保存总结
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "adaptation_plan.md"), "w", encoding="utf-8") as f:
        f.write(state["adaptation_plan"])

    return state