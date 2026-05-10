"""
智能体 4: 评审智能体
- 评估方案质量
- 决定是否需要重新搜索
"""

import re

from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from config import TOKEN_LIMITS, PLANNING_KNOWLEDGE_MD
from llm import get_llm
from prompts import load_prompt
from tools.data_loader import read_md_file


def extract_content(response):
    """Extract text from LangChain AIMessage response, handling both string and list content."""
    content = response.content
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


async def evaluation_agent(state: AgentState) -> AgentState:
    """
    评审智能体
    - 评估方案质量
    - 决定是否需要重新搜索
    """
    print("\n" + "="*60)
    print("⚖️ 评审智能体启动")
    print("="*60)

    llm = get_llm(max_tokens=TOKEN_LIMITS["evaluation_agent"])
    # 简化评估prompt
    try:
        from experiments.exp_flags import USE_GAP_ANALYSIS
    except ImportError:
        USE_GAP_ANALYSIS = True

    if not USE_GAP_ANALYSIS:
        print("⚠️ 差异分析已关闭，使用全球案例总结代替")
        summaries = []
        core_problems = state.get("core_problems", [])
        for i, (problem_key, cases) in enumerate(state.get("case_results", {}).items()):
            problem_text = core_problems[i] if i < len(core_problems) else problem_key
            if cases and isinstance(cases, list):
                summary = cases[0].get("global_summary", "")
                if summary:
                    summaries.append(f"### {problem_text}\n{summary}")
        adaptation_plan = "\n\n".join(summaries) if summaries else "无全球案例总结"
    else:
        adaptation_plan = state.get("adaptation_plan", "无改造方案可评审")

    md_content = read_md_file(PLANNING_KNOWLEDGE_MD)    
    prompt = load_prompt(
        "agents/evaluation_agent", "01_evaluation_prompt",
        user_query=state['user_query'],
        matched_area=state['local_context'].get('matched_area', 'Unknown'),
        data_analysis=state['local_context'].get('data_analysis', ''),
        context_summary=state['local_context'].get('context_summary', ''),
        full_response=state['local_context'].get('full_response', ''),
        md_content=md_content,
        core_problems_list=chr(10).join([f"{i+1}. {p}" for i, p in enumerate(state['core_problems'])]),
        adaptation_plan=adaptation_plan,
    )

    messages = [SystemMessage(content="你是评审专家"), HumanMessage(content=prompt)]
    response = await llm.ainvoke(messages)

    response_text = extract_content(response).strip()
    print(f"\n评审结果:\n{response_text}\n")

    # 提取分数（宽松匹配：兼容 markdown 加粗、编号前缀等 LLM 变体）
    scores = {}
    score_patterns = [
        (r'问题匹配度.*?[:：]\s*(\d+)', 'problem_matching'),
        (r'信息完整性.*?[:：]\s*(\d+)', 'information_completeness'),
        (r'逻辑连贯性.*?[:：]\s*(\d+)', 'logical_coherence'),
        (r'实施可行性.*?[:：]\s*(\d+)', 'implementation_feasibility')
    ]

    for pattern, key in score_patterns:
        match = re.search(pattern, response_text)
        if match:
            scores[key] = int(match.group(1))
        else:
            scores[key] = 70  # 默认分数

    total_score = sum(scores.values())
    all_passed = all(score >= 60 for score in scores.values())
    pass_threshold = all_passed and total_score >= 240

    # 判断是否需要改进
    needs_revision = not pass_threshold

    evaluation = {
        "scores": scores,
        "total_score": total_score,
        "all_passed": all_passed,
        "feedback": response_text,
        "needs_revision": needs_revision,
        "revision_instructions": "请根据评审意见改进" if needs_revision else ""
    }

    state["evaluation_scores"] = evaluation
    state["feedback"] = response_text

    # 打印评估结果
    print("\n📊 评估结果:")
    for dimension, score in scores.items():
        status = "✅" if score >= 60 else "❌"
        print(f"  {status} {dimension}: {score}/100")
    print(f"\n  总分: {total_score}/400")
    print(f"  是否通过: {'✅ 是' if pass_threshold else '❌ 否'}")

    # 决策逻辑
    if pass_threshold:
        state["is_complete"] = True
        print("\n✅ 方案通过评审,准备生成最终报告")
    else:
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        if state["iteration_count"] >= 3:
            state["is_complete"] = True
            print("\n⚠️ 已达最大迭代次数,强制生成报告")
        else:
            state["is_complete"] = False
            print(f"\n🔄 需要改进,启动第 {state['iteration_count']} 轮反馈循环")

    return state