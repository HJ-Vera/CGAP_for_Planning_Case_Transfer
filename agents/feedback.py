"""
反馈循环模块 — 根据评审反馈重新搜索案例
"""

from state import AgentState
from langchain_core.messages import SystemMessage, HumanMessage
from llm import get_llm
import json

def feedback_loop(state: AgentState) -> AgentState:
    """
    根据评审反馈:
      1. LLM 分析不足之处，生成新的搜索关键词
      2. 用新关键词重新搜索案例（覆盖 3 个问题）
    """
    iteration = state.get("iteration_count", 0)
    print(f"\n{'=' * 60}")
    print(f"🔄 反馈重查 — 第 {iteration} 轮")
    print(f"{'=' * 60}")

    feedback = state.get("feedback", "")
    evaluation = state.get("evaluation_scores", {})
    revision = evaluation.get("revision_instructions", "")
    scores = evaluation.get("scores", {})

    # ── 第1步: LLM 分析反馈，生成新的搜索方向 ──
    print("\n📝 步骤 1: 分析评审反馈，调整搜索方向...")

    llm = get_llm(max_tokens=8000)

    prompt = f"""你是城市规划研究顾问。评审专家对当前方案提出了以下反馈:

**评审反馈**: {feedback}
**各项得分**: {json.dumps(scores, ensure_ascii=False)}
**改进意见**: {revision}

**当前 3 个核心问题**:
{chr(10).join(f"  {i+1}. {p}" for i, p in enumerate(state.get("rewritten_problems", [])))}

请根据反馈，为每个核心问题生成改进后的搜索关键词（中文，20字以内），
重点补强得分最低的维度。

请严格按以下格式输出（每行一个，共3行）:
问题1新关键词: xxx
问题2新关键词: xxx
问题3新关键词: xxx
"""

    response = llm.invoke([
        SystemMessage(content="你是搜索策略优化专家"),
        HumanMessage(content=prompt),
    ])

    # 解析新关键词
    new_keywords = []
    for line in response.content.strip().split("\n"):
        line = line.strip()
        if ":" in line or "：" in line:
            keyword = line.split(":", 1)[-1].split("：", 1)[-1].strip()
            if keyword:
                new_keywords.append(keyword)

    # 回退: 如果解析失败，使用原问题
    original_problems = state.get("rewritten_problems", [])
    while len(new_keywords) < len(original_problems):
        idx = len(new_keywords)
        if idx < len(original_problems):
            new_keywords.append(original_problems[idx])
        else:
            break

    print(f"  ✅ 新搜索关键词:")
    for i, kw in enumerate(new_keywords):
        old = original_problems[i] if i < len(original_problems) else "N/A"
        print(f"    问题{i+1}: {old}  →  {kw}")

    # 更新 rewritten_problems 为新关键词
    state["rewritten_problems"] = new_keywords[:3]

    # ── 第2步: 根据新关键词重新搜索案例 ──
    print("\n📝 步骤 2: 根据新关键词重新搜索案例...")

    # 重新执行案例查询
    new_state = {
        **state,
        "feedback": revision,
        "iteration_count": state.get("iteration_count", 0),  # evaluation_agent已增加，保持当前值
        # 清除中间结果，准备重新搜索和分析
        "case_results": {},
        "gap_analysis": {},
        "adaptation_plan": "",
        "evaluation_scores": {},
        "final_report": "",
        "is_complete": False
    }

    return new_state
