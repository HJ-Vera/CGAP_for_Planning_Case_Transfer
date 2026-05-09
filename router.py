"""
路由函数 — 工作流条件判断
"""

from state import AgentState


def should_continue(state: AgentState) -> str:
    """决定工作流是否继续"""
    if state.get("is_complete", False):
        return "generate_report"
    if state.get("iteration_count", 0) >= 3:   # 最多重试3次
        return "generate_report"
    return "feedback_loop"
