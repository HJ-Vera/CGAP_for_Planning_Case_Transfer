"""
状态定义 — 系统全局状态类型
"""

from typing import TypedDict, List, Dict


class AgentState(TypedDict):
    """系统全局状态"""
    # 输入
    user_query: str
    target_city: str

    # 情景解构阶段
    local_context: Dict
    core_problems: List[str]
    rewritten_problems: List[str]

    # 案例查询阶段
    case_results: Dict[str, List[Dict]]

    # 差异分析阶段
    gap_analysis: Dict
    adaptation_plan: str

    # 评估阶段
    evaluation_scores: Dict
    feedback: str
    iteration_count: int

    # 最终输出
    final_report: str
    is_complete: bool
