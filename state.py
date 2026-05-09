"""
状态定义 — 系统全局状态类型
"""

from typing import TypedDict, List, Dict, Annotated


def _merge_case_results(existing: Dict, update: Dict) -> Dict:
    """
    合并来自并行 case_query 节点的局部 case_results 更新。
    LangGraph 并行分支写同一个 key 时，会依次调用此 reducer。
    """
    return {**existing, **update}


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
    # Annotated + reducer：允许三个并行节点分别写入 problem_1/2/3，自动合并
    case_results: Annotated[Dict[str, List[Dict]], _merge_case_results]

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
