"""
Plan-Execute 模式的状态定义
"""

from typing import TypedDict, List, Dict, Optional, Any


class SubTask(TypedDict):
    """单个子任务"""
    task_id: str            # 唯一标识，如 "task_1"
    task_type: str          # 任务类型: analyze_local_data / search_cases / deep_research / gap_analysis / custom
    description: str        # 任务描述（自然语言）
    dependencies: List[str] # 依赖的前置任务 ID 列表
    params: Dict            # 传给 sub agent 的参数
    status: str             # pending / running / done / failed
    result: Any             # 执行结果


class PlanExecuteState(TypedDict):
    """Plan-Execute 全局状态"""
    # 输入
    user_query: str
    target_city: str

    # Plan Agent 产出
    plan: List[SubTask]          # 当前任务计划
    plan_reasoning: str          # Plan Agent 的规划思路
    plan_version: int            # 计划版本号（每次重规划 +1）

    # 执行过程
    current_task_index: int      # 当前正在执行的任务索引
    completed_results: Dict[str, Any]  # task_id -> result 的映射

    # 本地数据（首次分析后缓存）
    local_context: Dict
    matched_area: str

    # 评估
    evaluation: Dict             # Plan Agent 的评估结果
    needs_replan: bool           # 是否需要重新规划
    replan_count: int            # 已重规划次数

    # 最终输出
    final_report: str
    is_complete: bool
