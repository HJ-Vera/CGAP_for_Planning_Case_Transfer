"""
工作流构建 — 使用 LangGraph 构建多智能体工作流
三个 case_query 节点现在并行执行（Fan-out → Fan-in）
"""

from langgraph.graph import StateGraph, END

from state import AgentState
from agents.scenario_agent import scenario_deconstruction_agent
from agents.case_query_agent import case_query_agent
from agents.gap_analysis_agent import gap_analysis_agent
from agents.evaluation_agent import evaluation_agent
from agents.feedback import feedback_loop
from agents.report_generator import generate_final_report
from router import should_continue

from checkpoint import CheckpointManager

# 模块级变量，运行时由 main.py 注入
_ckpt_manager: CheckpointManager = None


def set_checkpoint_manager(mgr: CheckpointManager):
    global _ckpt_manager
    _ckpt_manager = mgr


def _wrap_with_checkpoint(step_name: str, fn):
    """包装 agent 函数，执行后自动保存 checkpoint"""
    def wrapped(state):
        result = fn(state)
        if _ckpt_manager:
            _ckpt_manager.save(result, step_name)
        return result
    return wrapped


def build_workflow(ckpt: CheckpointManager = None):
    """
    构建 LangGraph 工作流。

    案例查询阶段采用 Fan-out / Fan-in 并行模式：
      scenario_deconstruction
           ├─▶ case_query_1 ─┐
           ├─▶ case_query_2 ─┼─▶ gap_analysis ─▶ ...
           └─▶ case_query_3 ─┘

    三个节点各自只返回局部状态 {"case_results": {"problem_N": [...]}}，
    由 state.py 中的 _merge_case_results reducer 自动合并。
    LangGraph 会等待全部三个分支完成后再推进到 gap_analysis。
    """
    if ckpt:
        set_checkpoint_manager(ckpt)

    workflow = StateGraph(AgentState)

    # ── 节点注册 ────────────────────────────────────────────────────

    workflow.add_node(
        "scenario_deconstruction",
        _wrap_with_checkpoint("scenario_deconstruction", scenario_deconstruction_agent),
    )

    # 每个 case_query 节点只返回自己负责的 case_results 片段
    # （不再返回完整 state），reducer 负责合并
    workflow.add_node(
        "case_query_1",
        _wrap_with_checkpoint(
            "case_query_1",
            lambda s: {"case_results": {"problem_1": case_query_agent(s, 0)}},
        ),
    )
    workflow.add_node(
        "case_query_2",
        _wrap_with_checkpoint(
            "case_query_2",
            lambda s: {"case_results": {"problem_2": case_query_agent(s, 1)}},
        ),
    )
    workflow.add_node(
        "case_query_3",
        _wrap_with_checkpoint(
            "case_query_3",
            lambda s: {"case_results": {"problem_3": case_query_agent(s, 2)}},
        ),
    )

    workflow.add_node(
        "gap_analysis",
        _wrap_with_checkpoint("gap_analysis", gap_analysis_agent),
    )
    workflow.add_node(
        "evaluation",
        _wrap_with_checkpoint("evaluation", evaluation_agent),
    )
    workflow.add_node(
        "feedback_loop",
        _wrap_with_checkpoint("feedback_loop", feedback_loop),
    )
    workflow.add_node(
        "generate_report",
        _wrap_with_checkpoint("generate_report", generate_final_report),
    )

    # ── 流程定义 ─────────────────────────────────────────────────────

    workflow.set_entry_point("scenario_deconstruction")

    # Fan-out：情景解构完成后同时触发三个案例查询
    workflow.add_edge("scenario_deconstruction", "case_query_1")
    workflow.add_edge("scenario_deconstruction", "case_query_2")
    workflow.add_edge("scenario_deconstruction", "case_query_3")

    # Fan-in：三个分支全部完成后汇入差异分析
    # LangGraph 会自动等待所有入边的节点都执行完毕
    workflow.add_edge("case_query_1", "gap_analysis")
    workflow.add_edge("case_query_2", "gap_analysis")
    workflow.add_edge("case_query_3", "gap_analysis")

    workflow.add_edge("gap_analysis", "evaluation")

    workflow.add_conditional_edges(
        "evaluation",
        should_continue,
        {
            "generate_report": "generate_report",
            "feedback_loop": "feedback_loop",
        },
    )

    workflow.add_edge("feedback_loop", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()


def visualize_workflow(app):
    """可视化工作流图结构（终端文字版）"""
    try:
        mermaid_text = app.get_graph().draw_mermaid()
        print("\n📊 工作流 Mermaid 图:")
        print(mermaid_text)
    except Exception as e:
        print(f"无法可视化图结构: {e}")
        print("\n工作流节点（并行版）:")
        print("1. scenario_deconstruction (情景解构)")
        print("2a. case_query_1  ┐")
        print("2b. case_query_2  ├─ 并行执行")
        print("2c. case_query_3  ┘")
        print("3. gap_analysis (差异分析)")
        print("4. evaluation (评审)")
        print("5. feedback_loop (反馈循环) / generate_report (生成报告)")
