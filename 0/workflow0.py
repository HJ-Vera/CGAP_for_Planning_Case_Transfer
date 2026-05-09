"""
工作流构建 — 使用 LangGraph 构建多智能体工作流
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
from agents.case_query_agent import case_query_agent_parallel


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
    """构建 LangGraph 工作流"""
    if ckpt:
        set_checkpoint_manager(ckpt)

    workflow = StateGraph(AgentState)

    # 添加节点（每个都包装 checkpoint）
    workflow.add_node("scenario_deconstruction",
                      _wrap_with_checkpoint("scenario_deconstruction", scenario_deconstruction_agent))

    workflow.add_node("case_query_parallel",
                      _wrap_with_checkpoint("case_query_parallel", 
                      lambda s: {**s, "case_results": {
                          # 保留已有的case_results
                          **s.get("case_results", {}),
                          # 并行处理三个问题，并将结果添加到case_results
                          **{f"problem_{idx+1}": cases 
                             for idx, cases in case_query_agent_parallel(s, [0, 1, 2]).items()}
                      }}))

    workflow.add_node("gap_analysis",
                      _wrap_with_checkpoint("gap_analysis", gap_analysis_agent))
    workflow.add_node("evaluation",
                      _wrap_with_checkpoint("evaluation", evaluation_agent))
    workflow.add_node("feedback_loop",
                      _wrap_with_checkpoint("feedback_loop", feedback_loop))
    workflow.add_node("generate_report",
                      _wrap_with_checkpoint("generate_report", generate_final_report))

    # 流程定义（和原来一样）
    workflow.set_entry_point("scenario_deconstruction")
    workflow.add_edge("scenario_deconstruction", "case_query_parallel")  # 原来指向case_query_1
    workflow.add_edge("case_query_parallel", "gap_analysis")  # 原来从case_query_3到gap_analysis
    workflow.add_edge("gap_analysis", "evaluation")

    workflow.add_conditional_edges(
        "evaluation",
        should_continue,
        {
            "generate_report": "generate_report",
            "feedback_loop": "feedback_loop"
        }
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
        print("\n工作流节点:")
        print("1. scenario_deconstruction (情景解构)")
        print("2. case_query_parallel (案例查询-并行处理3个问题)")  # 更新这里
        print("3. gap_analysis (差异分析)")
        print("4. evaluation (评审)")
        print("5. feedback_loop (反馈循环) / generate_report (生成报告)")
