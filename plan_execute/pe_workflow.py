"""
Plan-Execute LangGraph 工作流构建

流程:
  create_plan → execute_tasks → evaluate_results → [replan → execute → evaluate]* → generate_report

与 workflow 模式的区别:
  - workflow: 固定的 7 步流水线
  - plan-execute: Plan Agent 动态规划 → Sub Agent 执行 → Plan Agent 评估 → 可能重规划
"""

from langgraph.graph import StateGraph, END

from plan_execute.pe_state import PlanExecuteState
from plan_execute.plan_agent import create_plan, evaluate_results
from plan_execute.sub_agents import execute_task
from plan_execute.pe_report import generate_report


def _execute_all_tasks(state: PlanExecuteState) -> PlanExecuteState:
    """
    按依赖顺序执行所有 pending 状态的子任务
    """
    print("\n" + "=" * 60)
    print("🚀 Sub Agents 执行子任务")
    print("=" * 60)

    plan = state["plan"]
    completed = state.get("completed_results", {})

    # 已完成的任务 ID 集合
    done_ids = {t["task_id"] for t in plan if t["status"] == "done"}

    # 拓扑排序执行
    max_iterations = len(plan) * 2  # 防止死循环
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        progress = False

        for i, task in enumerate(plan):
            if task["status"] != "pending":
                continue

            # 检查依赖是否满足
            deps = task.get("dependencies", [])
            if all(d in done_ids for d in deps):
                task["status"] = "running"
                print(f"\n{'─' * 50}")
                print(f"📌 任务 {task['task_id']}: {task['description']}")
                print(f"   类型: {task['task_type']} | 依赖: {deps or '无'}")

                try:
                    result = execute_task(state, task)
                    task["result"] = result
                    task["status"] = "done"
                    completed[task["task_id"]] = result
                    done_ids.add(task["task_id"])
                    print(f"   ✅ 完成")
                    progress = True
                except Exception as e:
                    task["status"] = "failed"
                    task["result"] = {"error": str(e)}
                    completed[task["task_id"]] = {"error": str(e)}
                    done_ids.add(task["task_id"])  # 失败也算完成，不阻塞后续
                    print(f"   ❌ 失败: {str(e)[:100]}")
                    progress = True

        # 如果一轮下来没有新任务可执行，退出
        if not progress:
            break

    state["completed_results"] = completed

    # 统计
    total = len(plan)
    done = sum(1 for t in plan if t["status"] == "done")
    failed = sum(1 for t in plan if t["status"] == "failed")
    pending = sum(1 for t in plan if t["status"] == "pending")

    print(f"\n📊 执行统计: 共 {total} | 完成 {done} | 失败 {failed} | 未执行 {pending}")

    return state


def _should_continue(state: PlanExecuteState) -> str:
    """评估后路由：重新执行 or 生成报告"""
    if state.get("needs_replan", False):
        return "execute_tasks"
    else:
        return "generate_report"


def build_plan_execute_workflow():
    """构建 Plan-Execute LangGraph 工作流"""
    workflow = StateGraph(PlanExecuteState)

    # 节点
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("execute_tasks", _execute_all_tasks)
    workflow.add_node("evaluate_results", evaluate_results)
    workflow.add_node("generate_report", generate_report)

    # 流程
    workflow.set_entry_point("create_plan")
    workflow.add_edge("create_plan", "execute_tasks")
    workflow.add_edge("execute_tasks", "evaluate_results")

    # 条件分支: 评估后决定是否重新执行
    workflow.add_conditional_edges(
        "evaluate_results",
        _should_continue,
        {
            "execute_tasks": "execute_tasks",
            "generate_report": "generate_report",
        }
    )

    workflow.add_edge("generate_report", END)

    return workflow.compile()


def visualize_pe_workflow(app):
    """可视化 Plan-Execute 工作流"""
    try:
        mermaid = app.get_graph().draw_mermaid()
        print("\n📊 Plan-Execute 工作流图:")
        print(mermaid)
    except Exception as e:
        print(f"无法可视化: {e}")
        print("\nPlan-Execute 工作流:")
        print("  create_plan → execute_tasks → evaluate_results")
        print("                    ↑                  ↓")
        print("                    └── [needs_replan] ─┘")
        print("                                        ↓")
        print("                               generate_report → END")
