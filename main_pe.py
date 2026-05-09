"""
Plan-Execute 模式 — 主入口

与 main.py (workflow 模式) 并列，共享 tools/ 和 config。

使用方法:
    python main_pe.py
    python main_pe.py --query "你的城市规划问题"
"""

import os
import sys
import asyncio
import warnings
import logging
import traceback
import argparse

# 配置
warnings.filterwarnings("ignore", category=UserWarning, module="bs4")
logging.getLogger("bs4.dammit").setLevel(logging.ERROR)

import matplotlib
matplotlib.use("Agg")  # 非交互后端

import config
from plan_execute.pe_workflow import build_plan_execute_workflow, visualize_pe_workflow


async def main(user_query: str):
    """运行 Plan-Execute 模式"""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("🏙️  多智能体城市规划系统 — Plan-Execute 模式")
    print("=" * 60)
    print(f"\n📝 用户问题: {user_query}")
    print(f"🎯 目标城市: 香港")
    print(f"🔄 执行模式: Plan → Execute → Evaluate → [Replan] → Report")
    print()

    # 初始状态
    initial_state = {
        "user_query": user_query,
        "target_city": "香港",
        "plan": [],
        "plan_reasoning": "",
        "plan_version": 0,
        "current_task_index": 0,
        "completed_results": {},
        "local_context": {},
        "matched_area": "",
        "evaluation": {},
        "needs_replan": False,
        "replan_count": 0,
        "final_report": "",
        "is_complete": False,
    }

    # 构建工作流
    app = build_plan_execute_workflow()
    visualize_pe_workflow(app)

    try:
        result = await app.ainvoke(initial_state)

        # 输出最终报告
        print("\n" + "=" * 60)
        print("📊 最终报告")
        print("=" * 60 + "\n")
        print(result["final_report"])

        # 统计信息
        plan = result.get("plan", [])
        done = sum(1 for t in plan if t["status"] == "done")
        failed = sum(1 for t in plan if t["status"] == "failed")

        print("\n" + "=" * 60)
        print("📊 执行统计")
        print("=" * 60)
        print(f"  计划版本: v{result.get('plan_version', 1)}")
        print(f"  重规划次数: {result.get('replan_count', 0)}")
        print(f"  子任务总数: {len(plan)}")
        print(f"  成功: {done} | 失败: {failed}")
        print(f"  评估分数: {result.get('evaluation', {}).get('score', 'N/A')}/100")
        print(f"  报告长度: {len(result.get('final_report', ''))} 字符")

        return result

    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plan-Execute 城市规划案例分析")
    parser.add_argument("--query", type=str, default="新田科技城城市规划相关案例分析",
                        help="城市规划问题")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 多智能体城市规划系统 — Plan-Execute 模式")
    print("=" * 60)
    print()
    print("📋 与 Workflow 模式的区别:")
    print("  Workflow:      固定 7 步流水线")
    print("  Plan-Execute:  LLM 动态规划 → 执行 → 评估 → 可能重规划")
    print()
    print("共享组件: tools/, config.py, llm.py")
    print("=" * 60 + "\n")

    result = asyncio.run(main(args.query))
