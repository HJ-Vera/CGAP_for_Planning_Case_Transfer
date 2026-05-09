"""
多智能体城市规划案例分析系统 — 主入口

使用方法:
    1. 在 config.py 中设置 API Keys
    2. 将数据文件放入 data/ 目录
    3. 运行: python main.py

系统功能:
    1. 智能解读本地数据表格
    2. 全球案例搜索（支持中英文）
    3. 多源学术文献检索
    4. 深度信息提取和补充
    5. 差异分析和本地化改造
    6. 风险评估和实施建议
"""

import os
import sys
import warnings
import logging
import traceback

# 配置警告和日志
warnings.filterwarnings('ignore', category=UserWarning, module='bs4')
logging.getLogger('bs4.dammit').setLevel(logging.ERROR)

# 如果在无 GUI 的服务器上运行，取消下面这行注释以避免 plt.show() 阻塞
# import matplotlib; matplotlib.use('Agg')

import config
from workflow import build_workflow, visualize_workflow
from tools.search import (
    search_semantic_scholar,
    search_google_scholar_alternative,
    search_arxiv,
)


def test_academic_search():
    """
    测试学术搜索功能
    用于诊断 Semantic Scholar 连接问题
    """
    print("=" * 60)
    print("🔬 学术搜索功能测试")
    print("=" * 60)

    test_query = "urban planning transportation"

    print(f"\n测试查询: {test_query}\n")

    # 测试 Semantic Scholar
    print("1️⃣ 测试 Semantic Scholar...")
    print("-" * 40)
    ss_results = search_semantic_scholar(test_query, limit=5)
    if ss_results:
        print(f"✅ 成功! 找到 {len(ss_results)} 篇文献")
        print(f"示例: {ss_results[0].get('title', 'N/A')[:150]}")
    else:
        print("❌ Semantic Scholar 不可用")

    print("\n2️⃣ 测试 Google Scholar (via Serper)...")
    print("-" * 40)
    if config.SERPER_API_KEY and config.SERPER_API_KEY != "your_serper_api_key_here":
        gs_results = search_google_scholar_alternative(test_query, max_results=5)
        if gs_results:
            print(f"✅ 成功! 找到 {len(gs_results)} 篇文献")
            print(f"示例: {gs_results[0].get('title', 'N/A')[:150]}")
        else:
            print("❌ Google Scholar 不可用")
    else:
        print("⚠️ 需要设置 SERPER_API_KEY")
        gs_results = []

    print("\n3️⃣ 测试 ArXiv...")
    print("-" * 40)
    arxiv_results = search_arxiv(test_query, max_results=5)
    if arxiv_results:
        print(f"✅ 成功! 找到 {len(arxiv_results)} 篇预印本")
        print(f"示例: {arxiv_results[0].get('title', 'N/A')[:80]}")
    else:
        print("❌ ArXiv 不可用")

    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)

    available_sources = []
    if ss_results:
        available_sources.append("Semantic Scholar")
    if config.SERPER_API_KEY and gs_results:
        available_sources.append("Google Scholar")
    if arxiv_results:
        available_sources.append("ArXiv")

    if available_sources:
        print(f"✅ 可用的学术数据库: {', '.join(available_sources)}")
    else:
        print("❌ 没有可用的学术数据库")
        print("\n💡 可能的原因:")
        print("  1. 网络连接问题")
        print("  2. Serper API Key 未设置或无效")

    print()


def main(user_query: str, resume_run_id: str = None):
    """主函数，支持断点恢复"""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("🏙️  多智能体城市规划案例分析系统")
    print("=" * 60)

    from checkpoint import CheckpointManager

    # 判断是否恢复
    if resume_run_id:
        ckpt = CheckpointManager(run_id=resume_run_id)
        last_step, saved_state = ckpt.load_latest()
        if saved_state:
            print(f"\n🔄 从断点恢复: run_id={resume_run_id}, 上次完成步骤={last_step}")
            # 从已完成步骤的下一步开始
            return _resume_from_checkpoint(saved_state, last_step, ckpt)
        else:
            print(f"⚠️ 未找到 run_id={resume_run_id} 的 checkpoint，将从头开始")

    ckpt = CheckpointManager()
    print(f"\n📝 用户问题: {user_query}")
    print(f"🎯 目标城市: 香港")
    print(f"📁 Run ID: {ckpt.run_id}")
    print(f"💡 如果中途失败，可用以下命令恢复:")
    print(f"   python main.py --resume {ckpt.run_id}\n")

    # 初始化状态
    initial_state = {
        "user_query": user_query,
        "target_city": "香港",
        "local_context": {},
        "core_problems": [],
        "rewritten_problems": [],
        "case_results": {},
        "gap_analysis": {},
        "adaptation_plan": "",
        "evaluation_scores": {},
        "feedback": "",
        "iteration_count": 0,
        "final_report": "",
        "is_complete": False
    }

    app = build_workflow(ckpt=ckpt)
    visualize_workflow(app)

    try:
        result = app.invoke(initial_state)

        print("\n" + "=" * 60)
        print("📊 最终报告")
        print("=" * 60 + "\n")
        print(result["final_report"])

        report_path = os.path.join(config.OUTPUT_DIR, "planning_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(result["final_report"])
        print(f"\n💾 报告已保存至: {report_path}")

        return result

    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        print(f"💡 可以用以下命令从断点恢复:")
        print(f"   python main.py --resume {ckpt.run_id}")
        traceback.print_exc()
        return None


def _resume_from_checkpoint(saved_state: dict, last_step: str, ckpt):
    """从 checkpoint 恢复执行"""
    # 定义步骤顺序
    STEP_ORDER = [
        "scenario_deconstruction",
        "case_query_parallel",
        "gap_analysis", "evaluation",
        "generate_report",
    ]

    if last_step not in STEP_ORDER:
        print(f"⚠️ 未知步骤 {last_step}，从头开始")
        return None

    last_idx = STEP_ORDER.index(last_step)
    remaining = STEP_ORDER[last_idx + 1:]

    if not remaining:
        print("✅ 上次已经完成了所有步骤")
        return saved_state

    print(f"⏭️ 跳过已完成步骤: {STEP_ORDER[:last_idx + 1]}")
    print(f"▶️ 将从 {remaining[0]} 继续执行")

    # 重新构建工作流并从断点状态运行
    app = build_workflow(ckpt=ckpt)

    try:
        result = app.invoke(saved_state)
        print("\n" + "=" * 60)
        print("📊 最终报告")
        print("=" * 60 + "\n")
        print(result["final_report"])
        return result
    except Exception as e:
        print(f"\n❌ 恢复执行失败: {e}")
        print(f"💡 再次恢复: python main.py --resume {ckpt.run_id}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 多智能体城市规划案例分析系统")
    print("=" * 60)
    print("\n📋 功能说明:")
    print("1. 智能数据表分析")
    print("2. 全球案例搜索（英文+中文）")
    print("3. 多源学术文献（3个数据库）")
    print("4. 深度信息提取与补充")
    print("5. 风险与代价评估")
    print("\n⚙️  准备工作:")
    print("1. 将数据文件放入 data/ 目录")
    print("2. 在 config.py 中设置 API Keys")
    print("3. pip install -r requirements.txt")
    print("\n" + "=" * 60 + "\n")

    # ===== 在此修改你的查询 =====
    USER_QUERY = "新田科技城城市规划相关案例分析"

    # 可选: 先测试学术搜索
    # test_academic_search()

    # 运行系统
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 run_id 的断点恢复")
    parser.add_argument("--list-runs", action="store_true",
                        help="列出所有可恢复的运行")
    parser.add_argument("--query", type=str, default=USER_QUERY,
                        help="城市规划问题")
    args = parser.parse_args()

    if args.list_runs:
        from checkpoint import CheckpointManager
        runs = CheckpointManager.list_runs()
        if runs:
            print("📋 可恢复的运行:")
            for r in runs:
                print(f"  {r['run_id']} | 步骤: {r['last_step']} | 时间: {r['timestamp']}")
        else:
            print("没有可恢复的运行")
    else:
        result = main(args.query, resume_run_id=args.resume)