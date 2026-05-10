"""
Plan Agent — 任务规划与评估

职责:
  1. create_plan: 分析用户问题，制定子任务执行计划
  2. evaluate_results: 评估所有子任务的执行结果，决定是否重新规划
  3. replan: 根据评估结果调整计划
"""

import json
import re
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from llm import get_llm
from config import TOKEN_LIMITS
from prompts import load_prompt
from plan_execute.pe_state import PlanExecuteState, SubTask


# 可用的子任务类型及其说明
AVAILABLE_TASK_TYPES = """
可用的子任务类型（task_type）：

1. "analyze_local_data"
   - 分析本地区域数据（加载Excel，聚类分析，生成可视化）
   - params: {}  （无需额外参数，自动使用 config 中的数据路径）
   - 产出: 本地数据分析报告文本 + 图表

2. "query_rewrite"
   - 基于本地数据分析，用 LLM 重写核心问题并生成面向「国际案例检索」的优化搜索词
   - params: {}  （无需额外参数，自动从 state 的 local_context 获取分析结果）
   - 依赖: 必须依赖 analyze_local_data（需要其产出的本地数据分析结果）
   - 产出: {"core_problems": [...], "rewritten_problems": [...], "queries_en": [...], "queries_zh": [...]}
   - 注意: 该任务会自动将重写后的搜索词存入 state，后续 search_web_cases 可直接使用

3. "search_web_cases"
   - 使用 Serper API 搜索网页案例
   - params: {"queries": ["搜索词1", "搜索词2", ...], "max_results": 20}
   - 若依赖 query_rewrite，可在 params 中写 "queries_from_rewrite": true，系统自动使用重写后的搜索词
   - 产出: 搜索结果列表

4. "search_academic"
   - 搜索学术文献（Semantic Scholar + Google Scholar + ArXiv）
   - params: {"query": "搜索关键词", "limit": 10}
   - 产出: 学术文献列表

5. "deep_research_case"
   - 对案例进行深度研究（Gap-Driven Tree Search）
   - params: {"case_index": 0}  （从依赖任务的案例列表中取第 N 个，0=第一个，1=第二个，以此类推）
   - 如果要研究第1个案例就 case_index=0，第2个就 case_index=1
   - 依赖: 必须依赖 hybrid_retrieval_and_select 或 search_web_cases（系统从其结果中自动获取案例详情）
   - 产出: 结构化案例分析报告
   - 注意: 不需要在 params 里写 title/url/snippet，系统会自动从依赖任务的结果中提取

6. "hybrid_retrieval_and_select"
   - 对搜索结果进行 BM25+SBERT 混合检索排序，并用 LLM 选择最佳案例
   - params: {"problem_cn": "中文问题", "problem_en": "英文问题"}
   - 依赖: 需要 search_web_cases 和/或 search_academic 的结果
   - 产出: 精选案例列表

7. "gap_analysis"
   - 对比全球案例与本地情境的差异，生成适应性改造方案
   - params: {"problem": "核心问题描述"}
   - 依赖: 需要 analyze_local_data 和案例研究的结果
   - 产出: 差异分析报告 + 改造方案

8. "translate_query"
   - 将中文规划问题翻译为英文，用于国际案例搜索
   - params: {"text": "要翻译的中文文本"}
   - 产出: 英文翻译

9. "custom_llm_task"
   - 自定义 LLM 任务（让 LLM 做任意分析/总结/生成）
   - params: {"system_prompt": "系统提示", "user_prompt": "用户提示"}
   - 产出: LLM 回复文本
"""


def create_plan(state: PlanExecuteState) -> PlanExecuteState:
    """
    Plan Agent: 分析用户问题，制定详细的子任务执行计划
    """
    print("\n" + "=" * 60)
    print("📋 Plan Agent 启动 — 制定执行计划")
    print("=" * 60)

    llm = get_llm(max_tokens=30000)

    user_query = state["user_query"]
    target_city = state["target_city"]

    prompt = load_prompt(
        "plan_execute/plan_agent", "01_create_plan_prompt",
        user_query=user_query,
        target_city=target_city,
        AVAILABLE_TASK_TYPES=AVAILABLE_TASK_TYPES,
    )

    messages = [
        SystemMessage(content="你是项目规划专家，只输出 JSON，不输出其他内容"),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # 解析 JSON
    plan_data = _parse_json(raw)

    if not plan_data or "tasks" not in plan_data:
        print("⚠️ Plan Agent 输出解析失败，使用默认计划")
        plan_data = _default_plan(user_query)

    reasoning = plan_data.get("reasoning", "")
    core_problems = plan_data.get("core_problems", [])
    tasks = plan_data.get("tasks", [])

    # 构建 SubTask 列表
    plan: List[SubTask] = []
    for t in tasks:
        plan.append({
            "task_id": t.get("task_id", f"task_{len(plan)+1}"),
            "task_type": t.get("task_type", "custom_llm_task"),
            "description": t.get("description", ""),
            "dependencies": t.get("dependencies", []),
            "params": t.get("params", {}),
            "status": "pending",
            "result": None,
        })

    state["plan"] = plan
    state["plan_reasoning"] = reasoning
    state["plan_version"] = state.get("plan_version", 0) + 1
    state["current_task_index"] = 0

    # 把核心问题也存进 local_context
    if core_problems:
        ctx = state.get("local_context", {})
        ctx["core_problems"] = core_problems
        state["local_context"] = ctx

    print(f"\n📋 规划思路: {reasoning}")
    print(f"🎯 核心问题: {core_problems}")
    print(f"📝 共 {len(plan)} 个子任务:")
    for i, t in enumerate(plan):
        deps = f" (依赖: {t['dependencies']})" if t["dependencies"] else ""
        print(f"  {i+1}. [{t['task_type']}] {t['description']}{deps}")

    return state


def evaluate_results(state: PlanExecuteState) -> PlanExecuteState:
    """
    Plan Agent: 评估所有子任务的执行结果
    决定是否需要重新规划（replan）
    """
    print("\n" + "=" * 60)
    print("⚖️ Plan Agent 评估执行结果")
    print("=" * 60)

    llm = get_llm(max_tokens=30000)

    # 收集所有结果的摘要
    results_summary = []
    for task in state["plan"]:
        result = task.get("result")
        result_preview = ""
        if result is None:
            result_preview = "（未执行）"
        elif isinstance(result, str):
            result_preview = result[:30000]
        elif isinstance(result, list):
            # 显示列表结果的详细信息
            if len(result) == 0:
                result_preview = "返回 0 条结果"
            else:
                # 根据列表元素类型显示不同信息
                sample_items = []
                for i, item in enumerate(result[:3]):  # 只显示前3个作为示例
                    if isinstance(item, dict):
                        # 网页搜索结果
                        if "title" in item and "url" in item:
                            title = item.get("title", "无标题")
                            url = item.get("url", "无链接")
                            sample_items.append(f"    {i+1}. {title[:80]}... ({url[:50]}...)")
                        # 学术文献结果
                        elif "title" in item and "year" in item:
                            title = item.get("title", "无标题")
                            year = item.get("year", "未知年份")
                            authors = item.get("authors", [])
                            author_str = ", ".join([a.get("name", "") for a in authors[:2]]) if authors else "未知作者"
                            if len(authors) > 2:
                                author_str += "等"
                            sample_items.append(f"    {i+1}. {title[:80]}... ({author_str}, {year})")
                        # 通用字典结果
                        else:
                            keys = list(item.keys())[:3]
                            sample_items.append(f"    {i+1}. 字典包含字段: {', '.join(keys)}")
                    else:
                        sample_items.append(f"    {i+1}. {str(item)[:100]}")

                if len(result) > 3:
                    result_preview = f"返回 {len(result)} 条结果，示例如下：\n" + "\n".join(sample_items)
                else:
                    result_preview = f"返回 {len(result)} 条结果：\n" + "\n".join(sample_items)
        elif isinstance(result, dict):
            result_preview = json.dumps(result, ensure_ascii=False)[:30000]
        else:
            result_preview = str(result)[:30000]

        results_summary.append(
            f"[{task['task_id']}] {task['task_type']}: {task['description']}\n"
            f"  状态: {task['status']}\n"
            f"  结果摘要: {result_preview}"
        )

    prompt = load_prompt(
        "plan_execute/plan_agent", "02_evaluate_results_prompt",
        user_query=state["user_query"],
        target_city=state["target_city"],
        plan_version=str(state["plan_version"]),
        replan_count=str(state.get("replan_count", 0)),
        results_summary=chr(10).join(results_summary),
    )

    messages = [
        SystemMessage(content="你是评审专家，只输出 JSON"),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    eval_data = _parse_json(response.content.strip())

    if not eval_data:
        eval_data = {"score": 0, "needs_replan": False, "strengths": [], "weaknesses": []}
    
    # 最小修正
    relevance = eval_data.get("Relevance", "0/100 - 无评语")
    completeness = eval_data.get("Completeness", "0/100 - 无评语")
    coherence = eval_data.get("Coherence", "0/100 - 无评语")
    feasibility = eval_data.get("Feasibility", "0/100 - 无评语")

    
    score = eval_data.get("score", 0)
    needs_replan = eval_data.get("needs_replan", False)
    replan_count = state.get("replan_count", 0)

    # 最多重规划 2 次
    if replan_count >= 2:
        needs_replan = False
        print(f"  ⚠️ 已达到最大重规划次数 ({replan_count})，强制通过")

    state["evaluation"] = eval_data
    state["needs_replan"] = needs_replan
    
    print(f" 问题匹配度：{relevance}")
    print(f" 信息完整性：{completeness}")
    print(f" 逻辑连贯性：{coherence}")
    print(f" 实施可行性：{feasibility}")

    print(f"\n📊 评估分数: {score}/400")
    print(f"✅ 优势: {eval_data.get('strengths', [])}")
    print(f"⚠️ 不足: {eval_data.get('weaknesses', [])}")
    print(f"🔄 需要重规划: {'是' if needs_replan else '否'}")

    # 如果需要重规划，追加额外任务
    if needs_replan:
        additional = eval_data.get("additional_tasks", [])
        replan_queries = eval_data.get("replan_queries", [])

        if additional:
            plan = state["plan"]
            for i, at in enumerate(additional):
                task_params = at.get("params", {})

                # 如果 replan_queries 有值且任务是 search_web_cases 但没有指定搜索词，
                # 用 replan_queries 覆盖
                if (at.get("task_type") == "search_web_cases"
                        and not task_params.get("queries")
                        and replan_queries):
                    task_params["queries"] = replan_queries
                    print(f"  🔧 使用 replan_queries 覆盖搜索词: {replan_queries}")

                new_task: SubTask = {
                    "task_id": f"replan_{replan_count+1}_task_{i+1}",
                    "task_type": at.get("task_type", "custom_llm_task"),
                    "description": at.get("description", "补充任务"),
                    "dependencies": [],
                    "params": task_params,
                    "status": "pending",
                    "result": None,
                }
                plan.append(new_task)
                print(f"  ➕ 追加任务: [{new_task['task_type']}] {new_task['description']}")

            state["plan"] = plan
            state["replan_count"] = replan_count + 1
            state["plan_version"] += 1
        else:
            # LLM 说要重规划但没给任务，当作通过
            state["needs_replan"] = False

    return state


def _parse_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON"""
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        raw = match.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        raw = match.group(0) if match else ""

    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _default_plan(user_query: str) -> dict:
    """解析失败时的默认计划"""
    return {
        "reasoning": "使用默认计划：本地数据分析 → 翻译 → 使用翻译结果搜索 → 差异分析",
        "core_problems": [user_query],
        "tasks": [
            {"task_id": "task_1", "task_type": "analyze_local_data",
             "description": "分析本地区域数据，了解现状与特征", "dependencies": [], "params": {}},
            {"task_id": "task_2", "task_type": "translate_query",
             "description": "将中文规划问题翻译为英文，便于国际案例搜索", "dependencies": [],
             "params": {"text": user_query}},
            {"task_id": "task_3", "task_type": "search_web_cases",
             "description": "使用翻译后的英文关键词搜索全球网页案例", "dependencies": ["task_2"],
             "params": {"queries": [f"{user_query} case study", f"{user_query} urban planning best practices", f"{user_query} international examples"], "max_results": 20}},
            {"task_id": "task_4", "task_type": "search_academic",
             "description": "使用翻译后的关键词搜索学术文献", "dependencies": ["task_2"],
             "params": {"query": f"{user_query} urban planning", "limit": 10}},
            {"task_id": "task_5", "task_type": "gap_analysis",
             "description": "对比本地数据与国际案例，提出适应性改造方案", "dependencies": ["task_1", "task_3", "task_4"],
             "params": {"problem": user_query}},
        ],
    }
