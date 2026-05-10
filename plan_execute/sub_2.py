"""
Sub Agents — 子任务执行器

根据 task_type 调度到对应的工具函数执行，
所有工具来自共享的 tools/ 目录。
"""

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from prompts import load_prompt
from llm import get_llm
from config import (
    TOKEN_LIMITS, HONGKONG_DATA_FILE, KNOWLEDGE_BASE_MD,
    PLANNING_KNOWLEDGE_MD, OUTPUT_DIR, GEOJSON_PATH,
    BM25_CANDIDATE_K, FINAL_CASE_COUNT, HYBRID_ALPHA, SBERT_MODEL_NAME,
    SKIP_SEMANTIC_SCHOLAR,
)
from tools.data_loader import load_hongkong_data, read_md_file
from tools.data_analysis import analyze_regional_data
from tools.search import (
    search_serper, search_academic_sources,
    search_semantic_scholar, search_google_scholar_alternative, search_arxiv,
)
from tools.web_fetcher import fetch_webpage_content, fetch_webpage_content_alternative
from tools.retrieval import HybridRetriever, _deduplicate
from tools.deep_research import deep_case_research

from plan_execute.pe_state import PlanExecuteState, SubTask


def execute_task(state: PlanExecuteState, task: SubTask) -> Any:
    """
    根据 task_type 路由到对应的执行函数
    返回执行结果（任意类型）
    """
    task_type = task["task_type"]
    params = task.get("params", {})
    description = task.get("description", "")

    print(f"\n  🔧 执行子任务: [{task_type}] {description}")

    dispatcher = {
        "analyze_local_data":         _exec_analyze_local_data,
        "search_web_cases":           _exec_search_web_cases,
        "search_academic":            _exec_search_academic,
        "deep_research_case":         _exec_deep_research_case,
        "hybrid_retrieval_and_select": _exec_hybrid_retrieval,
        "gap_analysis":               _exec_gap_analysis,
        "translate_query":            _exec_translate,
        "custom_llm_task":            _exec_custom_llm,
    }

    executor = dispatcher.get(task_type, _exec_custom_llm)
    return executor(state, task)


# ═══════════════════════════════════════════════════════════
# 各类型任务的执行函数
# ═══════════════════════════════════════════════════════════

def _exec_analyze_local_data(state: PlanExecuteState, task: SubTask) -> Dict:
    """分析本地区域数据"""
    print("    📊 加载并分析本地数据...")

    llm = get_llm(max_tokens=TOKEN_LIMITS.get("scenario_agent", 50000))
    user_query = state["user_query"]
    target_city = state["target_city"]

    # 加载数据
    df = load_hongkong_data(HONGKONG_DATA_FILE)

    # 识别区域列
    area_column = None
    for col in df.columns:
        col_str = str(col)
        if any(kw in col_str for kw in ["区", "區", "district", "area", "選區", "地区", "region"]):
            area_column = col
            break
    if area_column is None:
        area_column = df.columns[0]

    all_areas = df[area_column].astype(str).tolist()

    # ── 用和 scenario_agent 一样的完整 prompt 做区域匹配 ──
    area_matching_prompt = load_prompt(
        "plan_execute/sub_2", "01_area_matching_prompt",
        user_query=user_query,
        target_city=target_city,
        total_areas=str(len(all_areas)),
        all_areas=", ".join(all_areas),
    )

    resp = llm.invoke([
        SystemMessage(content="你是香港地理信息匹配专家，熟悉香港18个区议会分区及其下辖选区"),
        HumanMessage(content=area_matching_prompt),
    ])

    response_text = resp.content.strip()
    print(f"    AI匹配回应: {response_text[:200]}")

    # 从回应中提取匹配到的区域（可能多个）
    matched_areas = []
    for area in all_areas:
        if area in response_text:
            matched_areas.append(area)

    # 模糊匹配兜底
    if not matched_areas:
        for area in all_areas:
            # 从用户查询中提取可能的区名关键词
            for keyword in ["油麻地", "旺角", "尖沙咀", "大角咀", "佐敦",
                            "深水埗", "长沙湾", "荃湾", "沙田", "大埔",
                            "元朗", "屯門", "將軍澳", "觀塘", "黃大仙"]:
                if keyword in user_query and keyword in str(area):
                    matched_areas.append(area)
        matched_areas = list(dict.fromkeys(matched_areas))  # 去重保序

    if not matched_areas:
        print(f"    ⚠️ 未匹配到任何区域，将使用第一个区域作为默认")
        matched_areas = [all_areas[0]]

    # 构建 matched_area 显示名
    if len(matched_areas) > 1:
        matched_area = "、".join(matched_areas[:5]) + (f"等{len(matched_areas)}个区域" if len(matched_areas) > 5 else "")
    else:
        matched_area = matched_areas[0]

    print(f"    ✅ 匹配到 {len(matched_areas)} 个区域: {', '.join(matched_areas[:5])}")

    # ── 多区域平均值处理（和 scenario_agent 一致）──
    import numpy as np

    if len(matched_areas) > 1:
        matched_rows = df[df[area_column].astype(str).isin(matched_areas)]
        if not matched_rows.empty:
            numeric_cols = matched_rows.select_dtypes(include=[np.number]).columns
            avg_data = {area_column: matched_area}
            for col in numeric_cols:
                avg_data[col] = matched_rows[col].mean()
            non_numeric_cols = matched_rows.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                if col != area_column:
                    avg_data[col] = matched_rows[col].iloc[0] if not matched_rows[col].empty else ""
            local_info = avg_data
        else:
            local_info = {area_column: matched_area, "备注": "未找到数据"}
    else:
        local_data_row = df[df[area_column].astype(str) == matched_area]
        if not local_data_row.empty:
            local_info = local_data_row.iloc[0].to_dict()
        else:
            local_data_row = df[df[area_column].astype(str).str.contains(matched_areas[0], na=False)]
            local_info = local_data_row.iloc[0].to_dict() if not local_data_row.empty else {area_column: matched_area}

    # 聚类分析
    result_text = analyze_regional_data(df, matched_area, output_dir=OUTPUT_DIR, matched_areas=matched_areas)

    # 存入 state
    state["matched_area"] = matched_area
    state["local_context"] = {
        "matched_area": matched_area,
        "matched_areas": matched_areas,
        "local_data": local_info,
        "area_column": area_column,
        "all_areas": all_areas,
        "analysis_text": result_text,
    }

    return {
        "matched_area": matched_area,
        "matched_areas": matched_areas,
        "analysis_text": result_text,
        "local_info_keys": list(local_info.keys())[:10],
    }


def _exec_search_web_cases(state: PlanExecuteState, task: SubTask) -> List[Dict]:
    """搜索网页案例"""
    params = task.get("params", {})
    queries = params.get("queries", [state["user_query"] + " case study"])
    max_results = params.get("max_results", 20)

    all_results = []
    for q in queries:
        results = search_serper(q, max_results=max_results)
        all_results.extend(results)
        print(f"    🔍 搜索 '{q[:40]}...': {len(results)} 个结果")

    all_results = _deduplicate(all_results)
    print(f"    ✅ 去重后共 {len(all_results)} 个网页结果")
    return all_results


def _exec_search_academic(state: PlanExecuteState, task: SubTask) -> List[Dict]:
    """搜索学术文献"""
    params = task.get("params", {})
    query = params.get("query", state["user_query"])
    limit = params.get("limit", 10)

    results = search_academic_sources(query, limit=limit)
    print(f"    ✅ 学术搜索: {len(results)} 篇文献")
    return results


def _exec_deep_research_case(state: PlanExecuteState, task: SubTask) -> Dict:
    """
    对案例进行深度研究。
    
    案例来源优先级:
      1. params 里直接指定了 title/url（Plan Agent 预设）
      2. params 里指定了 case_index，从依赖任务的结果列表中按索引取
      3. 从依赖任务的结果列表中自动取第一个未研究过的案例
    """
    params = task.get("params", {})
    title = params.get("title", "")
    url = params.get("url", "")
    snippet = params.get("snippet", "")

    completed = state.get("completed_results", {})
    case_index = params.get("case_index", None)

    # ── 如果 params 里没有有效的案例信息，从依赖任务的结果中获取 ──
    if not title or not url:
        # 收集所有依赖任务产出的案例列表
        candidate_cases = []
        for dep_id in task.get("dependencies", []):
            dep_result = completed.get(dep_id)
            if isinstance(dep_result, list):
                for item in dep_result:
                    if isinstance(item, dict) and item.get("title"):
                        candidate_cases.append(item)

        # 如果依赖里没找到，扫描所有已完成任务中的案例列表
        if not candidate_cases:
            for tid, result in completed.items():
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict) and item.get("title") and item.get("url"):
                            candidate_cases.append(item)

        if not candidate_cases:
            print("    ⚠️ 没有可用的案例数据")
            return {"title": "未知案例", "error": "没有可用的案例数据，前置搜索/筛选任务可能失败"}

        # 过滤掉已经研究过的案例
        researched_titles = set()
        for tid, result in completed.items():
            if isinstance(result, dict) and result.get("title") and result.get("final_report"):
                researched_titles.add(result["title"])

        unresearched = [c for c in candidate_cases if c.get("title") not in researched_titles]
        pool = unresearched if unresearched else candidate_cases

        # 按 case_index 或顺序取
        if case_index is not None and case_index < len(pool):
            selected = pool[case_index]
        else:
            selected = pool[0]

        title = selected.get("title", "未知案例")
        url = selected.get("url", selected.get("link", ""))
        snippet = selected.get("snippet", selected.get("abstract", ""))
        print(f"    📌 从前置任务中获取案例: {title}")

    # ── 抓取网页内容 ──
    llm = get_llm(max_tokens=TOKEN_LIMITS.get("case_query_agent", 50000))
    chat = get_llm(type="chat")
    initial_content = ""
    if url:
        try:
            initial_content = fetch_webpage_content(url, max_length=30000)
            if "内容解析失败" in initial_content or len(initial_content) < 2000:
                print(f"    ⚠️ 主方法效果不佳，尝试备用方法...")
                initial_content = fetch_webpage_content_alternative(url, max_length=30000)

                if initial_content and not initial_content.startswith("网页访问"):
                    print(f"    ✅ 抓取网页内容: {len(initial_content)} 字符")
                else:
                    print(f"    ⚠️ {initial_content}")
                    initial_content = snippet

            print(f"    📄 抓取内容: {len(initial_content)} 字符")
        except Exception as e:
            print(f"    ⚠️ 抓取失败: {str(e)[:50]}")
            initial_content = snippet

    if not initial_content or len(initial_content) < 2000:
        initial_content = snippet
        if len(initial_content) < 2000:
            # 内容不足时，用 LLM 基于标题做基础研究，而不是直接跳过
            print(f"    ⚠️ 网页内容不足，使用 LLM 基于标题进行基础研究")
            resp = llm.invoke([
                SystemMessage(content="你是城市规划案例研究专家"),
                HumanMessage(content=load_prompt(
                    "plan_execute/sub_2", "05_fallback_prompt",
                    title=title,
                    snippet=snippet,
                    url=url,
                )),
            ])
            return {
                "title": title,
                "extraction": {},
                "final_report": resp.content,
                "loop_count": 0,
                "note": "基于 LLM 知识的基础研究（网页内容不足）",
            }

    # ── 深度研究 ──
    case = {"title": title, "url": url, "snippet": snippet}
    result = deep_case_research(
        case=case,
        initial_content=initial_content,
        llm=llm,
        chat=chat,
        search_serper=search_serper,
        fetch_webpage_content=fetch_webpage_content,
        max_loops=3,
    )

    return {
        "title": title,
        "extraction": result.get("extraction", {}),
        "final_report": result.get("final_report", ""),
        "loop_count": result.get("loop_count", 0),
    }

def _exec_hybrid_retrieval(state: PlanExecuteState, task: SubTask) -> List[Dict]:
    """混合检索排序 + LLM 选择"""
    params = task.get("params", {})
    problem_cn = params.get("problem_cn", state["user_query"])
    problem_en = params.get("problem_en", "")

    # 收集所有前置搜索结果
    all_results = []
    completed = state.get("completed_results", {})

    for dep_id in task.get("dependencies", []):
        dep_result = completed.get(dep_id)
        if isinstance(dep_result, list):
            for r in dep_result:
                entry = {
                    "title": r.get("title", ""),
                    "url": r.get("url", r.get("link", "")),
                    "snippet": r.get("snippet") or r.get("abstract") or "",
                    "language": "zh" if any('\u4e00' <= c <= '\u9fff' for c in r.get("title", "")) else "en",
                    "source": r.get("source", "search"),
                }
                all_results.append(entry)

    if not all_results:
        print("    ⚠️ 没有可用的搜索结果进行检索排序")
        return []

    all_results = _deduplicate(all_results)
    print(f"    📊 待排序: {len(all_results)} 个结果")

    if len(all_results) < 3:
        return all_results

    # BM25 + SBERT
    doc_texts = [f"{r.get('title', '')} {r.get('snippet', '')}" for r in all_results]
    doc_languages = [r.get("language", "en") for r in all_results]

    retriever = HybridRetriever(alpha=HYBRID_ALPHA, sbert_model_name=SBERT_MODEL_NAME)
    retriever.fit(doc_texts, doc_languages)

    queries = {"en": problem_en or problem_cn, "zh": problem_cn}
    top_k = min(BM25_CANDIDATE_K, len(all_results))
    retrieval_results = retriever.retrieve(queries=queries, top_k=top_k)

    # 标记分数
    # 标记所有分数（修改后）
    for idx, hybrid_score, bm25_norm, sbert_sim in retrieval_results:
        all_results[idx]["hybrid_score"] = hybrid_score
        all_results[idx]["bm25_score"] = bm25_norm  # 新增
        all_results[idx]["sbert_similarity"] = sbert_sim  # 新增

    candidates = [all_results[idx] for idx, _, _, _ in retrieval_results]

    DEBUG = True  # 可以添加一个调试标志
    # 调试输出（可选，建议使用日志）
    if DEBUG:
        print(f"    🔍 BM25+SBERT检索结果:")
        for i, c in enumerate(candidates[:10]):
            print(f"    {i+1}. 混合:{c['hybrid_score']:.3f} | "
              f"BM25:{c.get('bm25_score',0):.3f} | "
              f"SBERT:{c.get('sbert_similarity',0):.3f}")
            print(f"{c['title']} | {(c['snippet'] or '')[:200]} | {c['url']}")


    # LLM 选择最终案例
    llm = get_llm(max_tokens=8000)
    n_select = min(FINAL_CASE_COUNT, len(candidates))

    results_text = "\n".join([
        f"{i+1}. [{c.get('hybrid_score', 0):.3f}] {c['title']} | {c['snippet']}"
        for i, c in enumerate(candidates[:20])
    ])
    print(f"    📊 候选案例列表:\n{results_text}")

    select_prompt = load_prompt(
        "plan_execute/sub_2", "02_select_prompt",
        n_select=str(n_select),
        problem_cn=problem_cn,
        results_text=results_text,
    )

    resp = llm.invoke([
        SystemMessage(content="你是案例选择专家"),
        HumanMessage(content=select_prompt),
    ])

    numbers = re.findall(r'\d+', resp.content)
    selected_indices = [int(n) - 1 for n in numbers[:n_select]]

    selected = []
    for idx in selected_indices:
        if 0 <= idx < len(candidates):
            selected.append(candidates[idx])
    print(f"    AI 选择回应: {resp.content.strip()}")

    # 补充不足的
    if len(selected) < n_select:
        used = set(selected_indices)
        for i in range(len(candidates)):
            if i not in used and len(selected) < n_select:
                selected.append(candidates[i])

    print(f"    ✅ 精选 {len(selected)} 个案例")
    return selected


def _exec_gap_analysis(state: PlanExecuteState, task: SubTask) -> str:
    """差异分析"""
    params = task.get("params", {})
    problem = params.get("problem", state["user_query"])

    llm = get_llm(max_tokens=TOKEN_LIMITS.get("gap_analysis_agent", 50000))

    local_context = state.get("local_context", {})
    completed = state.get("completed_results", {})

    # 收集案例信息
    cases_text_parts = []
    for tid, result in completed.items():
        if isinstance(result, list):
            for r in result[:5]:
                if isinstance(r, dict) and r.get("title"):
                    cases_text_parts.append(f"- {r.get('title', '')}: {(r.get('snippet') or '')[:200]}")
        elif isinstance(result, dict) and result.get("final_report"):
            cases_text_parts.append(result["final_report"])

    cases_text = "\n".join(cases_text_parts)

    md_content = read_md_file(PLANNING_KNOWLEDGE_MD)

    prompt = load_prompt(
        "plan_execute/sub_2", "03_gap_analysis_prompt",
        matched_area=local_context.get("matched_area", "未知"),
        analysis_text=local_context.get("analysis_text", "未知"),
        md_content=md_content,
        problem=problem,
        cases_text=cases_text,
    )

    resp = llm.invoke([
        SystemMessage(content="你是国际城市规划专家"),
        HumanMessage(content=prompt),
    ])

    analysis = resp.content.strip()
    print(f"    ✅ 差异分析完成: {len(analysis)} 字符")
    return analysis


def _exec_translate(state: PlanExecuteState, task: SubTask) -> str:
    """翻译任务"""
    params = task.get("params", {})
    text = params.get("text", state["user_query"])

    llm = get_llm(max_tokens=2000)

    prompt = load_prompt(
        "plan_execute/sub_2", "04_translate_prompt",
        text=text,
    )

    resp = llm.invoke([
        SystemMessage(content="你是专业翻译"),
        HumanMessage(content=prompt),
    ])

    result = resp.content.strip()
    print(f"    ✅ 翻译完成: {result[:80]}")
    return result


def _exec_custom_llm(state: PlanExecuteState, task: SubTask) -> str:
    """自定义 LLM 任务"""
    params = task.get("params", {})
    system_prompt = params.get("system_prompt", "你是城市规划专家")
    user_prompt = params.get("user_prompt", task.get("description", ""))

    # 允许在 prompt 中引用已完成任务的结果
    completed = state.get("completed_results", {})
    for tid, result in completed.items():
        placeholder = f"{{{tid}}}"
        if placeholder in user_prompt:
            if isinstance(result, str):
                user_prompt = user_prompt.replace(placeholder, result[:30000])
            else:
                user_prompt = user_prompt.replace(placeholder, json.dumps(result, ensure_ascii=False)[:30000])

    llm = get_llm(max_tokens=TOKEN_LIMITS.get("scenario_agent", 50000))

    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    result = resp.content.strip()
    print(f"    ✅ LLM 任务完成: {len(result)} 字符")
    return result
