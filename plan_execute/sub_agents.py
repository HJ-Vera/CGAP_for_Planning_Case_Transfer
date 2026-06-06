"""
Sub Agents — 子任务执行器

根据 task_type 调度到对应的工具函数执行，
所有工具来自共享的 tools/ 目录。
"""

import asyncio
import json
import re
from typing import Any, Dict, List

import numpy as np

from langchain_core.messages import SystemMessage, HumanMessage

from llm import get_llm
from prompts import load_prompt
from config import (
    TOKEN_LIMITS, HONGKONG_DATA_FILE,
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
        "query_rewrite":              _exec_query_rewrite,
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
        "plan_execute/sub_agents", "01_area_matching_prompt",
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


def _exec_query_rewrite(state: PlanExecuteState, task: SubTask) -> Dict:
    """
    搜索词重写 — 参考 scenario_agent 步骤7 的逻辑。
    基于本地数据分析结果，用 LLM 重写核心问题，
    生成面向「国际案例检索」的优化搜索词。

    依赖: 需要 analyze_local_data 的结果（local_context）
    产出: {"queries_en": [...], "queries_zh": [...], "rewritten_problems": [...]}
    """
    params = task.get("params", {})
    user_query = state["user_query"]
    target_city = state["target_city"]
    local_context = state.get("local_context", {})

    # 从 params 或 local_context 获取分析信息
    analysis_text = local_context.get("analysis_text", "")
    matched_area = local_context.get("matched_area", target_city)
    local_data = local_context.get("local_data", {})

    llm = get_llm(max_tokens=TOKEN_LIMITS.get("scenario_agent", 50000))

    rewrite_prompt = load_prompt(
        "plan_execute/sub_agents", "02_rewrite_prompt",
        user_query=user_query,
        target_city=target_city,
        matched_area=matched_area,
        analysis_text=analysis_text,
        local_data_json=json.dumps(local_data, ensure_ascii=False, default=str)[:3000],
    )

    resp = llm.invoke([
        SystemMessage(content="你是城市规划搜索策略专家，只输出 JSON，不输出其他内容"),
        HumanMessage(content=rewrite_prompt),
    ])

    result = _parse_task_json(resp.content.strip())

    # 确保 fallback
    core_problems = result.get("core_problems", [user_query])
    rewritten = result.get("rewritten_problems", [user_query])
    queries_en = result.get("queries_en", [f"{user_query} urban planning case study"])
    queries_zh = result.get("queries_zh", [f"{user_query} 国际案例"])

    # 把重写的问题和搜索词存入 local_context 供后续任务使用
    ctx = state.get("local_context", {})
    ctx["core_problems"] = core_problems
    ctx["rewritten_problems"] = rewritten
    ctx["search_queries_en"] = queries_en
    ctx["search_queries_zh"] = queries_zh
    state["local_context"] = ctx

    print(f"    🎯 核心问题: {core_problems}")
    print(f"    📝 重写关键词: {rewritten}")
    print(f"    🔍 英文搜索词 ({len(queries_en)}): {queries_en}")
    print(f"    🔍 中文搜索词 ({len(queries_zh)}): {queries_zh}")

    return {
        "core_problems": core_problems,
        "rewritten_problems": rewritten,
        "queries_en": queries_en,
        "queries_zh": queries_zh,
    }


def _parse_task_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON（子任务用）"""
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


def _exec_search_web_cases(state: PlanExecuteState, task: SubTask) -> List[Dict]:
    """搜索网页案例 — 支持从 query_rewrite 结果中自动获取优化搜索词"""
    params = task.get("params", {})
    max_results = params.get("max_results", 20)
    use_rewrite = params.get("queries_from_rewrite", False)

    queries = params.get("queries", None)

    # 如果指定使用 query_rewrite 的结果，从依赖任务中自动提取搜索词
    if use_rewrite and not queries:
        completed = state.get("completed_results", {})
        for dep_id in task.get("dependencies", []):
            dep_result = completed.get(dep_id)
            if isinstance(dep_result, dict) and dep_result.get("queries_en"):
                queries = dep_result["queries_en"] + dep_result.get("queries_zh", [])
                print(f"    📝 从 {dep_id} 的 query_rewrite 结果中获取搜索词")
                break
        # 兜底: 从 state 的 local_context 中获取
        if not queries:
            ctx = state.get("local_context", {})
            queries = ctx.get("search_queries_en", [])
            if queries:
                print(f"    📝 从 local_context 中获取重写搜索词")

    if not queries:
        queries = [state["user_query"] + " international case study urban planning"]

    print(f"    🔍 共 {len(queries)} 条搜索词:")
    for q in queries:
        print(f"       → {q[:80]}{'...' if len(q) > 80 else ''}")

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

    results = search_academic_sources(
        query,
        limit=limit,
        skip_semantic_scholar=SKIP_SEMANTIC_SCHOLAR
    )
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
        # 收集所有依赖任务产出的案例列表（优先）
        dep_cases = []
        for dep_id in task.get("dependencies", []):
            dep_result = completed.get(dep_id)
            if isinstance(dep_result, list):
                for item in dep_result:
                    if isinstance(item, dict) and item.get("title"):
                        dep_cases.append(item)

        # 扫描所有已完成任务中的案例列表（包括 _exec_hybrid_retrieval 返回的结果）
        all_cases = []
        for tid, result in completed.items():
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict) and item.get("title"):
                        all_cases.append(item)

        # 合并：依赖任务案例优先，然后其他已完成任务案例（去重）
        candidate_cases = []
        seen_keys = set()

        # 首先添加依赖任务案例
        for case in dep_cases:
            case_title = case.get("title", "")
            case_url = case.get("url", case.get("link", ""))
            key = (case_title, case_url)
            if case_title and key not in seen_keys:
                seen_keys.add(key)
                candidate_cases.append(case)

        # 然后添加其他已完成任务案例（排除已添加的）
        for case in all_cases:
            case_title = case.get("title", "")
            case_url = case.get("url", case.get("link", ""))
            key = (case_title, case_url)
            if case_title and key not in seen_keys:
                seen_keys.add(key)
                candidate_cases.append(case)

        print(f"    📊 收集到 {len(candidate_cases)} 个候选案例（依赖任务 {len(dep_cases)} 个，所有任务 {len(all_cases)} 个）")
        # 如果候选案例超过20个，取前20个（形成完整的候选结果）
        if len(candidate_cases) > 20:
            candidate_cases = candidate_cases[:20]
            print(f"    📊 候选案例超过20个，取前20个")

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

        # 确定起始索引
        start_idx = 0
        if case_index is not None:
            start_idx = case_index if case_index < len(pool) else 0

        # 循环尝试 pool 中的案例，直到找到内容充足的案例
        for i in range(start_idx, len(pool)):
            selected = pool[i]
            title = selected.get("title", "未知案例")
            url = selected.get("url", selected.get("link", ""))
            snippet = selected.get("snippet", selected.get("abstract", ""))
            print(f"    📌 尝试案例 {i+1}/{len(pool)}: {title}")

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

            # 检查内容是否充足
            if initial_content and len(initial_content) >= 2000:
                # 内容充足，跳出循环进行深度研究
                break
            else:
                # 内容不足，尝试下一个案例
                print(f"    ⚠️ 案例内容不足 ({len(initial_content) if initial_content else 0} 字符)，尝试下一个案例")
                # 如果已经是最后一个案例，使用当前信息进行LLM基础研究
                if i == len(pool) - 1:
                    print(f"    ⚠️ 所有案例内容均不足，使用 LLM 基于标题进行基础研究")
                    resp = llm.invoke([
                        SystemMessage(content="你是城市规划案例研究专家"),
                        HumanMessage(content=load_prompt(
                            "plan_execute/sub_agents", "06_fallback_prompt",
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
                        "note": "基于 LLM 知识的基础研究（所有案例网页内容不足）",
                    }
                # 否则继续循环尝试下一个案例
                continue

        # 如果循环结束（通过break跳出），表示找到了内容充足的案例，继续执行深度研究
        print(f"    ✅ 使用案例 {i+1}/{len(pool)}: {title}，内容充足 ({len(initial_content)} 字符)")
    else:
        # params 已提供案例信息，直接使用
        print(f"    📌 使用 params 指定的案例: {title}")
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
                        "plan_execute/sub_agents", "06_fallback_prompt",
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
    result = asyncio.run(deep_case_research(
        case=case,
        initial_content=initial_content,
        llm=llm,
        chat=chat,
        search_serper=search_serper,
        fetch_webpage_content=fetch_webpage_content,
        max_loops=3,
    ))

    return {
        "title": title,
        "extraction": result.get("extraction", {}),
        "final_report": result.get("final_report", ""),
        "loop_count": result.get("loop_count", 0),
    }

def _exec_hybrid_retrieval(state: PlanExecuteState, task: SubTask) -> List[Dict]:
    """RRF (Reciprocal Rank Fusion) 混合检索排序 + LLM 选择"""
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

    n_docs = len(all_results)
    doc_texts = [f"{r.get('title', '')} {r.get('snippet', '')}" for r in all_results]
    doc_languages = [r.get("language", "en") for r in all_results]

    # ── 初始化检索器，训练 BM25 + 编码 SBERT ──
    retriever = HybridRetriever(alpha=HYBRID_ALPHA, sbert_model_name=SBERT_MODEL_NAME)
    retriever.fit(doc_texts, doc_languages)

    queries = {"en": problem_en or problem_cn, "zh": problem_cn}

    # ── BM25 排名 ──
    bm25_raw = np.array(retriever.bm25.score_all(queries))
    bm25_ranking = np.argsort(-bm25_raw)  # 按分数降序排列的 doc index 列表

    # ── SBERT 相似度排名 ──
    query_vecs = {}
    for lang, q_text in queries.items():
        query_vecs[lang] = retriever.sbert.encode([q_text])[0]

    sbert_sims = np.zeros(n_docs)
    for i in range(n_docs):
        doc_lang = doc_languages[i]
        q_vec = query_vecs.get(doc_lang, query_vecs.get("en"))
        if q_vec is None:
            continue
        sbert_sims[i] = retriever.sbert.cosine_similarity(
            q_vec, retriever._sbert_doc_vecs[i:i+1]
        )[0]
    sbert_ranking = np.argsort(-sbert_sims)

    # ── RRF 融合: score(d) = Σ 1/(k + rank_i(d)) ──
    RRF_K = 60  # 标准 RRF 参数
    bm25_rank_map = {int(idx): rank + 1 for rank, idx in enumerate(bm25_ranking)}  # doc_idx → rank (1-based)
    sbert_rank_map = {int(idx): rank + 1 for rank, idx in enumerate(sbert_ranking)}

    rrf_scores = np.zeros(n_docs)
    for i in range(n_docs):
        rrf_scores[i] = 1.0 / (RRF_K + bm25_rank_map[i]) + 1.0 / (RRF_K + sbert_rank_map[i])

    top_k = min(BM25_CANDIDATE_K, n_docs)
    top_indices = np.argsort(-rrf_scores)[:top_k]

    # 标记分数到 all_results
    for idx in top_indices:
        i = int(idx)
        all_results[i]["rrf_score"] = float(rrf_scores[i])
        all_results[i]["bm25_score"] = float(bm25_raw[i])
        all_results[i]["bm25_rank"] = bm25_rank_map[i]
        all_results[i]["sbert_similarity"] = float(sbert_sims[i])
        all_results[i]["sbert_rank"] = sbert_rank_map[i]

    candidates = [all_results[int(idx)] for idx in top_indices]

    # 调试输出
    print(f"    🔍 RRF 融合检索结果 (k={RRF_K}):")
    for i, c in enumerate(candidates[:10]):
        print(f"    {i+1}. RRF:{c['rrf_score']:.5f} | "
              f"BM25 rank:{c.get('bm25_rank',0)} ({c.get('bm25_score',0):.2f}) | "
              f"SBERT rank:{c.get('sbert_rank',0)} ({c.get('sbert_similarity',0):.3f})")
        print(f"       {c['title'][:80]} | {c['url']}")

    # ── LLM 选择最终案例 ──
    llm = get_llm(max_tokens=8000)
    n_select = min(FINAL_CASE_COUNT, len(candidates))

    results_text = "\n".join([
        f"{i+1}. [RRF:{c.get('rrf_score', 0):.5f}] {c['title']} | {c['snippet']}"
        for i, c in enumerate(candidates[:20])
    ])
    print(f"    📊 候选案例列表:\n{results_text}")

    select_prompt = load_prompt(
        "plan_execute/sub_agents", "03_select_prompt",
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
        "plan_execute/sub_agents", "04_gap_analysis_prompt",
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
        "plan_execute/sub_agents", "05_translate_prompt",
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
                user_prompt = user_prompt.replace(placeholder, result[:3000])
            else:
                user_prompt = user_prompt.replace(placeholder, json.dumps(result, ensure_ascii=False)[:3000])

    llm = get_llm(max_tokens=TOKEN_LIMITS.get("scenario_agent", 50000))

    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    result = resp.content.strip()
    print(f"    ✅ LLM 任务完成: {len(result)} 字符")
    return result
