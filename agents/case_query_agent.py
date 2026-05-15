"""
智能体 2: 案例查询智能体 (异步版)
- BM25 + Sentence-BERT 混合检索
- 去重、语言感知分词
- 深度信息提取：Gap-Driven Tree Search + 渐进式摘要
"""

import asyncio
import os
import re
import time
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from config import (
    TOKEN_LIMITS, BM25_CANDIDATE_K, FINAL_CASE_COUNT,
    SBERT_MODEL_NAME, OUTPUT_DIR,
    SKIP_SEMANTIC_SCHOLAR,
)
from llm import get_llm
from prompts import load_prompt
from tools.retrieval import HybridRetriever, _deduplicate
from tools.retrieval import BM25Scorer
from tools.deep_research import deep_case_research
from services.search_service import SearchService
from services.fetch_service import async_fetch_webpage_content, async_fetch_webpage_content_alternative
from services.llm_service import LLMService

SELECTED_CASE_COUNT = 2

def extract_content(response):
    """Extract text from LangChain AIMessage response, handling both string and list content."""
    content = response.content
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if hasattr(block, 'text'):
                parts.append(block.text)
            elif isinstance(block, dict) and 'text' in block:
                parts.append(block['text'])
            else:
                parts.append(str(block))
        return ' '.join(parts)
    else:
        return str(content)

# ==============================================================
# case_query_agent —— 主函数（修复 + 混合检索版 + deep_case_research整合）
# ==============================================================

async def case_query_agent(state, problem_index: int) -> List[Dict]:
    """
    案例查询智能体 v3 (异步版)
      - BM25 bug 修复
      - 集成 Sentence-BERT 混合检索
      - 去重、语言感知分词、短内容兜底
      - 深度信息提取：Gap-Driven Tree Search + 渐进式摘要
    """
    print(f"\n{'=' * 60}")
    print(f"🌍 案例查询智能体 #{problem_index + 1} 启动 (混合检索模式)")
    print(f"{'=' * 60}")

    problem_cn = state["rewritten_problems"][problem_index]
    print(f"📌 中文问题: {problem_cn}")

    user_query = state["user_query"]
    target_city = state["target_city"]
    matched_area = state["local_context"]["matched_area"]

    # ========== 第一步: 翻译问题为英文 ==========
    print("\n🌐 步骤 1: 翻译问题为英文...")

    translation_prompt = load_prompt(
        "agents/case_query_agent", "01_translation_prompt",
        problem_cn=problem_cn,
    )
    messages = [
        SystemMessage(content="你是专业翻译专家"),
        HumanMessage(content=translation_prompt)
    ]
    problem_en = await LLMService.ainvoke("chat", max_tokens=4500, messages=messages)
    problem_en = problem_en.strip()
    print(f"✅ 英文问题: {problem_en}")

    # ========== 第二步: 多语言全球搜索（并发） ==========
    print("\n🔍 步骤 2: 全球案例搜索...")

    en_queries = [
        f"{problem_en} case study",
        f"{problem_en} international best practices",
    ]

    # 所有搜索并发
    search_tasks = [
        SearchService.search_serper(en_queries[0], max_results=50),
        SearchService.search_serper(en_queries[1], max_results=50),
        SearchService.search_academic_sources(problem_en, limit=50, skip_semantic_scholar=SKIP_SEMANTIC_SCHOLAR),
        SearchService.search_serper(f"{problem_cn} 案例", max_results=50),
    ]
    en_result1, en_result2, ss_results, cn_results = await asyncio.gather(*search_tasks)

    serper_results = []
    serper_results.extend(en_result1 or [])
    print(f"    找到 {len(en_result1 or [])} 个结果 (英文查询1)")
    serper_results.extend(en_result2 or [])
    print(f"    找到 {len(en_result2 or [])} 个结果 (英文查询2)")
    print(f"    找到 {len(cn_results or [])} 个中文案例")

    # --- 合并 + 去重 ---
    all_results = []

    for r in serper_results:
        all_results.append({
            "source": "Serper_EN",
            "title": r["title"],
            "url": r["url"],
            "snippet": r["snippet"],
            "language": "en"
        })

    for r in ss_results:
        all_results.append({
            "source": r.get("source", "Semantic_Scholar"),
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": (r.get("abstract") or "")[:200],
            "year": r.get("year", "N/A"),
            "citations": r.get("citationCount", 0),
            "venue": r.get("venue", ""),
            "language": "en"
        })

    for r in cn_results:
        all_results.append({
            "source": "Serper_CN",
            "title": r["title"],
            "url": r["url"],
            "snippet": r["snippet"],
            "language": "zh"
        })

    # 去重
    all_results = _deduplicate(all_results)

    en_count = sum(1 for r in all_results if r.get("language") == "en")
    zh_count = sum(1 for r in all_results if r.get("language") == "zh")
    print(f"\n✅ 去重后共 {len(all_results)} 个结果 (英文: {en_count}, 中文: {zh_count})")

    # ========== 第三步: 混合检索筛选 ==========
    try:
        from experiments.exp_flags import USE_HYBRID
    except ImportError:
        USE_HYBRID = True

    if USE_HYBRID:
        print(f"\n🎯 步骤 3: BM25 + Sentence-BERT 混合检索筛选...")

        doc_texts = [f"{r['title']} {r['snippet']}" for r in all_results]
        doc_languages = [r.get("language", "en") for r in all_results]

        retriever = HybridRetriever(sbert_model_name=SBERT_MODEL_NAME)
        retriever.fit(doc_texts, doc_languages)

        queries = {"en": problem_en, "zh": problem_cn}
        retrieval_results = retriever.retrieve(
            queries=queries,
            top_k=BM25_CANDIDATE_K
        )

        for idx, rrf_score, bm25_rank, sbert_rank in retrieval_results:
            all_results[idx]["rrf_score"] = rrf_score
            all_results[idx]["bm25_rank"] = bm25_rank
            all_results[idx]["sbert_rank"] = sbert_rank

        candidates = [all_results[idx] for idx, _, _, _ in retrieval_results]

        print(f"\n  📊 RRF 融合检索分数范围:")
        print(f"      RRF:  {retrieval_results[-1][1]:.5f} ~ {retrieval_results[0][1]:.5f}")
        print(f"      BM25 rank:  {min(s[2] for s in retrieval_results)} ~ {max(s[2] for s in retrieval_results)}")
        print(f"      SBERT rank: {min(s[3] for s in retrieval_results)} ~ {max(s[3] for s in retrieval_results)}")

        results_summary = "\n".join([
            f"{i+1}. [RRF:{c['rrf_score']:.5f} | BM25 rank:{c['bm25_rank']} | SBERT rank:{c['sbert_rank']}] "
            f"[{c['source']}] {c['title']}"
            for i, c in enumerate(candidates)
        ])

        results_input = "\n".join([
            f"{i+1}. [RRF:{c['rrf_score']:.5f} | BM25 rank:{c['bm25_rank']} | SBERT rank:{c['sbert_rank']}] "
            f"[{c['source']}] {c['title']}| {c['snippet']}"
            for i, c in enumerate(candidates)
        ])

        print(f"\n  候选案例列表:\n{results_summary}")
        print(f"  ✅ 混合检索筛选出前 {BM25_CANDIDATE_K} 个候选案例")

    else:
        print(f"\n🎯 步骤 3: 仅 BM25 检索筛选（消融模式）...")

        doc_texts = [f"{r['title']} {r['snippet']}" for r in all_results]
        doc_languages = [r.get("language", "en") for r in all_results]

        
        scorer = BM25Scorer()
        scorer.fit(doc_texts, doc_languages)
        queries_bm25 = {"en": problem_en, "zh": problem_cn}
        bm25_scores = scorer.score_all(queries_bm25)

        top_indices = sorted(range(len(bm25_scores)),
                             key=lambda i: bm25_scores[i], reverse=True)[:BM25_CANDIDATE_K]

        for i in top_indices:
            all_results[i]["rrf_score"] = bm25_scores[i]
            all_results[i]["bm25_rank"] = 0
            all_results[i]["sbert_rank"] = 0

        candidates = [all_results[i] for i in top_indices]
        results_summary = "\n".join([
            f"{i+1}. [BM25:{c['rrf_score']:.3f}] "
            f"[{c['source']}] {c['title']}"
            for i, c in enumerate(candidates)
        ])

        results_input = "\n".join([
            f"{i+1}. [BM25:{c['rrf_score']:.3f}] "
            f"[{c['source']}] {c['title']}| {c['snippet']}"
            for i, c in enumerate(candidates)
        ])
    
        print(f"\n  候选案例列表:\n{results_summary}")
        print(f"  ✅ BM25检索筛选出前 {BM25_CANDIDATE_K} 个候选案例")

    print(f"  ✅ 筛选出 {len(candidates)} 个候选案例")

    # ========== 第四步: LLM 智能选择最终案例 ==========
    selected_cases = []
    used = set()

    if USE_HYBRID:
        print(f"\n🤖 步骤 4: LLM 智能选择最终案例...")

        # ... 原来的 LLM 选择代码保持不动 ...
        selection_prompt = load_prompt(
            "agents/case_query_agent", "02_selection_prompt",
            BM25_CANDIDATE_K=str(BM25_CANDIDATE_K),
            problem_cn=problem_cn,
            problem_en=problem_en,
            user_query=user_query,
            target_city=target_city,
            matched_area=matched_area,
            results_input=results_input,
            FINAL_CASE_COUNT=str(FINAL_CASE_COUNT),
        )

        messages = [
             SystemMessage(content="你是国际城市规划专家，请综合考虑检索分数和案例质量"),
             HumanMessage(content=selection_prompt)
        ]
        response = await LLMService.ainvoke("default", max_tokens=TOKEN_LIMITS["case_query_agent"], messages=messages)

        numbers = re.findall(r'\d+', response)
        top_indices = [int(n) - 1 for n in numbers[:FINAL_CASE_COUNT]]
        if not top_indices:
             print("  ⚠️ LLM 未输出有效选择，使用混合检索前 N 名")
             top_indices = list(range(min(FINAL_CASE_COUNT, len(candidates))))

        for idx in top_indices:
            if 0 <= idx < len(candidates) and idx not in used:
                selected_cases.append(candidates[idx])
                used.add(idx)
        if len(selected_cases) < FINAL_CASE_COUNT:
            print(f"  ⚠️ LLM 只选了 {len(selected_cases)} 个，用检索排名补充")
            for i in range(len(candidates)):
                if i not in used and len(selected_cases) < FINAL_CASE_COUNT:
                    selected_cases.append(candidates[i])
                    used.add(i)
    
    else:
        print(f"\n🤖 步骤 4: 直接使用检索排名前 {FINAL_CASE_COUNT} 个（消融模式）")
        selected_cases = candidates[:FINAL_CASE_COUNT]


    print(f"✅ 最终选出 {len(selected_cases)} 个案例")


    # ========== 第五步: 深度信息提取与补充（Gap-Driven Tree Search） ==========
    print("\n📊 步骤 5: 深度信息提取与补充（Gap-Driven Tree Search）...")

    # 构建合并候选列表（selected_cases优先，去重）
    merged_candidates = []
    seen_urls = set()

    # 首先添加selected_cases
    for case in selected_cases:
        url = case.get("url", "")
        if url and url not in seen_urls:
            merged_candidates.append(case)
            seen_urls.add(url)
        elif not url:  # 如果没有URL，使用标题作为去重依据
            title = case.get("title", "")
            if title and title not in seen_urls:
                merged_candidates.append(case)
                seen_urls.add(title)

    # 然后添加candidates中未重复的案例
    for case in candidates:
        url = case.get("url", "")
        if url and url not in seen_urls:
            merged_candidates.append(case)
            seen_urls.add(url)
        elif not url:  # 如果没有URL，使用标题作为去重依据
            title = case.get("title", "")
            if title and title not in seen_urls:
                merged_candidates.append(case)
                seen_urls.add(title)

    print(f"✅ 合并候选列表: {len(merged_candidates)} 个案例（去重后）")

    structured_cases = []
    processed_urls_titles = set()
    candidate_index = 0

    # 循环处理，直到成功处理FINAL_CASE_COUNT个案例或候选列表耗尽
    # 注意：内容不足（<2000字符）的案例会被跳过，不计入成功处理的案例
    while len(structured_cases) < SELECTED_CASE_COUNT and candidate_index < len(merged_candidates):
        case = merged_candidates[candidate_index]
        candidate_index += 1

        # 检查是否已处理（基于URL或标题）
        url = case.get("url", "")
        title = case.get("title", "")
        identifier = url if url else title

        if not identifier or identifier in processed_urls_titles:
            continue

        print(f"\n  📄 处理案例 {len(structured_cases)+1}/{SELECTED_CASE_COUNT}: {case['title']}")

        # ── 5.1 抓取网页内容（异步）──────────────────────────────
        initial_content = ""
        if case.get("url"):
            try:
                initial_content = await async_fetch_webpage_content(case["url"], max_length=30000)

                if "内容解析失败" in initial_content or len(initial_content) < 100:
                    print(f"    ⚠️ 主方法效果不佳，尝试备用方法...")
                    initial_content = await async_fetch_webpage_content_alternative(case["url"], max_length=30000)

                if initial_content and not initial_content.startswith("网页访问"):
                    print(f"    ✅ 抓取网页内容: {len(initial_content)} 字符")
                else:
                    print(f"    ⚠️ {initial_content}")
                    initial_content = case.get("snippet", "")
            except Exception as e:
                print(f"    ⚠️ 网页抓取异常: {str(e)[:50]}")
                initial_content = case.get("snippet", "")
        else:
            initial_content = case.get("snippet", "")
            print(f"    ℹ️ 使用摘要信息: {len(initial_content)} 字符")

        # ── 5.2 初步判断：内容不足则跳过深度研究 ─────────────────────
        # 如果内容不足2000字符，跳过此案例，继续处理下一个（不计入总数）
        if not initial_content or len(initial_content) < 2000:
            print(f"    ✗ 初始内容不足2000字符（当前 {len(initial_content)} 字符），跳过该案例")
            processed_urls_titles.add(identifier)
            continue

        # 内容足够，分配案例编号
        case_num = len(structured_cases) + 1

        # ── 5.3 Gap-Driven深度研究（替换原提取+补全逻辑）─────────────
        try:
            from experiments.exp_flags import USE_DEEP_RESEARCH
        except ImportError:
            USE_DEEP_RESEARCH = True

        if not USE_DEEP_RESEARCH:
            print(f"    ⏭️ 跳过深度研究（消融模式），只用初步抓取内容")
            print(f"    🧠 综合分析...")

            comprehensive_prompt = load_prompt(
                "agents/case_query_agent", "03_comprehensive_prompt",
                case_title=case['title'],
                initial_content=initial_content[:30000],
                case_url=str(case.get('url', '未知')),
            )

            
            comprehensive_response = await LLMService.ainvoke_raw("default", max_tokens=TOKEN_LIMITS["case_query_agent"], messages=[
                SystemMessage(content="你是国际城市规划案例分析专家"),
                HumanMessage(content=comprehensive_prompt)
                ])

            comprehensive_analysis = extract_content(comprehensive_response).strip() if comprehensive_response else ""
            print(f"    ✅ 综合分析完成: {len(comprehensive_analysis)} 字符")
            # print(f"\n💡 案例综合分析: {comprehensive_analysis}")

            structured_case = {
                "case_number": case_num,
                "title": case["title"],
                "url": case.get("url", "N/A"),
                "source": case.get("source", "Unknown"),
                "language": case.get("language", "unknown"),
                "has_supplement": False,
                "loop_count": 0,
                "hybrid_score": case.get("rrf_score", 0),
                "bm25_norm": case.get("bm25_rank", 0),
                "sbert_sim": case.get("sbert_rank", 0),
                "city_country": "",
                "time": "",
                "core_problem": "",
                "solution": "",
                "key_results": "",
                "preconditions": "",
                "downsides": "",
                "missing_fields": ["city_country", "time", "core_problem", "solution", "key_results", "preconditions", "downsides"],
                "initial_extraction": initial_content[:30000],
                "supplementary_info": "",
                "comprehensive_analysis": comprehensive_analysis,
            }
            structured_cases.append(structured_case)
            processed_urls_titles.add(identifier)
            continue

        # ── 原来的 deep_case_research 调用代码继续 ──

        research_result = await deep_case_research(
            case=case,
            initial_content=initial_content,
            llm=get_llm(max_tokens=TOKEN_LIMITS["case_query_agent"]),
            chat=get_llm(type="chat", max_tokens=4500),
            max_loops=3
        )

        extraction     = research_result["extraction"]       # 结构化dict，含7个字段
        final_report   = research_result["final_report"]     # 最终Markdown报告
        loop_count     = research_result["loop_count"]       # 实际搜索轮次
        missing_fields = extraction.get("missing_aspects", [])

        # ── 5.3 保存报告文件（逻辑不变）──────────────────────────────
        try:
            sanitized_title = re.sub(r'[\\/:*?"<>|]', '_', str(case['title']))
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            report_path = os.path.join(OUTPUT_DIR, f"comprehensive_analysis_{sanitized_title}.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(final_report)
            print(f"    💾 报告已保存: {report_path}")
        except Exception as e:
            print(f"    ⚠️ 报告保存失败: {str(e)[:50]}")

        # print(final_report[:500] + "...")

        # ── 5.4 构造structured_case（字段对齐新函数输出）─────────────
        structured_case = {
            # 原有元信息字段（保持不变，下游兼容）
            "case_number":          case_num,
            "title":                case["title"],
            "url":                  case.get("url", "N/A"),
            "source":               case.get("source", "Unknown"),
            "language":             case.get("language", "unknown"),
            "has_supplement":       loop_count > 0,
            "loop_count":           loop_count,
            "hybrid_score":         case.get("rrf_score", 0),
            "bm25_norm":            case.get("bm25_rank", 0),
            "sbert_sim":            case.get("sbert_rank", 0),

            # 新结构化提取字段（替换原 initial_extraction + supplementary_info）
            "city_country":         extraction.get("city_country", ""),
            "time":                 extraction.get("time", ""),
            "core_problem":         extraction.get("core_problem", ""),
            "solution":             extraction.get("solution", ""),
            "key_results":          extraction.get("key_results", ""),
            "preconditions":        extraction.get("preconditions", ""),
            "downsides":            extraction.get("downsides", ""),
            "missing_fields":       missing_fields,

            # 兼容下游仍读取这两个key的地方
            "initial_extraction":   final_report,
            "supplementary_info":   "已整合入final_report（Gap-Driven Tree Search）",

            # 最终报告
            "comprehensive_analysis": final_report,
        }

        structured_cases.append(structured_case)
        processed_urls_titles.add(identifier)
        print(f"    ✅ 案例处理完成，字段完整度: {7 - len(missing_fields)}/7，搜索轮次: {loop_count}")

    # 检查是否达到目标案例数量
    if len(structured_cases) < SELECTED_CASE_COUNT:
        print(f"⚠️  警告: 只成功处理了 {len(structured_cases)}/{SELECTED_CASE_COUNT} 个案例（内容足够的案例数量不足）")

    # ========== 第六步: 生成中文总结报告 ==========
    print(f"\n📝 步骤 6: 生成中文总结报告...")

    # 用结构化extraction构造摘要输入，比直接拼标题信息量更大
    cases_summary_input = "\n\n".join([
        f"案例{c['case_number']}: {c['title']}\n"
        f"  城市/国家: {c['city_country']}\n"
        f"  核心问题: {c['core_problem']}\n"
        f"  解决方案: {c['solution']}\n"
        f"  关键成果: {c['key_results']}\n"
        f"  前置条件: {c['preconditions']}\n"
        f"  潜在代价: {c['downsides']}"
        for c in structured_cases
    ])

    summary_prompt = load_prompt(
        "agents/case_query_agent", "04_summary_prompt",
        problem_cn=problem_cn,
        cases_summary_input=cases_summary_input,
    )
    try:
        summary_response = await LLMService.ainvoke("default", max_tokens=TOKEN_LIMITS["case_query_agent"], messages=[
            SystemMessage(content="你是国际城市规划研究专家"),
            HumanMessage(content=summary_prompt)
        ])
        global_summary = summary_response.strip()
        print(f"✅ 全球案例总结完成")
    except Exception as e:
        print(f"⚠️ 总结生成失败: {e}")
        global_summary = f"针对'{problem_cn}'问题，共收集了{len(structured_cases)}个全球案例。"

    for case in structured_cases:
        case["global_summary"] = global_summary

    # --- 最终统计 ---
    if structured_cases:
        print(f"\n📊 RRF 检索效果统计:")
        print(f"  {'指标':<12} {'最低':>8} {'平均':>8} {'最高':>8}")
        print(f"  {'-'*40}")
        for field, label in [("hybrid_score", "RRF"), ("bm25_norm", "BM25 rank"), ("sbert_sim", "SBERT rank")]:
            vals = [c.get(field) for c in structured_cases if c.get(field) not in (None, 0)]
            if vals:
                fmt = ".5f" if field == "hybrid_score" else ".1f"
                print(f"  {label:<12} {min(vals):>{8}{fmt}} {sum(vals)/len(vals):>{8}{fmt}} {max(vals):>{8}{fmt}}")
            else:
                print(f"  {label:<12} {'N/A':>8} {'N/A':>8} {'N/A':>8}")

        # 新增：深度研究统计
        print(f"\n📊 深度研究统计:")
        avg_loops = sum(c.get("loop_count", 0) for c in structured_cases) / len(structured_cases)
        avg_complete = sum(7 - len(c.get("missing_fields", [])) for c in structured_cases) / len(structured_cases)
        print(f"  平均搜索轮次: {avg_loops:.1f} / 3")
        print(f"  平均字段完整度: {avg_complete:.1f} / 7")

    print(f"\n{'=' * 60}")
    print(f"✅ 案例查询完成!")
    print(f"{'=' * 60}")
    print(f"📊 收集案例: {len(structured_cases)} 个")
    print(f"🌍 英文案例: {sum(1 for c in structured_cases if c.get('language') == 'en')} 个")
    print(f"🇨🇳 中文案例: {sum(1 for c in structured_cases if c.get('language') == 'zh')} 个")
    print(f"🔍 补充搜索: {sum(1 for c in structured_cases if c.get('has_supplement'))} 个")
    print(f"📊 global_summary 长度: {len(global_summary)} 字符")

    safe_name = re.sub(r'[\\/:*?"<>|\s]', '_', str(problem_cn))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"global_summary_{safe_name}.md"), "w", encoding="utf-8") as f:
        f.write(global_summary)

    return structured_cases