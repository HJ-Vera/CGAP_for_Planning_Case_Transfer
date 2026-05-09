"""
智能体 2: 案例查询智能体
- BM25 + Sentence-BERT 混合检索
- 去重、语言感知分词
- 深度信息提取：Gap-Driven Tree Search + 渐进式摘要
"""

import os
import re
import time
import concurrent.futures
import threading
from typing import List, Dict, Union


from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from config import (
    TOKEN_LIMITS, BM25_CANDIDATE_K, FINAL_CASE_COUNT,
    HYBRID_ALPHA, SBERT_MODEL_NAME, OUTPUT_DIR,
)
from llm import get_llm
from tools.search import search_serper, search_academic_sources
from tools.web_fetcher import fetch_webpage_content, fetch_webpage_content_alternative
from tools.retrieval import HybridRetriever, _deduplicate
from tools.deep_research import deep_case_research
from tools.retrieval import BM25Scorer

# ==============================================================
# case_query_agent —— 主函数（修复 + 混合检索版 + deep_case_research整合）
# ==============================================================

def case_query_agent(state, problem_index: int) -> List[Dict]:
    """
    案例查询智能体 v3
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

    llm = get_llm(max_tokens=TOKEN_LIMITS["case_query_agent"])
    chat = get_llm(type="chat", max_tokens=4500)
    user_query = state["user_query"]
    target_city = state["target_city"]
    matched_area = state["local_context"]["matched_area"]

    # ========== 第一步: 翻译问题为英文 ==========
    print("\n🌐 步骤 1: 翻译问题为英文...")

    translation_prompt = f"""
请将以下中文城市规划问题翻译成英文，用于国际案例搜索。
翻译时要保留关键概念，使其适合在国际学术和实践案例中搜索。

**中文问题**: {problem_cn}

请只输出英文翻译，不需要其他解释。
"""
    messages = [
        SystemMessage(content="你是专业翻译专家"),
        HumanMessage(content=translation_prompt)
    ]
    response = chat.invoke(messages)
    problem_en = response.content.strip()
    print(f"✅ 英文问题: {problem_en}")

    # ========== 第二步: 多语言全球搜索 ==========
    print("\n🔍 步骤 2: 全球案例搜索...")

    # 2.1 英文搜索
    print("  🌐 英文搜索 (Serper)...")
    en_queries = [
        f"{problem_en} case study",
        f"{problem_en} international best practices",
    ]
    serper_results = []
    for query in en_queries:
        results = search_serper(query, max_results=50)
        serper_results.extend(results)
        print(f"    找到 {len(results)} 个结果")

    # 2.2 学术搜索
    print("  📚 学术文献搜索...")
    ss_results = search_academic_sources(problem_en, limit=50)

    # 2.3 中文搜索
    print("  🇨🇳 中文案例搜索...")
    cn_results = search_serper(f"{problem_cn} 案例", max_results=30)
    print(f"    找到 {len(cn_results)} 个中文案例")

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

        retriever = HybridRetriever(alpha=HYBRID_ALPHA, sbert_model_name=SBERT_MODEL_NAME)
        retriever.fit(doc_texts, doc_languages)

        queries = {"en": problem_en, "zh": problem_cn}
        retrieval_results = retriever.retrieve(
            queries=queries,
            top_k=BM25_CANDIDATE_K
        )

        for idx, hybrid_score, bm25_norm, sbert_sim in retrieval_results:
            all_results[idx]["hybrid_score"] = hybrid_score
            all_results[idx]["bm25_norm"] = bm25_norm
            all_results[idx]["sbert_sim"] = sbert_sim

        candidates = [all_results[idx] for idx, _, _, _ in retrieval_results]

        print(f"\n  📊 检索分数范围:")
        print(f"      Hybrid: {retrieval_results[-1][1]:.3f} ~ {retrieval_results[0][1]:.3f}")
        print(f"      BM25:   {min(s[2] for s in retrieval_results):.3f} ~ {max(s[2] for s in retrieval_results):.3f}")
        print(f"      SBERT:  {min(s[3] for s in retrieval_results):.3f} ~ {max(s[3] for s in retrieval_results):.3f}")

        results_summary = "\n".join([
            f"{i+1}. [Hybrid:{c['hybrid_score']:.3f} | BM25:{c['bm25_norm']:.3f} | SBERT:{c['sbert_sim']:.3f}] "
            f"[{c['source']}] {c['title']}"
            for i, c in enumerate(candidates)
        ])

        results_input = "\n".join([
            f"{i+1}. [Hybrid:{c['hybrid_score']:.3f} | BM25:{c['bm25_norm']:.3f} | SBERT:{c['sbert_sim']:.3f}] "
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
            all_results[i]["hybrid_score"] = bm25_scores[i]
            all_results[i]["bm25_norm"] = bm25_scores[i]
            all_results[i]["sbert_sim"] = 0.0

        candidates = [all_results[i] for i in top_indices]
        results_summary = "\n".join([
            f"{i+1}. [BM25:{c['bm25_norm']:.3f}] "
            f"[{c['source']}] {c['title']}"
            for i, c in enumerate(candidates)
        ])
    
        results_input = "\n".join([
            f"{i+1}. [BM25:{c['bm25_norm']:.3f}] "
            f"[{c['source']}] {c['title']}| {c['snippet']}"
            for i, c in enumerate(candidates)
        ])
    
        print(f"\n  候选案例列表:\n{results_summary}")
        print(f"  ✅ BM25检索筛选出前 {BM25_CANDIDATE_K} 个候选案例")

    print(f"  ✅ 筛选出 {len(candidates)} 个候选案例")

    # ========== 第四步: LLM 智能选择最终案例 ==========
    selected_cases = []
    used = set()
    try:
        from experiments.exp_flags import USE_HYBRID
    except ImportError:
        USE_HYBRID = True

    if USE_HYBRID:
        print(f"\n🤖 步骤 4: LLM 智能选择最终案例...")

        # ... 原来的 LLM 选择代码保持不动 ...
        selection_prompt = f"""
基于混合检索算法已经筛选出以下 {BM25_CANDIDATE_K} 个候选案例。
分数说明: Hybrid(综合分数) = BM25(关键词匹配)*0.6 + SBERT(语义相似度)*0.4

**中文问题**: {problem_cn}
**英文问题**: {problem_en}
**用户输入**： {user_query}
**目标城市**： {target_city}
**匹配区域**： {matched_area}

**候选案例 (已按混合检索算法预排序)**:
{results_input}

你是一个顶尖城市规划案例专家，分析请从这些案例中选出和问题，用户输入，以及目标城市最相关的 {FINAL_CASE_COUNT} 个案例（需要全球视野，注重地区适配性）。

要求:
1. 参考混合检索分数，但也要考虑案例质量和多元性.不要香港本地的。
2. 优先考虑和目标城市背景相似，具有适配性的案例
3. 优先权威来源的案例（如知名学术期刊、政府报告，国际组织报告、顶尖机构和主流媒体等）
4. 平衡学术研究和实践案例
5. 注意案例的时效性
6. 尽量选外国的案例，不要中国的案例，除非它们非常突出且相关

只需输出 {FINAL_CASE_COUNT} 个序号（对应上面的编号），用逗号分隔:
"""

        messages = [
             SystemMessage(content="你是国际城市规划专家，请综合考虑检索分数和案例质量"),
             HumanMessage(content=selection_prompt)
        ]
        response = llm.invoke(messages)

        numbers = re.findall(r'\d+', response.content)
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
    SELECTED_CASE_COUNT = 2
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

        # ── 5.1 抓取网页内容（逻辑不变）──────────────────────────────
        initial_content = ""
        if case.get("url"):
            try:
                initial_content = fetch_webpage_content(case["url"], max_length=30000)

                if "内容解析失败" in initial_content or len(initial_content) < 100:
                    print(f"    ⚠️ 主方法效果不佳，尝试备用方法...")
                    initial_content = fetch_webpage_content_alternative(case["url"], max_length=30000)

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

            comprehensive_prompt = f"""
作为国际城市规划专家，请对以下案例进行综合分析（用中文输出）:

**案例标题**: {case['title']}

**提取信息**:
{initial_content[:30000]}

请按以下结构分析（不需要JSON，直接markdown格式，缺失信息请标注，不要直接推测，特别注意：直接输出报告内容，不要包含任何解释或前置文本，比如，好的，作为资深城市规划报告撰写专家，我将基于您提供的全部信息，为您生成一份专业、全面的报告。）：
    
【案例来源】
网址链接：{case.get('url', '未知')}
案例标题: {case['title']}
案例来源：（从初步的提取信息中判断，从以下类型中选择，或标注"未知"）
            1. 外国政府官方规划文件（法定图则、政策白皮书、议会报告或者具体部门研究）
            2. 政府委托公开研究报告（标注发布机构）
            3. 学术研究（标注期刊名字）
            4. 专业机构报告（ULI、RICS、ISOCARP 等）
            5. 行业咨询报告（Savills、CBRE 等市场研究）
            6. 新闻报道、项目宣传材料、无法核查来源
    
【基本信息】
城市/国家:
时间:
背景:

【核心问题】
问题描述:

【解决方案】
具体措施:
关键技术/政策工具:

【实施成果】
定量成果:
定性影响:

【前置条件】
制度条件:
经济条件:
技术条件:
社会条件:

【潜在代价/负面影响】
经济代价:
社会影响:
实施风险:
长期挑战:

【可借鉴性评估】
适用情境:
迁移难度:

请尽可能详细，如果某些信息缺失请明确说明。
"""

            
            comprehensive_response = llm.invoke([
                SystemMessage(content="你是国际城市规划案例分析专家"),
                HumanMessage(content=comprehensive_prompt)
                ])

            comprehensive_analysis = comprehensive_response.content.strip()
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
                "hybrid_score": case.get("hybrid_score", 0),
                "bm25_norm": case.get("bm25_norm", 0),
                "sbert_sim": case.get("sbert_sim", 0),
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

        research_result = deep_case_research(
            case=case,
            initial_content=initial_content,
            llm=llm,
            chat=chat,
            search_serper=search_serper,
            fetch_webpage_content=fetch_webpage_content,
            max_loops=3

        )

        extraction     = research_result["extraction"]       # 结构化dict，含7个字段
        final_report   = research_result["final_report"]     # 最终Markdown报告
        loop_count     = research_result["loop_count"]       # 实际搜索轮次
        missing_fields = extraction.get("missing_aspects", [])

        # ── 5.3 保存报告文件（逻辑不变）──────────────────────────────
        try:
            sanitized_title = re.sub(r'[\\/:*?"<>|]', '_', case['title'])
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            report_path = os.path.join(OUTPUT_DIR, f"comprehensive_analysis_{sanitized_title}.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(final_report)
            print(f"    💾 报告已保存: {report_path}")
        except Exception as e:
            print(f"    ⚠️ 报告保存失败: {str(e)[:50]}")

        print(final_report[:500] + "...")

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
            "hybrid_score":         case.get("hybrid_score", 0),
            "bm25_norm":            case.get("bm25_norm", 0),
            "sbert_sim":            case.get("sbert_sim", 0),

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

    summary_prompt = f"""
请对以下全球案例进行总结（中文输出）:

**目标问题**: {problem_cn}

**收集的案例**:
{cases_summary_input}

请撰写一份内容详细且逻辑清晰的总结报告:
1. 全球范围内解决该问题的主要趋势
2. 不同国家/地区的代表性做法和主要案例（分小节罗列案例的详细信息）
3. 共同的成功要素
4. 主要的前置条件
5. 常见的代价和挑战

直接输出中文报告（Markdown 格式）。
"""
    try:
        summary_response = llm.invoke([
            SystemMessage(content="你是国际城市规划研究专家"),
            HumanMessage(content=summary_prompt)
        ])
        global_summary = summary_response.content.strip()
        print(f"✅ 全球案例总结完成")
    except Exception as e:
        print(f"⚠️ 总结生成失败: {e}")
        global_summary = f"针对'{problem_cn}'问题，共收集了{len(structured_cases)}个全球案例。"

    for case in structured_cases:
        case["global_summary"] = global_summary

    # --- 最终统计 ---
    if structured_cases:
        print(f"\n📊 混合检索效果统计:")
        print(f"  {'指标':<12} {'最低':>8} {'平均':>8} {'最高':>8}")
        print(f"  {'-'*40}")
        for field, label in [("hybrid_score", "Hybrid"), ("bm25_norm", "BM25"), ("sbert_sim", "SBERT")]:
            vals = [c.get(field) for c in structured_cases if c.get(field) not in (None, 0)]
            if vals:
                print(f"  {label:<12} {min(vals):>8.3f} {sum(vals)/len(vals):>8.3f} {max(vals):>8.3f}")
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
    print(global_summary[:500] + "...")

    safe_name = re.sub(r'[\\/:*?"<>|\s]', '_', problem_cn)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"global_summary_{safe_name}.md"), "w", encoding="utf-8") as f:
        f.write(global_summary)

    return structured_cases


def case_query_agent_parallel(state, problem_indices: Union[int, List[int]]) -> Dict[int, List[Dict]]:
    """
    并行处理多个问题索引的案例查询 - 修复版
    修复问题：避免LLM API竞争和资源耗尽
    """
    import copy

    if isinstance(problem_indices, int):
        return {problem_indices: case_query_agent(state, problem_indices)}

    print(f"\n{'='*60}")
    print(f"🚀 启动并行案例查询，处理 {len(problem_indices)} 个问题")
    print(f"📌 问题索引: {problem_indices}")

    # 检查必要状态字段
    if 'rewritten_problems' not in state:
        print(f"❌ 错误: state中缺少'rewritten_problems'字段")
        print(f"    State keys: {list(state.keys())}")
        return {idx: [] for idx in problem_indices}

    print(f"📝 可用问题: {len(state['rewritten_problems'])} 个")
    for i, problem in enumerate(state['rewritten_problems']):
        print(f"    {i+1}. {problem[:100]}...")

    # 对每个问题索引创建state副本
    states = {idx: copy.deepcopy(state) for idx in problem_indices}

    results = {}
    # 减少并发数，避免LLM API竞争
    max_workers = min(2, len(problem_indices))  # 从3减少到2

    print(f"📊 并发数: {max_workers} (已降低以避免API竞争)")
    print(f"{'='*60}")

    # 添加全局锁，用于控制LLM调用
    llm_lock = threading.Lock()

    start_time = time.time()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务到线程池
            future_to_index = {}
            for idx in problem_indices:
                # 包装任务函数，添加锁控制，修复闭包变量捕获问题
                def _create_task(task_idx, task_state):
                    def _task():
                        # 使用锁来控制LLM调用
                        with llm_lock:
                            print(f"🔒 线程获取锁，开始处理问题 {task_idx+1}")
                            result = case_query_agent(task_state, task_idx)
                            print(f"🔓 线程释放锁，问题 {task_idx+1} 处理完成")
                            return result
                    return _task

                task_func = _create_task(idx, states[idx])
                future = executor.submit(task_func)
                future_to_index[future] = idx

            # 收集并处理结果
            completed = 0
            all_futures = list(future_to_index.keys())

            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                completed += 1

                try:
                    # 添加超时控制（45分钟，因为现在有锁控制，可能更慢）
                    cases = future.result(timeout=2700)  # 45分钟超时
                    results[idx] = cases
                    print(f"✅ 问题 {idx+1}/{len(problem_indices)} 完成 "
                          f"({completed}/{len(problem_indices)}) - 获取 {len(cases)} 个案例")

                except concurrent.futures.TimeoutError:
                    print(f"⏰ 问题 {idx+1} 超时 (45分钟)")
                    results[idx] = []

                    # 记录超时错误
                    try:
                        error_log_path = os.path.join(OUTPUT_DIR, f"timeout_problem_{idx+1}.txt")
                        with open(error_log_path, "w", encoding="utf-8") as f:
                            f.write(f"问题索引: {idx}\n")
                            f.write(f"中文问题: {state['rewritten_problems'][idx] if idx < len(state.get('rewritten_problems', [])) else '未知'}\n")
                            f.write(f"错误: 操作超时 (45分钟)\n")
                            f.write(f"时间戳: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    except:
                        pass

                except KeyboardInterrupt:
                    print(f"🛑 用户中断，取消所有任务...")
                    # 取消所有未完成的任务
                    for f in all_futures:
                        if not f.done():
                            f.cancel()
                    raise  # 重新抛出中断

                except Exception as e:
                    print(f"❌ 问题 {idx+1} 处理失败: {str(e)}")
                    import traceback
                    print(f"详细错误: {traceback.format_exc()[:500]}")
                    results[idx] = []

                    # 记录详细错误信息
                    try:
                        error_log_path = os.path.join(OUTPUT_DIR, f"error_problem_{idx+1}.txt")
                        with open(error_log_path, "w", encoding="utf-8") as f:
                            f.write(f"问题索引: {idx}\n")
                            f.write(f"中文问题: {state['rewritten_problems'][idx] if idx < len(state.get('rewritten_problems', [])) else '未知'}\n")
                            f.write(f"错误信息: {str(e)}\n")
                            f.write(f"时间戳: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"堆栈跟踪:\n{traceback.format_exc()}\n")
                    except:
                        pass

    except KeyboardInterrupt:
        print(f"🛑 并行查询被用户中断")
        # 确保返回已收集的结果
        elapsed = time.time() - start_time
        print(f"\n📊 部分完成统计:")
        print(f"   运行时间: {elapsed:.2f} 秒")
        print(f"   完成: {completed}/{len(problem_indices)} 个问题")
        print(f"{'='*60}")
        return results

    except Exception as e:
        print(f"💥 并行查询框架错误: {str(e)}")
        import traceback
        traceback.print_exc()
        # 返回已收集的结果或空字典
        return results

    # 检查是否所有任务都已完成
    if len(results) != len(problem_indices):
        print(f"⚠️  警告: 只完成了 {len(results)}/{len(problem_indices)} 个问题")
        # 为未完成的任务添加空结果
        for idx in problem_indices:
            if idx not in results:
                results[idx] = []
                print(f"   问题 {idx+1} 未返回结果，添加空列表")

    elapsed = time.time() - start_time
    print(f"\n📊 并行查询统计:")
    print(f"   总耗时: {elapsed:.2f} 秒")
    print(f"   成功: {sum(1 for cases in results.values() if cases)} 个问题")
    print(f"   失败: {sum(1 for cases in results.values() if not cases)} 个问题")
    print(f"{'='*60}")

    # 检查并行执行结果质量
    successful_count = sum(1 for cases in results.values() if cases)
    if successful_count == 0 and len(problem_indices) > 1:
        print(f"\n⚠️  并行执行未获取到任何案例，尝试顺序执行作为备选方案...")
        sequential_results = {}
        seq_start_time = time.time()

        for idx in problem_indices:
            try:
                print(f"   🔄 顺序处理问题 {idx+1}/{len(problem_indices)}...")
                cases = case_query_agent(state, idx)
                sequential_results[idx] = cases
                print(f"   ✅ 问题 {idx+1} 完成 - 获取 {len(cases)} 个案例")
            except Exception as e:
                print(f"   ❌ 问题 {idx+1} 顺序处理失败: {str(e)[:100]}")
                sequential_results[idx] = []

        seq_elapsed = time.time() - seq_start_time
        print(f"   📊 顺序执行完成，耗时: {seq_elapsed:.2f} 秒")
        return sequential_results

    return results

