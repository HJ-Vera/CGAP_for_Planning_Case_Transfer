"""
深度案例研究模块 — Gap-Driven Tree Search + 渐进式摘要 (异步版)
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable


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


EMPTY_MARKERS = {"", "未知", "不详", "无", "N/A", "暂无", "需要补充"}

# -------- 改进1: 按字段设定不同的最低长度阈值 --------
FIELD_MIN_LENGTHS = {
    "city_country": 2,    # "深圳" = 2字符，应视为有效
    "time": 2,            # "2019" = 4字符，"3月" = 2字符
    "core_problem": 8,
    "solution": 8,
    "key_results": 6,
    "preconditions": 6,
    "downsides": 6,
}

# 标准化字段名映射: 中文名 → 英文key
CN_TO_EN = {
    "城市/国家":         "city_country",
    "城市":             "city_country",
    "时间":             "time",
    "核心问题":         "core_problem",
    "解决方案":         "solution",
    "关键数据/成果":     "key_results",
    "关键数据":         "key_results",
    "成果":             "key_results",
    "前置条件":         "preconditions",
    "潜在代价/负面影响": "downsides",
    "潜在代价":         "downsides",
    "负面影响":         "downsides",
}

EN_TO_CN = {
    "city_country":   "城市/国家",
    "time":           "时间",
    "core_problem":   "核心问题",
    "solution":       "解决方案",
    "key_results":    "关键数据/成果",
    "preconditions":  "前置条件",
    "downsides":      "潜在代价/负面影响",
}

SEVEN_FIELDS = list(EN_TO_CN.keys())


def _is_empty(val: str, field_key: str = "") -> bool:
    """改进: 按字段名使用不同的最低长度阈值"""
    if not val or val.strip() in EMPTY_MARKERS:
        return True
    min_len = FIELD_MIN_LENGTHS.get(field_key, 6)
    return len(val.strip()) < min_len


def _normalize_field_name(name: str) -> str:
    """
    改进2: 将LLM返回的中文字段名标准化为英文key。
    解决LLM返回"潜在代价"而非"潜在代价/负面影响"导致的匹配失败。
    """
    name = name.strip()
    # 直接匹配
    if name in CN_TO_EN:
        return CN_TO_EN[name]
    # 模糊匹配: 检查是否是某个中文key的子串或包含关系
    for cn_key, en_key in CN_TO_EN.items():
        if name in cn_key or cn_key in name:
            return en_key
    # 如果本身就是英文key
    if name in EN_TO_CN:
        return name
    return name  # 无法匹配则原样返回


def _get_missing_fields(ext: dict) -> list:
    """程序校验7个字段，返回英文key列表（标准化）"""
    return [k for k in SEVEN_FIELDS if _is_empty(ext.get(k, ""), k)]


def _format_extraction(ext: dict) -> str:
    """把当前extraction格式化为可读文本"""
    def _val(k):
        v = ext.get(k, "")
        return v if v and not _is_empty(v, k) else "（暂无）"
    return "\n".join(f"{EN_TO_CN[k]}: {_val(k)}" for k in SEVEN_FIELDS)


def _parse_json(text: str, fallback: dict) -> dict:
    """从LLM输出中提取JSON（兼容R1思维链）"""
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        raw = match.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        raw = match.group(0) if match else ""

    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


# ====================== LLM调用函数 ======================

@traceable(name="llm_extract", run_type="chain")
async def _llm_extract(llm, title: str, content: str) -> dict:
    """初步提取：从网页原文提取7个字段"""
    prompt = f"""你是城市规划专家。请从以下案例原文中提取信息。

**标题**: {title}
**内容**: {content[:30000]}

请严格按以下JSON格式输出，不要输出其他内容：
```json
{{
  "city_country": "城市/国家，不确定则留空字符串",
  "time": "时间，不确定则留空字符串",
  "core_problem": "核心问题，不确定则留空字符串",
  "solution": "解决方案，不确定则留空字符串",
  "key_results": "关键数据/成果，不确定则留空字符串",
  "preconditions": "前置条件（制度/经济/技术/社会），不确定则留空字符串",
  "downsides": "潜在代价/负面影响，不确定则留空字符串"
}}
```"""
    fallback = {k: "" for k in SEVEN_FIELDS}
    try:
        resp = await llm.ainvoke([
            SystemMessage(content="你是信息提取专家，只输出JSON，不输出其他内容"),
            HumanMessage(content=prompt)
        ])
        result = _parse_json(extract_content(resp), fallback)
        for k in fallback:
            result.setdefault(k, "")
        return result
    except Exception as e:
        print(f"    ⚠️ 初步提取LLM调用失败: {str(e)[:80]}")
        return fallback


@traceable(name="llm_check_missing", run_type="chain")
async def _llm_check_missing(llm, title: str, content: str, ext: dict) -> Tuple[list, list]:
    """
    改进3: LLM审查缺失字段，返回标准化的英文key列表。
    将 missing 和 unobtainable 严格分开，不再合并。
    """
    prompt = f"""你是城市规划案例审查专家。请判断以下案例哪些字段信息缺失或不足。

【案例标题】{title}

【参考内容节选】
{content[:30000] if content else "（无内容）"}

【当前已提取信息】
{_format_extraction(ext)}

请审查以上所有内容，判断以下7个字段是否有实质内容：
1. 城市/国家 (city_country)
2. 时间 (time)
3. 核心问题 (core_problem)
4. 解决方案 (solution)
5. 关键数据/成果 (key_results)
6. 前置条件 (preconditions)
7. 潜在代价/负面影响 (downsides)

判断标准：
- 在参考内容或已提取信息中有具体、实质描述 → 视为"已有"，不列入缺失
- 仅有模糊提及或完全没有 → 视为"缺失"，列入 missing
- 该信息在此类官方资料中本就不会公开（谨慎判断）→ 列入 unobtainable

**重要**: 请在返回的字段名中使用括号内的英文key，例如 "city_country" 而非 "城市/国家"。

请严格按以下JSON格式输出，不要输出其他内容：
```json
{{
  "missing": ["english_key_1", "english_key_2"],
  "unobtainable": ["english_key_1"],
  "notes": "简短说明"
}}
```"""

    fallback_missing = _get_missing_fields(ext)
    try:
        resp = await llm.ainvoke([
            SystemMessage(content="你是案例信息审查专家，只输出JSON，不输出其他内容"),
            HumanMessage(content=prompt)
        ])
        result = _parse_json(extract_content(resp), {})
        raw_missing      = result.get("missing", [])
        raw_unobtainable = result.get("unobtainable", [])

        # 标准化字段名
        missing      = [_normalize_field_name(f) for f in raw_missing]
        unobtainable = [_normalize_field_name(f) for f in raw_unobtainable]

        # 过滤掉不在7字段范围内的名字（LLM幻觉保护）
        missing      = [f for f in missing if f in SEVEN_FIELDS]
        unobtainable = [f for f in unobtainable if f in SEVEN_FIELDS]

        if unobtainable:
            print(f"    ℹ️  LLM判断不可得字段: {unobtainable}")

        # -------- 改进4: 交叉验证 --------
        # 如果LLM说某字段缺失，但程序判断该字段有内容 → 信任程序，移除
        # 如果LLM说某字段已有，但程序判断该字段为空 → 信任LLM（可能是短值）
        verified_missing = []
        for f in missing:
            if f in unobtainable:
                continue  # 不可得字段不放入missing
            val = ext.get(f, "")
            if not _is_empty(val, f):
                # 程序认为有内容，LLM说缺失 → 打印警告，不加入missing
                print(f"    ⚠️ 交叉验证冲突: {f} 有值「{val[:30]}」但LLM判为缺失，以程序为准保留")
            else:
                verified_missing.append(f)

        return verified_missing, unobtainable

    except Exception as e:
        print(f"    ⚠️ LLM审查失败，退回程序校验: {str(e)[:60]}")
        return fallback_missing, []


@traceable(name="llm_summary", run_type="chain")
async def _llm_summary(llm, extraction: dict, round_raw: str) -> dict:
    """渐进式摘要：合并已有摘要和本轮新内容"""
    prompt = f"""你是城市规划专家。请根据【新增内容】补充完善【当前案例摘要】。

【当前案例摘要】（已有信息，请保留并补充）：
{_format_extraction(extraction)}

【本轮新增内容】：
{round_raw[:30000]}

规则：
- 已有实质内容的字段请完整保留，可在末尾追加新信息
- 空白字段尽量从新增内容中填写；若新增内容中确实没有该信息，保持空字符串
- is_complete：当7个字段均有实质内容时为true

请严格按以下JSON格式输出，不要输出其他内容：
```json
{{
  "city_country": "...",
  "time": "...",
  "core_problem": "...",
  "solution": "...",
  "key_results": "...",
  "preconditions": "...",
  "downsides": "...",
  "is_complete": true或false
}}
```"""
    fallback = {**extraction, "is_complete": False}
    try:
        resp = await llm.ainvoke([
            SystemMessage(content="你是信息整合专家，只输出JSON，不输出其他内容"),
            HumanMessage(content=prompt)
        ])
        result = _parse_json(extract_content(resp), fallback)
        # -------- 改进5: 防止摘要覆盖已有内容 --------
        for k in SEVEN_FIELDS:
            new_val = result.get(k, "")
            old_val = extraction.get(k, "")
            # 如果新值为空但旧值有内容，保留旧值（防止LLM丢失信息）
            if _is_empty(new_val, k) and not _is_empty(old_val, k):
                result[k] = old_val
        result.setdefault("is_complete", False)
        return result
    except Exception as e:
        print(f"    ⚠️ 摘要LLM调用失败: {str(e)[:80]}")
        return fallback


@traceable(name="llm_decide", run_type="chain")
async def _llm_decide(llm, title: str, extraction: dict,
                search_log: list, round_raw: str = "") -> dict:
    """搜索决策：LLM决定是否继续以及搜什么"""
    history_str = "\n".join([
        f"  第{i+1}轮: 查询={r['queries']} | 收获={r['gain_summary']}"
        for i, r in enumerate(search_log)
    ]) if search_log else "  （尚未进行任何搜索）"

    new_content_block = (
        f"\n【本轮新抓取内容节选】\n{round_raw[:30000]}"
        if round_raw else ""
    )

    unobtainable = extraction.get("unobtainable_fields", [])
    unobtainable_block = (
        f"\n【已判定不可得字段（无需再搜）】{unobtainable}"
        if unobtainable else ""
    )

    # -------- 改进6: 只展示真正可搜索的缺失字段 --------
    searchable_missing = [
        f for f in extraction.get("missing_aspects", [])
        if f not in unobtainable
    ]

    prompt = f"""你是城市规划案例研究专家。请决定是否继续补充搜索。

【案例标题】{title}

【当前已有信息】
{_format_extraction(extraction)}

【仍需搜索的缺失字段】{searchable_missing}{unobtainable_block}

【历史搜索记录】
{history_str}
{new_content_block}

【决策思路】
1. 若"仍需搜索的缺失字段"为空列表 → 应终止
2. 缺失字段是否可能在公开资料中找到？
3. 连续2轮收获为"无新增字段"则应终止
4. 若继续，下一轮应换什么搜索角度（与历史查询明显不同）

请严格按以下JSON格式输出，不要输出其他内容：
```json
{{
  "should_continue": true或false,
  "next_queries": ["搜索词1", "搜索词2"],
  "stop_reason": "若终止则填写原因，否则留空字符串"
}}
```"""

    fallback = {"should_continue": False,
                "next_queries": [],
                "stop_reason": "决策LLM解析失败，默认终止"}
    try:
        resp = await llm.ainvoke([
            SystemMessage(content="你是搜索策略专家，只输出JSON，不输出其他内容"),
            HumanMessage(content=prompt)
        ])
        result = _parse_json(extract_content(resp), fallback)
        result.setdefault("should_continue", False)
        result.setdefault("next_queries", [])
        result.setdefault("stop_reason", "")
        return result
    except Exception as e:
        print(f"    ⚠️ 决策LLM调用失败: {str(e)[:60]}")
        return fallback


# ====================== 核心函数 ======================

@traceable(name="deep_case_research", run_type="chain")
async def deep_case_research(
    case: dict,
    initial_content: str,
    llm,
    chat,
    search_serper=None,
    fetch_webpage_content=None,
    max_loops: int = 5
) -> dict:
    """
    LLM决策驱动的深度补全（异步改进版）。

    关键改进:
    1. _is_empty 按字段使用不同阈值，不再误判短值
    2. 字段名标准化，消除中英文不匹配
    3. missing 与 unobtainable 严格分离
    4. is_complete 计算时排除 unobtainable
    5. 交叉验证: LLM判断 + 程序判断互相校验
    6. 摘要更新时防止已有信息被覆盖
    7. 异步 LLM 调用 + 异步搜索/抓取
    """
    from services.search_service import SearchService
    from services.fetch_service import async_fetch_webpage_content

    title = case.get("title", "未知案例")
    print(f"🚀 LLM决策深度研究: {title}")

    # ── 第1步：初步提取 + LLM审查缺失字段 ───────────────────────
    extraction: dict = {}

    if initial_content and len(initial_content) >= 2000:
        print("    🔍 初步信息提取（7字段）...")
        extraction = await _llm_extract(llm, title, initial_content)

        print("    🔎 LLM审查缺失字段...")
        missing, unobtainable = await _llm_check_missing(
            chat, title, initial_content, extraction
        )
        extraction["missing_aspects"]     = missing
        extraction["unobtainable_fields"] = unobtainable
        print(f"    ✅ 审查完成 | 缺失: {missing} | 不可得: {unobtainable}")
    else:
        print("    ⚠️ 初始内容不足2000字符，跳过初步提取")
        extraction = {k: "" for k in SEVEN_FIELDS}
        extraction["missing_aspects"]     = list(_get_missing_fields(extraction))
        extraction["unobtainable_fields"] = []

    loop_count = 0
    search_log: list = []

    # ── 改进7: is_complete 判断排除 unobtainable ──────────────────
    searchable_missing = [
        f for f in extraction.get("missing_aspects", [])
        if f not in extraction.get("unobtainable_fields", [])
    ]

    if not searchable_missing:
        print("    ✅ 初步提取已完整（或剩余字段均不可得），跳过补全搜索")
    else:
        print(f"\n    🤔 LLM决策初始搜索方向...")
        first_decision = await _llm_decide(chat, title, extraction, search_log)

        if not first_decision["should_continue"]:
            print(f"    🛑 LLM判断无需补充搜索: {first_decision['stop_reason']}")
        else:
            current_queries = first_decision["next_queries"]
            print(f"    ➡️  初始查询: {current_queries}")

            while loop_count < max_loops:
                loop_count += 1
                print(f"\n    🔎 第{loop_count}/{max_loops}轮")
                print(f"       查询: {current_queries}")

                # ── 搜索 + 抓取（同轮内多查询并发）────────────────
                round_raw = ""
                for query in current_queries:
                    results = await SearchService.search_serper(query, max_results=4) or []
                    fetch_tasks = [
                        async_fetch_webpage_content(res["url"])
                        for res in results[:3]
                    ]
                    snippets = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                    for i, snippet in enumerate(snippets):
                        if isinstance(snippet, Exception):
                            snippet = results[i].get("snippet", "") if i < len(results) else ""
                        round_raw += f"\n---来源: {results[i].get('title','') if i < len(results) else ''}\n{snippet[:10000]}\n"

                print(f"       抓取完成: {len(round_raw)} 字符")

                # ── 渐进式摘要更新 ────────────────────────────────
                print("    📝 渐进式摘要更新...")
                prev_missing = set(extraction.get("missing_aspects", []))

                updated = await _llm_summary(llm, extraction, round_raw)
                extraction.update(updated)

                # LLM重新审查（同时传入round_raw和当前extraction）
                print("    🔎 LLM重新审查缺失字段...")
                missing, unobtainable = await _llm_check_missing(
                    chat, title, round_raw, extraction
                )
                extraction["missing_aspects"]     = missing
                extraction["unobtainable_fields"] = unobtainable

                # -------- 改进7: is_complete 排除不可得 --------
                searchable_missing = [
                    f for f in missing if f not in unobtainable
                ]
                extraction["is_complete"] = len(searchable_missing) == 0

                newly_filled = prev_missing - set(missing)
                gain_summary = (
                    f"新填补: {list(newly_filled)}"
                    if newly_filled else "无新增字段"
                )
                print(f"    ✅ {gain_summary} | 仍缺: {searchable_missing} | 不可得: {unobtainable}")

                search_log.append({
                    "queries":      current_queries,
                    "gain_summary": gain_summary
                })

                if extraction.get("is_complete"):
                    print("    🎯 所有可搜索字段已完整，提前结束")
                    break

                if loop_count < max_loops:
                    print("    🤔 LLM决策下一轮...")
                    decision = await _llm_decide(
                        chat, title, extraction, search_log, round_raw
                    )
                    if not decision["should_continue"]:
                        print(f"    🛑 LLM决定终止: {decision['stop_reason']}")
                        break
                    current_queries = decision["next_queries"]
                    print(f"    ➡️  下轮查询: {current_queries}")
                else:
                    print(f"    🔚 已达上限 {max_loops} 轮，停止")

    # ── 第3步：最终报告 ──────────────────────────────────────────
    print("\n📝 生成最终案例报告...")

    report_prompt = f"""请对以下案例进行综合分析（用中文输出）：
    **案例标题**: {case['title']}
    **初步提取信息**:{initial_content}
    **后续完整信息**: {_format_extraction(extraction)}

    仍缺失信息: {extraction.get('missing_aspects', [])}
    不可得信息: {extraction.get('unobtainable_fields', [])}
    注：- 标注"不可得"的字段在公开资料中不存在，请在报告中说明而非留空。
        - 不需要JSON，直接markdown格式
        - 缺失信息请标注，不要直接推测。
        - 直接输出报告内容，不要包含任何解释或前置文本，比如，好的，作为资深城市规划报告撰写专家，我将基于您提供的全部信息，为您生成一份专业、全面的报告。
    
    报告结构要求：

    1. **案例来源**
       - 1.1 网址链接：{case.get('url', '未知')}
       - 1.2 案例标题: {case['title']}
       - 1.3 案例来源：（从初步的提取信息中判断，从以下类型中选择，或标注"未知"）
            a. 外国政府官方规划文件（法定图则、政策白皮书、议会报告或者具体部门研究）
            b. 政府委托公开研究报告（标注发布机构）
            c. 学术研究（标注期刊名字）
            d. 专业机构报告（ULI、RICS、ISOCARP 等）
            e. 行业咨询报告（Savills、CBRE 等市场研究）
            f. 新闻报道、项目宣传材料、无法核查来源
    
    2. **基本信息**
       - 2.1 城市/国家: {extraction.get('city', '未知')}
       - 2.2 时间: {extraction.get('time', '未知')}
       - 2.3 背景: {extraction.get('background', '未知')}

    3. **核心问题**
       - 3.1 问题描述
       - 3.2 项目难点/限制

    4. **解决方案**
       - 4.1 具体措施
       - 4.2 关键技术/政策工具

    5. **实施成果**
       - 5.1 定量成果
       - 5.2 定性影响

    6. **前置条件**
       - 6.1 制度条件
       - 6.2 经济条件
       - 6.3 技术条件
       - 6.4 社会条件

    7. **潜在代价/负面影响**
       - 7.1 经济代价
       - 7.2 社会影响
       - 7.3 实施风险
       - 7.4 长期挑战

    8. **可借鉴性评估**
       - 8.1 适用情境
       - 8.2 迁移难度

    ## 撰写注意事项：
    1. 按以上结构分节撰写
    2. 对缺失信息注明"资料不足，待补充"，对不可得信息注明"公开资料不涉及"
    3. 风格：专业、客观、中文，信息尽量详细完整。
    4. 输出必须严谨，不能编造或者自行推测不存在的信息,所有内容必须基于提供的信息和分析结果，不能凭空捏造数据或案例。
    5. 结构清晰，层次分明，使用适当的标题和小标题，便于阅读和理解。
    6. 内容详实：每个部分都要有充分的分析和细节支持

    请直接输出最终报告内容，不要包含任何解释或前置文本。"""

    try:
        resp = await llm.ainvoke([
            SystemMessage(content="你是资深城市规划报告专家"),
            HumanMessage(content=report_prompt)
        ])
        final_report = resp.content
    except Exception as e:
        final_report = f"报告生成失败: {str(e)}"

    complete_count = 7 - len(extraction.get("missing_aspects", []))
    print(f"✅ 研究完成！{loop_count}轮 | 字段完整度: {complete_count}/7")
    for i, r in enumerate(search_log):
        print(f"   第{i+1}轮: {r['gain_summary']}")

    return {
        "extraction":   extraction,
        "final_report": final_report,
        "loop_count":   loop_count,
        "search_log":   search_log,
        "tree": [
            {"id": f"round_{i}", "query": str(r["queries"]), "gain": r["gain_summary"]}
            for i, r in enumerate(search_log)
        ],
    }