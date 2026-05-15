import asyncio
import json
import numpy as np

from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from config import TOKEN_LIMITS, KNOWLEDGE_BASE_MD, HONGKONG_DATA_FILE
from llm import get_llm
from tools.data_loader import load_hongkong_data, read_md_file
from prompts import load_prompt
from tools.data_analysis import analyze_regional_data
from services.search_service import SearchService
from services.llm_service import LLMService
from .case_query_agent import extract_content


async def scenario_deconstruction_agent(state: AgentState) -> AgentState:
    """
    情景解构智能体
    - 智能理解数据表结构
    - 智能匹配区域
    - 具体查找规划区域的地理空间特质，包括容积率，建筑情况，道路交通状况；可能相关的法律法规，包括香港2030，城市规划条例，土地征收程序；以及潜在的利益相关方和他们之间的冲突，谁支持，谁反对
    - 深度分析本地情境
    - 重写问题
    - 输出调研报告
    """
    print("\n" + "="*60)
    print("🔍 情景解构智能体启动 (智能分析模式)")
    print("="*60)

    user_query = state["user_query"]
    target_city = state["target_city"]

    # ========== 第一步: 加载并理解数据表 ==========
    print("\n📊 步骤 1: 加载并理解数据表结构...")
    df = load_hongkong_data(HONGKONG_DATA_FILE)

    # 生成数据表元信息
    table_metadata = {
        "列名": list(df.columns),
        "行数": len(df),
        "列数": len(df.columns),
        "数据预览": df.head(3).to_dict('records'),
        "列数据类型": df.dtypes.to_dict()
    }

    print(f"  📋 数据表包含 {len(df)} 行, {len(df.columns)} 列")
    print(f"  📋 列名: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")

    # ========== 第二步: 让 LLM 理解数据表 ==========
    print("\n🧠 步骤 2: AI 理解数据表含义...")

    table_understanding_prompt = load_prompt(
        "agents/scenario_agent", "01_table_understanding_prompt",
        df_columns=str(list(df.columns)),
        df_rows=str(len(df)),
        df_head=df.head(3).to_string(),
    )

    messages = [
        SystemMessage(content="你是数据分析专家"),
        HumanMessage(content=table_understanding_prompt)
    ]
    response_text = await LLMService.ainvoke("chat", max_tokens=4500, messages=messages)
    print(f"  AI理解: {response_text}")

    # 智能提取区域列名
    area_column = None
    for col in df.columns:
        col_str = str(col)
        if any(keyword in col_str for keyword in ["区", "區", "district", "area", "選區", "地区", "region"]):
            area_column = col
            break

    if area_column is None:
        area_column = df.columns[0]
        print(f"  ⚠️ 使用第一列作为区域列: {area_column}")
    else:
        print(f"  ✅ 识别区域列: {area_column}")

    table_summary = response_text  # 使用前300字符作为总结

    # ========== 第三步: 智能匹配区域 ==========
    print(f"\n🎯 步骤 3: 智能匹配区域 '{user_query}'")

    # 提取可能的区域名
    area_name = user_query.split("更新")[0].strip() if "更新" in user_query else user_query.strip()

    # 获取所有区域选项
    all_areas = df[area_column].astype(str).tolist()
    print(f"  📍 数据表中的所有区域: {', '.join(all_areas[:5])}{'...' if len(all_areas) > 5 else ''}")

    # 让 LLM 进行智能匹配
    area_matching_prompt = load_prompt(
        "agents/scenario_agent", "02_area_matching_prompt",
        user_query=user_query,
        area_name=area_name,
        table_summary=table_summary,
        all_areas=", ".join(all_areas),
    )

    messages = [
        SystemMessage(content="你是一位香港选区地名匹配专家，熟悉香港所有行政区划及地理常识。"),
        HumanMessage(content=area_matching_prompt)
    ]
    response_text = await LLMService.ainvoke("default", max_tokens=TOKEN_LIMITS["scenario_agent"], messages=messages)
    response_text = response_text.strip()
    print(f"  AI回应: {response_text}")

    # 尝试在响应中找到匹配的区域（可能多个）
    matched_areas = []
    for area in all_areas:
        if area in response_text:
            matched_areas.append(area)

    # 如果AI回应中没有找到，使用模糊匹配
    if not matched_areas:
        for area in all_areas:
            if area_name in str(area) or str(area) in area_name:
                matched_areas.append(area)
                break

    # 确定主要匹配区域（用于向后兼容）
    # 保存原始匹配区域名（用于数据提取）
    original_matched_area = matched_areas[0] if matched_areas else area_name

    # 如果有多个区域，将区域列表转换为字符串表示（用顿号分隔）
    if len(matched_areas) > 1:
        matched_area = "、".join(matched_areas[:5]) + (f"等{len(matched_areas)}个区域" if len(matched_areas) > 5 else "")
    else:
        matched_area = original_matched_area

    if len(matched_areas) > 1:
        print(f"  ✅ 匹配到 {len(matched_areas)} 个区域: {', '.join(matched_areas[:5])}{'...' if len(matched_areas) > 5 else ''}")
        print(f"  📍 匹配区域显示: {matched_area}")
    else:
        print(f"  ✅ 匹配到区域: {matched_area}")

    # ========== 第四步: 提取匹配区域的详细数据 ==========
    if len(matched_areas) > 1:
        print(f"\n📈 步骤 4: 提取 {len(matched_areas)} 个匹配区域的详细数据并计算平均值...")
        matched_rows_data = []
        try:
            # 提取所有匹配区域的数据行
            matched_rows = df[df[area_column].astype(str).isin(matched_areas)]
            if not matched_rows.empty:
                print(f"  ✅ 找到 {len(matched_rows)} 个匹配区域的数据行")
                # 计算数值列的平均值
                numeric_cols = matched_rows.select_dtypes(include=[np.number]).columns
                avg_data = {}
                # 区域列使用多个区域名的拼接（与matched_area一致）
                avg_data[area_column] = matched_area
                # 对于数值列，计算平均值
                for col in numeric_cols:
                    avg_data[col] = matched_rows[col].mean()
                # 对于非数值列，使用第一个区域的值或留空
                non_numeric_cols = matched_rows.select_dtypes(exclude=[np.number]).columns
                for col in non_numeric_cols:
                    if col != area_column:
                        avg_data[col] = matched_rows[col].iloc[0] if not matched_rows[col].empty else ""
                local_info = avg_data
                print(f"  ✅ 成功计算平均值，生成平均区域数据")
                # 保存原始匹配区域的数据行，以备后用
                matched_rows_data = matched_rows.to_dict('records')
            else:
                # 如果没有找到数据，使用第一个匹配区域进行模糊匹配
                print(f"  ⚠️ 未找到匹配区域的数据行，尝试模糊匹配第一个区域")
                local_data_row = df[df[area_column].astype(str).str.contains(original_matched_area, na=False)]
                if not local_data_row.empty:
                    local_info = local_data_row.iloc[0].to_dict()
                    print(f"  ✅ 通过模糊匹配提取数据")
                else:
                    local_info = {area_column: matched_area, "备注": "未找到详细数据"}
                    print(f"  ⚠️ 未找到数据,使用默认信息")
                    matched_rows_data = []
        except Exception as e:
            print(f"  ⚠️ 数据提取失败: {e}")
            local_info = {area_column: matched_area, "备注": "数据提取失败"}
            matched_rows_data = []
    else:
        print(f"\n📈 步骤 4: 提取区域 '{matched_area}' 的详细数据...")
        matched_rows_data = []
        try:
            local_data_row = df[df[area_column].astype(str) == matched_area]

            if not local_data_row.empty:
                local_info = local_data_row.iloc[0].to_dict()
                matched_rows_data = [local_info]
                print(f"  ✅ 成功提取 {len(local_info)} 个数据字段")
            else:
                # 再次尝试模糊匹配
                local_data_row = df[df[area_column].astype(str).str.contains(matched_area, na=False)]
                if not local_data_row.empty:
                    local_info = local_data_row.iloc[0].to_dict()
                    matched_rows_data = [local_info]
                    print(f"  ✅ 通过模糊匹配提取数据")
                else:
                    local_info = {area_column: matched_area, "备注": "未找到详细数据"}
                    print(f"  ⚠️ 未找到数据,使用默认信息")

        except Exception as e:
            print(f"  ⚠️ 数据提取失败: {e}")
            local_info = {area_column: matched_area, "备注": "数据提取失败"}

    # ========== 第五步: 深度分析本地数据 ==========
    md_content = ""
    try:
        from experiments.exp_flags import USE_LOCAL_ANALYSIS
    except ImportError:
        USE_LOCAL_ANALYSIS = True

    if USE_LOCAL_ANALYSIS:
        print(f"\n🔬 步骤 5: 深度分析本地数据...")
        from config import OUTPUT_DIR
        result_text = analyze_regional_data(df, matched_area, output_dir=OUTPUT_DIR, matched_areas=matched_areas)
        md_content = read_md_file(KNOWLEDGE_BASE_MD)
        print(result_text)
    else:
        print(f"\n⏭️ 步骤 5: 跳过本地数据分析（消融模式）")
        result_text = f"（消融实验：跳过本地数据分析。匹配区域: {matched_area}）"

    # ========== 第六步: 搜索网络信息 ==========
    try:
        from experiments.exp_flags import USE_WEB_SEARCH
    except ImportError:
        USE_WEB_SEARCH = True

    if USE_WEB_SEARCH:
        print("\n🌐 步骤 6: 搜索网络舆论和相关信息 (Serper API)")
        public_opinion_query = f"{target_city} {original_matched_area} 土地权属 城市规划争议 利益相关方 空间问题"
        opinion_results = await SearchService.search_serper(public_opinion_query, max_results=10)
        opinion_summary = "\n".join([
            f"- {r['title']}: {r['snippet']}"
            for r in opinion_results[:15]
        ])
        print(f"  ✅ 找到 {len(opinion_results)} 条相关信息")
    else:
        print("\n⏭️ 步骤 6: 跳过网络信息搜索（消融模式）")
        opinion_summary = "（消融实验：跳过网络信息搜索）"

    # ========== 第五/六步的 LLM 分析并发 ==========
    print("\n🧠 步骤 5-6: 并行分析本地数据与网络信息...")

    data_analysis_prompt = load_prompt(
        "agents/scenario_agent", "03_data_analysis_prompt",
        matched_area=matched_area,
        local_info_json=json.dumps(local_info, ensure_ascii=False, indent=2),
        result_text=result_text,
    )

    network_analysis_prompt = load_prompt(
        "agents/scenario_agent", "04_network_analysis_prompt",
        matched_area=matched_area,
        opinion_summary=opinion_summary,
    )

    data_analysis_text, opinion_summary = await asyncio.gather(
        LLMService.ainvoke("default", max_tokens=TOKEN_LIMITS["scenario_agent"], messages=[
            SystemMessage(content="你是城市规划数据分析专家"),
            HumanMessage(content=data_analysis_prompt)
        ]),
        LLMService.ainvoke("chat", max_tokens=4500, messages=[
            SystemMessage(content="你是网络信息分析专家"),
            HumanMessage(content=network_analysis_prompt)
        ])
    )
    data_analysis_text = data_analysis_text.strip()
    opinion_summary = opinion_summary.strip()

    print(f"  ✅ 数据分析: {data_analysis_text}")
    print(f"  ✅ 网络分析: {opinion_summary}")

    # ========== 第七步: 综合分析并重写问题 ==========
    print("\n✍️ 步骤 7: 综合分析并重写核心问题...")

    problem_rewriting_prompt = load_prompt(
        "agents/scenario_agent", "05_problem_rewriting_prompt",
        user_query=user_query,
        target_city=target_city,
        matched_area=matched_area,
        md_content=md_content,
        local_info_json=json.dumps(local_info, ensure_ascii=False),
        data_analysis_text=data_analysis_text,
        opinion_summary=opinion_summary,
    )

    messages = [
        SystemMessage(content="你是城市规划专家"),
        HumanMessage(content=problem_rewriting_prompt)
    ]
    response_text = await LLMService.ainvoke("default", max_tokens=TOKEN_LIMITS["scenario_agent"], messages=messages)
    response_text = response_text.strip()
    
    print(f"\n✅ LLM 情景分析完成，响应长度: {len(response_text)} 字符\n")

    # 灵活解析响应
    core_problems = []
    rewritten_problems = []
    context_summary = ""

    # 尝试提取核心问题
    if "【核心问题】" in response_text:
        parts = response_text.split("【核心问题】")[1].split("【情景化重写】")
        core_section = parts[0] if parts else ""

        for line in core_section.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                problem = line.lstrip('0123456789.-) ').strip()
                if problem:
                    core_problems.append(problem)

    # 尝试提取重写问题
    if "【情景化重写】" in response_text:
        parts = response_text.split("【情景化重写】")[1].split("【情境总结】")
        rewrite_section = parts[0] if parts else ""

        for line in rewrite_section.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                problem = line.lstrip('0123456789.-) ').strip()
                if problem:
                    rewritten_problems.append(problem)

    # 提取情境总结
    if "【情境总结】" in response_text:
        context_summary = response_text.split("【情境总结】")[1].strip()

    # 备用方案
    if len(core_problems) < 3:
        core_problems = [
            f"{matched_area}的交通优化问题",
            f"{matched_area}的住房改善问题",
            f"{matched_area}的公共空间提升问题"
        ]

    if len(rewritten_problems) < 3:
        rewritten_problems = [
            f"如何在{matched_area}优化交通系统",
            f"如何改善{matched_area}的住房条件",
            f"如何提升{matched_area}的公共空间质量"
        ]

    if not context_summary:
        context_summary = f"{matched_area}是{target_city}的重要区域，面临城市更新的多重挑战。"

    # 保存到状态
    state["core_problems"] = core_problems[:3]
    state["rewritten_problems"] = rewritten_problems[:3]
    state["local_context"] = {
        "matched_area": matched_area,
        "matched_areas": matched_areas,
        "matched_rows_data": matched_rows_data,
        "local_data": local_info,
        "data_analysis": data_analysis_text,
        "public_opinion": opinion_summary,
        "context_summary": context_summary,
        "full_response": response_text
    }

    print(f"\n" + "="*60)
    print("✅ 情景解构完成!")
    print("="*60)
    print(f"\n📍 匹配区域: {matched_area}")
    print(f"\n🎯 核心问题:")
    for i, problem in enumerate(state['core_problems'], 1):
        print(f"  {i}. {problem}")
    print(f"\n📝 情景化重写:")
    for i, problem in enumerate(state['rewritten_problems'], 1):
        print(f"  {i}. {problem}")
    print(f"\n💡 情境总结: {context_summary}")
    import os
    from config import OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "context_analysis.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(state, indent=2, ensure_ascii=False))
    
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "context_analysis.md"), "w", encoding="utf-8") as f:
        f.write(response_text)

    return state