"""
实验配置 — 用配置控制实验变量，不改代码
"""

# 测试查询集
TEST_QUERIES = [
    "新田科技城城市规划相关案例分析",
    "葵涌城市规划策略研究",
    "将军澳城市规划策略研究",
    "元朗新市镇城市更新策略研究",
    "米埔生态保护区（錦綉花園）城市发展案例研究",
    "屯门城市更新案例研究",
    "沙田城市更新案例研究",
    "九龙城市更新案例研究",
    "北角城市更新案例研究",
    "洪水桥-厦村城市规划相关案例分析",
    "西九龙文化区城市规划相关案例分析",
    "文锦渡口岸（上水城郊及沙打）城市规划相关案例分析",
    "深水埗城市更新案例研究",
    "尖沙咀城市更新案例研究",
    "湾仔城市更新案例研究",
    "坚尼地城城市更新案例研究",
    "鴨脷洲城市更新案例研究",
    "香港仔城市更新案例研究",
    "流浮山城市规划案例分析",
    "启德发展区滨水空间设计参考",
    "天水围社区设施优化方案案例参考",
    "牛潭尾城市规划案例分析",
]


# 实验组定义
EXPERIMENTS = {
    # ── 主实验 ──
    "workflow_full": {
        "mode": "workflow",
        "use_local_analysis": True,
        "use_web_search": True,
        "use_hybrid_retrieval": True,
        "use_llm_selection": True,
        "use_deep_research": True,
        "description": "完整 Workflow 模式",
    },
    "plan_execute_full": {
        "mode": "plan_execute",
        "use_local_analysis": True,
        "use_web_search": True,
        "use_hybrid_retrieval": True,
        "use_llm_selection": True,
        "use_deep_research": True,
        "description": "完整 Plan-Execute 模式",
    },

    # ── 消融实验 ──
    "ablation_no_local_and_web": {
        "mode": "workflow",
        "use_local_analysis": False,    # 跳过本地数据分析
        "use_web_search": False,        # 跳过网络信息搜索
        "use_hybrid_retrieval": True,
        "use_llm_selection": True,
        "use_deep_research": True,
        "use_gap_analysis": True, 
        "description": "消融: 去掉本地数据分析 + 网络信息搜索",
    },
    "ablation_no_hybrid_no_llm_select": {
        "mode": "workflow",
        "use_local_analysis": True,
        "use_web_search": True,
        "use_hybrid_retrieval": False,  # 只用 BM25
        "use_llm_selection": False,     # 不用 LLM 选择，直接取 BM25 排名前 N
        "use_deep_research": False,
        "use_gap_analysis": True, 
        "description": "消融: 去掉混合检索 + LLM 选择案例",
    },
    "ablation_no_deep_research": {
        "mode": "workflow",
        "use_local_analysis": True,
        "use_web_search": True,
        "use_hybrid_retrieval": True,
        "use_llm_selection": True,
        "use_deep_research": False,     # 跳过深度研究
        "use_gap_analysis": True, 
        "description": "消融: 去掉深度研究",
    },

    "ablation_no_gap_analysis": {
    "mode": "workflow",
    "use_local_analysis": True,
    "use_web_search": True,
    "use_hybrid_retrieval": True,
    "use_llm_selection": True,
    "use_deep_research": True,
    "use_gap_analysis": False,     # 新增：跳过差异分析
    "description": "消融: 去掉差异分析（gap analysis）",
},


    # ── Baseline ──
    "baseline_single_llm": {
        "mode": "single_llm",
        "description": "Baseline: 单个 LLM 直接生成",
    },
}

# 输出目录结构规范
# 每次实验运行的输出会按以下结构存放:
#
# output/experiments/
# ├── workflow_full/
# │   ├── query_1_新田科技城/
# │   │   ├── meta.json              ← 运行元信息（查询、模式、耗时、评估分数）
# │   │   ├── final_report.md        ← 最终报告
# │   │   ├── cases/                 ← 各案例的 comprehensive analysis
# │   │   │   ├── case_1.json        ← 结构化字段（title, url, city_country, solution...）
# │   │   │   ├── case_1_report.md   ← 完整分析报告
# │   │   │   ├── case_2.json
# │   │   │   ├── case_2_report.md
# │   │   │   └── ...
# │   │   ├── context.json           ← 本地情境分析（matched_area, core_problems, local_data）
# │   │   └── gap_analysis.md        ← 差异分析报告
# │   ├── query_2_油尖旺区/
# │   │   └── ...
# │   └── summary.json               ← 该实验组所有查询的汇总指标
# ├── plan_execute_full/
# │   └── ...
# ├── ablation_no_local_and_web/
# │   └── ...
# └── comparison.csv                  ← 跨实验组对比表