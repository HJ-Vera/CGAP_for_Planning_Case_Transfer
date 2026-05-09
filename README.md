# 多智能体城市规划案例分析系统

基于 LangGraph 的多智能体系统，用于城市规划方案的全球案例研究与本地化适配分析。

## 项目结构

```
urban_planning_agents/
├── main.py                      # Workflow 模式主入口
├── main_pe.py                   # Plan-Execute 模式主入口
├── config.py                    # 配置文件（API Keys、路径、参数）
├── state.py                     # AgentState 全局状态定义
├── llm.py                       # LLM 初始化（DeepSeek / Gemini / MiniMax）
├── router.py                    # 工作流路由函数
├── workflow.py                  # LangGraph 工作流构建（Workflow 模式）
├── server.py                    # FastAPI Web 服务端（SSE 流式输出）
├── checkpoint.py                # 断点恢复（agent 完成后自动保存 state）
├── download.py                  # HuggingFace 模型下载脚本
├── requirements.txt             # Python 核心依赖
├── requirements_full.txt        # Python 完整依赖
├── README.md
│
├── agents/                      # 智能体模块
│   ├── __init__.py
│   ├── scenario_agent.py        # 智能体1: 情景解构
│   ├── case_query_agent.py      # 智能体2: 案例查询（混合检索）
│   ├── gap_analysis_agent.py    # 智能体3: 差异分析与改造
│   ├── evaluation_agent.py      # 智能体4: 评审
│   ├── report_generator.py      # 最终报告生成
│   └── feedback.py              # 反馈循环
│
├── tools/                       # 工具模块
│   ├── __init__.py
│   ├── data_loader.py           # 数据加载（Excel/CSV/MD）
│   ├── data_analysis.py         # 数据分析与可视化（聚类、PCA、地图）
│   ├── search.py                # 搜索工具（Serper/Scholar/ArXiv）
│   ├── web_fetcher.py           # 网页抓取 + PDF 提取
│   ├── retrieval.py             # BM25 + Sentence-BERT 混合检索
│   ├── HybridRetriever.py       # 混合检索器封装
│   └── deep_research.py         # Gap-Driven 深度案例研究
│
├── plan_execute/                # Plan-Execute 模式
│   ├── __init__.py
│   ├── plan_agent.py            # 规划智能体
│   ├── sub_agents.py            # 子智能体
│   ├── sub_2.py                 # 子智能体2
│   ├── pe_state.py              # Plan-Execute 状态定义
│   ├── pe_workflow.py           # Plan-Execute 工作流
│   └── pe_report.py             # Plan-Execute 报告生成
│
├── experiments/                 # 实验模块（消融实验）
│   ├── __init__.py
│   ├── exp_config.py            # 实验配置
│   ├── exp_flags.py             # 实验标志位
│   ├── runner.py                # 实验运行器
│   └── analyze_results.py       # 实验结果分析
│
├── web/                         # Web 前端界面
│   ├── index.html               # 主页面
│   └── static/
│       ├── app.js               # 前端逻辑
│       └── style.css            # 样式
│
├── models/                      # 本地模型缓存
│   └── paraphrase-multilingual-MiniLM-L12-v2/
│
├── data/                        # 数据文件目录（需自行放入）
│   ├── hongkong_llm_data.xlsx   # 香港区域数据表
│   ├── 香港规划情景知识库.md      # 规划情景知识库
│   ├── HK_Planning_Knowledge.md # 规划法规知识库
│   ├── hk_districts.geojson     # 地图文件
│   └── README.md
│
├── eval/                        # 评估结果
│   ├── baseline/                # 基线实验
│   ├── no_gap/                  # 消融: 无差异分析
│   ├── no_local/                # 消融: 无本地化适配
│   ├── no_rag/                  # 消融: 无 RAG 检索
│   └── plan/                    # Plan-Execute 评估
│
└── outputtt/                    # 输出目录（运行时生成）
    ├── planning_report.md
    └── case_summary_*.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

系统依赖（PDF OCR，可选）：
```bash
# Ubuntu/Debian
sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim poppler-utils

# macOS
brew install tesseract poppler
```

中文字体（可视化，可选）：
```bash
# Ubuntu/Debian
sudo apt-get install -y fonts-noto-cjk
```

### 2. 配置 API Keys

编辑 `config.py`，填入你的 API Keys：

```python
DEEPSEEK_API_KEY = "sk-..."
SERPER_API_KEY = "..."
# 其他可选 Keys
```

### 3. 准备数据

将以下文件放入 `data/` 目录：
- `hongkong_llm_data.xlsx` — 香港区域数据表
- `香港规划情景知识库.md` — 规划情景知识库
- `HK_Planning_Knowledge.md` — 规划法规知识库

### 4. 运行

```bash
python main.py
```

或修改 `main.py` 底部的 `USER_QUERY` 变量来更改查询问题。

## 工作流

```
情景解构 → 案例查询(×3) → 差异分析 → 评审 → [通过]报告生成
                                        ↓
                                    [未通过]反馈循环 → 差异分析
```

