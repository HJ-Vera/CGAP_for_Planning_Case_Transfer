"""
配置文件 — 从 .env 加载 API Keys 和全局参数
"""

import os
from dotenv import load_dotenv

load_dotenv()  # 从 .env 文件加载环境变量

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'

# ========================== API Keys ==========================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

CLAUDE_API_BASE = os.environ.get("CLAUDE_API_BASE", "")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_API_BASE = os.environ.get("MINIMAX_API_BASE", "https://api.minimax.chat/v1")
MINIMAX_MODEL = os.environ.get("MINIMAX_MODEL", "")

QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

GLM_API_KEY = os.environ.get("GLM_API_KEY", "")
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

# ========================== Token 限制配置 ==========================
TOKEN_LIMITS = {
    "scenario_agent": 50000,
    "case_query_agent": 50000,
    "gap_analysis_agent": 50000,
    "evaluation_agent": 50000,
}

# ========================== 检索参数 ==========================
BM25_CANDIDATE_K = 20       # BM25 筛选保留前 N 条
FINAL_CASE_COUNT = 10      # 最终选出的案例数
HYBRID_ALPHA = 0.5         # 混合权重: alpha*BM25 + (1-alpha)*SBERT
SBERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# ========================== 搜索参数 ==========================
SKIP_SEMANTIC_SCHOLAR = True  # 设为 True 可跳过 Semantic Scholar

SEARCH_MAX_RESULTS = 50     # 每次搜索返回的最大结果数
ACADEMIC_SEARCH_LIMIT = 50  # 学术搜索返回的最大结果数

# ========================== 文件路径 ==========================
DATA_DIR = "./data"
OUTPUT_DIR = "./output"
HONGKONG_DATA_FILE = f"{DATA_DIR}/hongkong_llm_data.xlsx"
KNOWLEDGE_BASE_MD = f"{DATA_DIR}/香港规划情景知识库.md"
PLANNING_KNOWLEDGE_MD = f"{DATA_DIR}/HK_Planning_Knowledge.md"
GEOJSON_PATH = f"{DATA_DIR}/hk_districts.geojson"  # 可选，地图文件
