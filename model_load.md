# 模型加载指南

本项目使用两类模型：**远程 LLM API** 和 **本地 Embedding/SBERT 模型**。所有模型无需手动下载，首次运行时会自动拉取。

---

## 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境（Windows）
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

---

## 2. 配置 .env（API Keys）

在项目根目录创建 `.env` 文件，填入对应平台的 API Key：

```env
# DeepSeek（必须，默认使用的 LLM）
DEEPSEEK_API_KEY=你的deepseek_key
DEEPSEEK_API_BASE=https://api.deepseek.com

# Google Gemini（可选）
GOOGLE_API_KEY=你的google_key

# MiniMax（可选）
MINIMAX_API_KEY=你的minimax_key
MINIMAX_API_BASE=https://api.minimax.chat/v1
MINIMAX_MODEL=模型名称

# 千问（阿里）（可选）
QWEN_API_KEY=你的qwen_key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 智谱 GLM（可选）
GLM_API_KEY=你的glm_key
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/

# 搜索引擎（可选，用于案例检索）
SERPER_API_KEY=你的serper_key
```

**最少只需配置 `DEEPSEEK_API_KEY`**，其余均为可选，不影响核心功能运行。

---

## 3. 模型的加载方式

### 3.1 远程 LLM（API 调用，无需本地模型文件）

所有 LLM 通过各自平台的 API 调用，**不占用本地显存/硬盘**：

| 类型 | 模型 | API 平台 | 调用方式 |
|------|------|----------|----------|
| default | deepseek-v4-pro | DeepSeek 官方 | OpenAI 兼容接口 |
| chat | deepseek-v4-flash | DeepSeek 官方 | OpenAI 兼容接口 |
| Gemini | gemini-3-flash-preview | Google | Google GenAI SDK |
| minimax | 由 .env 指定 | MiniMax | OpenAI 兼容接口 |
| glm | glm-5 | 智谱 AI | OpenAI 兼容接口 |
| qwen | qwen3-max | 阿里云百炼 | OpenAI 兼容接口 |

在代码中通过 `get_llm(type)` 切换模型：

```python
from llm import get_llm

# 默认 DeepSeek v4-pro
llm = get_llm()

# 切换到 Gemini
llm = get_llm("Gemini")

# 切换到 Claude Opus（需要先配好 CLAUDE_API_KEY 和 CLAUDE_API_BASE）
llm = get_llm("opus")
```

### 3.2 本地 SBERT 模型（自动下载）

项目中用于案例相似度计算的句子嵌入模型自动从 HuggingFace Hub 下载，模型名在 `config.py` 中定义：

```python
# config.py 第 46 行
SBERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
```

**首次运行时**，sentence-transformers 会自动将该模型下载到本地缓存目录：
- **Windows**: `C:\Users\<用户名>\.cache\huggingface\hub\`
- **Linux/Mac**: `~/.cache/huggingface/hub/`

项目已配置 HuggingFace 镜像（`config.py` 第 10 行），国内用户会自动走 `hf-mirror.com` 加速下载：

```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

**无需手动操作**，调 `SentenceTransformer(SBERT_MODEL_NAME)` 时会自动拉取。模型大小约 470MB，首次下载需等待 2-5 分钟。

---

## 4. models/ 目录说明

项目 `.gitignore` 中排除了 `models/` 文件夹。这个目录留给用户存放自定义的本地模型文件（如微调后的模型、私有模型等），不会被 Git 追踪。如果你没有自定义本地模型，可以忽略该目录。

---

## 5. 验证安装

在项目根目录运行以下命令验证所有模型是否能正确加载：

```python
# 验证 LLM API 连通性
from llm import get_llm

llm = get_llm()
response = llm.invoke("你好，请回复：模型加载成功")
print(response.content)

# 验证 SBERT 模型
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(["测试文本"])
print(f"Embedding 维度: {len(embeddings[0])}")  # 应输出 384
```

输出 `模型加载成功` 和 `Embedding 维度: 384` 即表示一切就绪。
