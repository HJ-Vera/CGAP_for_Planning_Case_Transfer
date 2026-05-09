"""
LLM 初始化 — 根据不同智能体返回对应的 LLM 实例
"""

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_API_BASE,
    GOOGLE_API_KEY,
    MINIMAX_API_KEY, MINIMAX_API_BASE, MINIMAX_MODEL,
    CLAUDE_API_KEY, CLAUDE_API_BASE,
    GLM_API_KEY, GLM_BASE_URL,
    QWEN_API_KEY, QWEN_BASE_URL,
)


def get_llm(type="default", max_tokens: int = 50000):
    """根据不同智能体返回对应的LLM"""

    if type == "Gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=GOOGLE_API_KEY,
            max_output_tokens=max_tokens,
            temperature=1
        )
    elif type == "minimax":   # 新增分支
        return ChatOpenAI(
            model=MINIMAX_MODEL,
            openai_api_key=MINIMAX_API_KEY,
            openai_api_base=MINIMAX_API_BASE,
            max_tokens=max_tokens,
            temperature=0.8
        )
    
    elif type == "chat":   # 新增分支
        return ChatOpenAI(
            model="deepseek-v4-flash",
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base=DEEPSEEK_API_BASE,
            max_tokens=20000,
            max_retries=3,
            request_timeout=300,  # 增加到5分钟
            reasoning_effort="max",
            extra_body={"thinking": {"type": "enabled"}}
        )
    
    elif type == "claude":
        return ChatOpenAI(
            model="[m1]claude-sonnet-4-6",    # 指定模型名称，注意格式
            openai_api_key=CLAUDE_API_KEY,       # 中转平台提供的 API Key
            openai_api_base=CLAUDE_API_BASE,               # 中转平台的 Base URL（如 https://api.jiekou.ai/anthropic）
            max_tokens=max_tokens,
            temperature=0.8,
            request_timeout=500,
            max_retries=2
        )
    
    elif type == "opus":
        return ChatOpenAI(
            model="[max]claude-opus-4-6-thinking",    # 指定模型名称，注意格式
            openai_api_key=CLAUDE_API_KEY,       # 中转平台提供的 API Key
            openai_api_base=CLAUDE_API_BASE,               # 中转平台的 Base URL（如 https://api.jiekou.ai/anthropic）
            max_tokens=max_tokens,
            temperature=0.8,
            request_timeout=500,
            max_retries=2
        )
    
    elif type == "glm":
        return ChatOpenAI(
            model="glm-5",    # 指定模型名称，注意格式
            openai_api_key=GLM_API_KEY,       # 中转平台提供的 API Key
            openai_api_base=GLM_BASE_URL,               # 中转平台的 Base URL（如 https://api.jiekou.ai/anthropic）
            max_tokens=max_tokens,
            temperature=0.8,
            request_timeout=500,
            max_retries=2
        )
    
    elif type == "qwen":
        return ChatOpenAI(
            model="qwen3-max",    # 指定模型名称，注意格式
            openai_api_key=QWEN_API_KEY,       # 中转平台提供的 API Key
            openai_api_base=QWEN_BASE_URL,               # 中转平台的 Base URL（如 https://api.jiekou.ai/anthropic）
            max_tokens=max_tokens,
            temperature=0.8,
            request_timeout=500,
            max_retries=2
        )


    else:
        # 默认返回原来的DeepSeek配置
        return ChatOpenAI(
            model="deepseek-v4-flash",
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base=DEEPSEEK_API_BASE,
            max_tokens=max_tokens,
            max_retries=3,
            request_timeout=300,   # 增加到5分钟
            reasoning_effort="max",
            extra_body={"thinking": {"type": "enabled"}}
        )
