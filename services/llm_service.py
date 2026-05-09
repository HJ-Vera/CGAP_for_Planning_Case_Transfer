"""
LLM 异步服务层 — 统一 ainvoke + retry + timeout + 并发控制
"""

import asyncio
from functools import wraps

from llm import get_llm

LLM_CONCURRENCY = 2
_llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)


def _extract_content(response):
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


class LLMService:
    """异步 LLM 调用服务，封装 ainvoke + 重试 + 超时 + 并发控制"""

    @staticmethod
    async def ainvoke(llm_type="default", max_tokens=50000, messages=None, timeout=300):
        """
        异步调用 LLM，带重试+超时+并发控制。

        Args:
            llm_type: LLM 类型 ("default", "chat", "Gemini", "claude", "opus", "glm", "qwen", "minimax")
            max_tokens: 最大 token 数
            messages: LangChain 消息列表
            timeout: 超时秒数

        Returns:
            str: 提取后的文本内容
        """
        if messages is None:
            return ""

        llm = get_llm(type=llm_type, max_tokens=max_tokens)

        last_error = None
        for attempt in range(3):
            try:
                async with _llm_semaphore:
                    response = await asyncio.wait_for(
                        llm.ainvoke(messages),
                        timeout=timeout
                    )
                    return _extract_content(response)
            except asyncio.TimeoutError:
                last_error = f"LLM timeout after {timeout}s"
                if attempt < 2:
                    wait = 2 ** attempt
                    print(f"  ⏰ LLM timeout, retrying in {wait}s (attempt {attempt+1}/3)")
                    await asyncio.sleep(wait)
            except Exception as e:
                last_error = str(e)
                if attempt < 2:
                    wait = 2 ** attempt
                    print(f"  ⚠️ LLM error: {e}, retrying in {wait}s (attempt {attempt+1}/3)")
                    await asyncio.sleep(wait)

        print(f"  ❌ LLM call failed after 3 attempts: {last_error}")
        return ""

    @staticmethod
    async def ainvoke_raw(llm_type="default", max_tokens=50000, messages=None, timeout=300):
        """
        异步调用 LLM，返回原始响应对象（不提取 content）。
        用于需要直接访问 response.content 等属性的场景。
        """
        if messages is None:
            return None

        llm = get_llm(type=llm_type, max_tokens=max_tokens)

        last_error = None
        for attempt in range(3):
            try:
                async with _llm_semaphore:
                    response = await asyncio.wait_for(
                        llm.ainvoke(messages),
                        timeout=timeout
                    )
                    return response
            except asyncio.TimeoutError:
                last_error = f"LLM timeout after {timeout}s"
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                last_error = str(e)
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        print(f"  ❌ LLM raw call failed after 3 attempts: {last_error}")
        return None
