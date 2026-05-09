"""
网页抓取异步服务层 — aiohttp + 并发控制
PDF 提取保持同步（CPU-bound）
"""

import asyncio
import io
import re
import logging

import aiohttp
from bs4 import BeautifulSoup

FETCH_CONCURRENCY = 5
_fetch_semaphore = asyncio.Semaphore(FETCH_CONCURRENCY)

logger = logging.getLogger(__name__)


async def async_fetch_webpage_content(url: str, max_length: int = 20000, timeout: int = 30) -> str:
    """
    异步抓取网页内容（支持 HTML 和 PDF）。

    Args:
        url: 网页 URL
        max_length: 最大返回字符数
        timeout: 请求超时

    Returns:
        str: 提取的文本内容
    """
    for attempt in range(3):
        try:
            async with _fetch_semaphore:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers,
                                           timeout=aiohttp.ClientTimeout(total=timeout),
                                           allow_redirects=True) as response:
                        content = await response.read()
                        content_type = response.headers.get("Content-Type", "").lower()

                        is_pdf = ("pdf" in content_type) or (content[:4] == b"%PDF")

                        if is_pdf:
                            return await _extract_pdf_text(content, max_length)

                        html_content = None
                        for encoding in _get_encodings(response.charset, content):
                            try:
                                html_content = content.decode(encoding, errors='strict')
                                break
                            except (UnicodeDecodeError, LookupError):
                                continue

                        if html_content is None:
                            html_content = content.decode('utf-8', errors='replace')

                        text = _parse_html(html_content)
                        return _clean_and_truncate(text, max_length)

        except asyncio.TimeoutError:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                return f"网页访问超时: {url}"
        except aiohttp.ClientError as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                return f"网页访问失败: {str(e)[:100]}"
        except Exception as e:
            return f"内容解析失败: {str(e)[:100]}"


async def async_fetch_webpage_content_alternative(url: str, max_length: int = 10000,
                                                   timeout: int = 30) -> str:
    """备用异步抓取（使用 lxml 解析器，更快）"""
    try:
        async with _fetch_semaphore:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers,
                                       timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    content = await response.read()

                    try:
                        from lxml import html as lxml_html
                        tree = lxml_html.fromstring(content)
                        text = tree.text_content()
                    except ImportError:
                        soup = BeautifulSoup(content, 'html.parser')
                        for script in soup(["script", "style"]):
                            script.decompose()
                        text = soup.get_text()

                    text = re.sub(r'\s+', ' ', text.strip())
                    if len(text) > max_length:
                        text = text[:max_length] + "..."
                    return text
    except Exception as e:
        return f"备用方法也失败: {str(e)[:100]}"


async def _extract_pdf_text(pdf_bytes: bytes, max_length: int) -> str:
    """PDF 文本提取（同步 CPU-bound 操作，在线程池中执行）"""
    from tools.web_fetcher import extract_pdf_text
    import concurrent.futures as cf
    loop = asyncio.get_running_loop()
    with cf.ThreadPoolExecutor(max_workers=1) as pool:
        return await loop.run_in_executor(pool, extract_pdf_text, pdf_bytes, max_length)


def _get_encodings(charset, content):
    """获取编码探测列表"""
    encodings = [charset] if charset else []
    try:
        import chardet
        result = chardet.detect(content)
        if result['confidence'] > 0.7:
            encodings.append(result['encoding'])
    except ImportError:
        pass
    encodings.extend(['utf-8', 'gb18030', 'gbk', 'big5', 'big5hkscs',
                      'shift_jis', 'euc-kr', 'iso-8859-1', 'windows-1252'])
    seen = set()
    return [e for e in encodings if e and not (e in seen or seen.add(e))]


def _parse_html(html_content: str) -> str:
    """解析 HTML 提取文本"""
    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(["script", "style", "meta", "link", "noscript"]):
        script.decompose()
    text = soup.get_text(separator=' ', strip=True)
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    return text


def _clean_and_truncate(text: str, max_length: int) -> str:
    """清理并截断文本"""
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > max_length:
        text = text[:max_length] + "..."
    return text
