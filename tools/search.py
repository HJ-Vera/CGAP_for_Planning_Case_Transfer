"""
搜索工具 — Serper、Semantic Scholar、Google Scholar、ArXiv
"""

import time
import random
import json
import urllib.parse
from typing import List, Dict
from xml.etree import ElementTree as ET

import requests
from langsmith import traceable

from config import SERPER_API_KEY


@traceable(name="search_serper", run_type="tool")
def search_serper(query: str, max_results: int = 50) -> List[Dict]:
    """
    使用 Serper API 进行网页搜索

    注意：Serper API 每次请求最多返回 10 个结果（免费计划限制）
    如需更多结果，需要使用分页参数或升级订阅计划

    参数说明：
    - num: 请求的结果数量（API 可能限制为 10）
    - start: 分页起始位置（用于获取更多结果）
    """
    all_results = []

    try:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }

        # Serper API 限制每次最多返回 10 个结果
        # 使用分页获取更多结果
        requests_needed = (max_results + 9) // 10  # 向上取整
        requests_needed = min(requests_needed, 10)  # 最多 10 次请求，避免过度调用

        for page in range(requests_needed):
            start_index = page * 10

            payload = {
                "q": query,
                "num": min(10, max_results - len(all_results)),  # 每次请求 10 个
                "start": start_index  # 分页起始位置
            }

            # print(f"    📄 Serper 请求第 {page + 1} 页 (start={start_index})...")

            response = requests.post(url, headers=headers, json=payload, timeout=15)

            if response.status_code == 200:
                data = response.json()

                # 提取有机搜索结果
                organic_results = data.get("organic", [])
                if not organic_results:
                    break  # 没有更多结果了

                for r in organic_results:
                    all_results.append({
                        "title": r.get("title", ""),
                        "url": r.get("link", ""),
                        "snippet": r.get("snippet", ""),
                        "position": r.get("position", 0)
                    })

                # 如果返回结果少于 10 个，说明没有更多结果了
                if len(organic_results) < 10:
                    break

            elif response.status_code == 401:
                print(f"⚠️ Serper API 认证失败，请检查 API Key")
                break
            else:
                print(f"⚠️ Serper API 错误: {response.status_code} - {response.text[:100]}")
                break

            # 避免请求过快
            if page < requests_needed - 1:
                time.sleep(0.5)

            # 已获取足够结果
            if len(all_results) >= max_results:
                break

        print(f"✅ Serper 搜索成功: 找到 {len(all_results)} 个结果")
        return all_results[:max_results]

    except Exception as e:
        print(f"⚠️ Serper 搜索失败: {e}")
        return all_results  # 返回已获取的结果


# ========================== 速率限制器 ==========================
class RateLimiter:
    """简单的速率限制器，控制请求间隔"""
    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self.last_request_time = 0

    def wait(self):
        """等待直到可以发起下一次请求"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()


# 全局速率限制器实例
# Semantic Scholar 未认证: 每 5 分钟 100 次 ≈ 每 3 秒 1 次
_semantic_scholar_limiter = RateLimiter(min_interval=3.5)


@traceable(name="search_semantic_scholar", run_type="tool")
def search_semantic_scholar(
    query: str,
    limit: int = 50,
    api_key: str = None,
    max_retries: int = 3
) -> List[Dict]:
    """
    使用 Semantic Scholar API 搜索学术文献
    带速率限制处理和重试机制

    速率限制说明：
    - 未认证用户：每 5 分钟 100 次请求 (约 1 次/3秒)
    - 认证用户：每 5 分钟 5000 次请求 (API Key)

    建议在 config.py 中设置 SEMANTIC_SCHOLAR_API_KEY 以获得更高限额
    """
    # 验证参数
    if limit < 1 or limit > 100:
        limit = 20  # 使用默认值

    # API配置
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,authors,url,citationCount,venue,publicationDate,isOpenAccess"
    }

    # 随机User-Agent
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ]

    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": "application/json"
    }

    # API Key 认证（如果有）
    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(max_retries + 1):
        try:
            # 使用全局速率限制器
            _semantic_scholar_limiter.wait()

            if attempt > 0:
                # 重试时使用指数退避 + 随机抖动
                base_delay = 2 ** attempt
                jitter = random.uniform(0.5, 1.5)
                delay = base_delay * jitter + 2  # 额外 +2 秒确保安全
                print(f"    ⏰ 等待 {delay:.1f} 秒后重试 (第 {attempt} 次)...")
                time.sleep(delay)

            print(f"    🔍 Semantic Scholar 搜索: '{query[:50]}{'...' if len(query) > 50 else ''}'")

            response = requests.get(url, params=params, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                papers = data.get("data", [])

                if papers:
                    sorted_papers = sorted(
                        papers,
                        key=lambda x: x.get("citationCount", 0),
                        reverse=True
                    )
                    print(f"    ✅ 找到 {len(sorted_papers)} 篇学术文献")
                    return sorted_papers[:limit]
                else:
                    print(f"    ⚠️ 未找到相关文献")
                    return []

            elif response.status_code == 429:
                # 速率限制 - 未认证用户每 5 分钟 100 次
                retry_after = response.headers.get('Retry-After', '300')  # 默认等 5 分钟
                try:
                    wait_seconds = int(retry_after)
                except ValueError:
                    wait_seconds = 300  # 默认等待 5 分钟

                if attempt < max_retries:
                    # 限制等待时间，最多 2 分钟
                    wait_seconds = min(wait_seconds, 120)
                    print(f"    ⚠️ 达到速率限制 (429)，等待 {wait_seconds} 秒后重试...")
                    time.sleep(wait_seconds)
                    continue
                else:
                    print(f"    ❌ Semantic Scholar 速率限制，已尝试 {max_retries} 次重试")
                    print(f"    💡 未认证用户限制：每 5 分钟 100 次请求")
                    print(f"    💡 建议：稍后再试，或跳过 Semantic Scholar 使用其他数据源")
                    return []

            elif response.status_code == 400:
                print(f"    ⚠️ 请求参数错误: {response.text[:100]}")
                return []

            elif response.status_code == 403:
                print(f"    ⚠️ 访问被拒绝 (403)")
                return []

            else:
                print(f"    ⚠️ API错误 [{response.status_code}]: {response.text[:100]}")
                return []

        except requests.Timeout:
            print(f"    ⚠️ 请求超时 (30s)")
            if attempt < max_retries:
                continue
            return []

        except requests.RequestException as e:
            print(f"    ⚠️ 网络错误: {str(e)[:80]}")
            if attempt < max_retries:
                continue
            return []

        except json.JSONDecodeError:
            print(f"    ⚠️ 响应解析失败")
            return []

        except Exception as e:
            print(f"    ⚠️ 未知错误: {str(e)[:80]}")
            return []

    return []


@traceable(name="search_google_scholar", run_type="tool")
def search_google_scholar_alternative(query: str, max_results: int = 50) -> List[Dict]:
    """
    备用学术搜索：使用 Serper API 搜索 Google Scholar
    当 Semantic Scholar 不可用时使用

    注意：Serper Scholar API 同样有分页限制
    """
    all_results = []

    try:
        url = "https://google.serper.dev/scholar"
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }

        # 分页获取结果
        requests_needed = (max_results + 9) // 10
        requests_needed = min(requests_needed, 5)  # Scholar 限制更严格

        for page in range(requests_needed):
            start_index = page * 10

            payload = {
                "q": query,
                "num": min(10, max_results - len(all_results)),
                "start": start_index
            }

            print(f"    🔍 Google Scholar 搜索 (备用): '{query[:50]}...' 第 {page + 1} 页")

            response = requests.post(url, headers=headers, json=payload, timeout=15)

            if response.status_code == 200:
                data = response.json()

                organic_results = data.get("organic", [])
                if not organic_results:
                    break

                for r in organic_results:
                    all_results.append({
                        "title": r.get("title", ""),
                        "url": r.get("link", ""),
                        "abstract": r.get("snippet", ""),
                        "year": "N/A",
                        "citationCount": 0,
                        "source": "Google Scholar"
                    })

                if len(organic_results) < 10:
                    break
            else:
                print(f"    ⚠️ Google Scholar API 错误: {response.status_code}")
                break

            if page < requests_needed - 1:
                time.sleep(0.5)

            if len(all_results) >= max_results:
                break

        print(f"    ✅ Google Scholar 找到 {len(all_results)} 篇文献")
        return all_results[:max_results]

    except Exception as e:
        print(f"    ⚠️ Google Scholar 搜索失败: {str(e)[:100]}")
        return all_results


@traceable(name="search_arxiv", run_type="tool")
def search_arxiv(query: str, max_results: int = 50) -> List[Dict]:
    """
    备用学术搜索：ArXiv API
    专注于计算机科学、数学、物理等领域的预印本
    """
    try:
        base_url = "http://export.arxiv.org/api/query"

        search_query = f"all:{query}"
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        query_string = urllib.parse.urlencode(params)
        url = f"{base_url}?{query_string}"

        print(f"    🔍 ArXiv 搜索: '{query[:50]}...'")

        response = requests.get(url, timeout=15)

        if response.status_code == 200:
            root = ET.fromstring(response.content)

            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }

            results = []
            entries = root.findall('atom:entry', ns)

            for entry in entries:
                title = entry.find('atom:title', ns)
                summary = entry.find('atom:summary', ns)
                link = entry.find('atom:id', ns)
                published = entry.find('atom:published', ns)

                results.append({
                    "title": title.text.strip() if title is not None else "",
                    "abstract": summary.text.strip() if summary is not None else "",
                    "url": link.text.strip() if link is not None else "",
                    "year": published.text[:4] if published is not None else "N/A",
                    "citationCount": 0,
                    "source": "ArXiv"
                })

            print(f"    ✅ ArXiv 找到 {len(results)} 篇预印本")
            return results
        else:
            print(f"    ⚠️ ArXiv API 错误: {response.status_code}")
            return []

    except Exception as e:
        print(f"    ⚠️ ArXiv 搜索失败: {str(e)[:100]}")
        return []


@traceable(name="search_academic_sources", run_type="tool")
def search_academic_sources(query: str, limit: int = 50, skip_semantic_scholar: bool = False) -> List[Dict]:
    """
    综合学术搜索：依次尝试多个学术数据库
    1. Semantic Scholar (首选，但可能被限流)
    2. Google Scholar via Serper (备用1，推荐)
    3. ArXiv (备用2，适合理工科)

    参数：
        skip_semantic_scholar: 是否跳过 Semantic Scholar（避免速率限制）
    """
    print(f"  📚 开始学术文献搜索...")

    all_results = []

    # 尝试 Semantic Scholar（可选择跳过）
    if not skip_semantic_scholar:
        ss_results = search_semantic_scholar(query, limit=limit)
        if ss_results:
            all_results.extend(ss_results)
            print(f"    ✅ Semantic Scholar 贡献 {len(ss_results)} 篇")
    else:
        print(f"    ⏭️ 跳过 Semantic Scholar（避免速率限制）")

    # 如果结果不足，尝试 Google Scholar
    if len(all_results) < limit:
        remaining = limit - len(all_results)
        gs_results = search_google_scholar_alternative(query, max_results=remaining)
        if gs_results:
            all_results.extend(gs_results)
            print(f"    ✅ Google Scholar 贡献 {len(gs_results)} 篇")

    # 如果仍不足，尝试 ArXiv
    if len(all_results) < limit and any(keyword in query.lower() for keyword in ['technology', 'system', 'algorithm', 'data', 'smart', 'planning', 'urban']):
        remaining = limit - len(all_results)
        arxiv_results = search_arxiv(query, max_results=remaining)
        if arxiv_results:
            all_results.extend(arxiv_results)
            print(f"    ✅ ArXiv 贡献 {len(arxiv_results)} 篇")

    if all_results:
        print(f"  ✅ 学术搜索完成: 共 {len(all_results)} 篇文献")
    else:
        print(f"  ⚠️ 所有学术数据库均未返回结果")

    return all_results
