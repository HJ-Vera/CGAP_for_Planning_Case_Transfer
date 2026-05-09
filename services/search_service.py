"""
搜索异步服务层 — aiohttp Serper / Semantic Scholar / ArXiv
统一 retry + timeout + Semaphore 控制并发
"""

import asyncio
import json
import random
import time
import urllib.parse
from typing import List, Dict
from xml.etree import ElementTree as ET

import aiohttp

from config import SERPER_API_KEY

SEARCH_CONCURRENCY = 3
_search_semaphore = asyncio.Semaphore(SEARCH_CONCURRENCY)


class SearchService:
    """异步搜索服务，封装 aiohttp 调用 + 重试 + 超时"""

    @staticmethod
    async def search_serper(query: str, max_results: int = 50, timeout: int = 30) -> List[Dict]:
        """异步 Serper 搜索"""
        all_results = []
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }

        for attempt in range(3):
            try:
                async with _search_semaphore:
                    requests_needed = min((max_results + 9) // 10, 10)
                    async with aiohttp.ClientSession() as session:
                        for page in range(requests_needed):
                            start_index = page * 10
                            payload = {
                                "q": query,
                                "num": min(10, max_results - len(all_results)),
                                "start": start_index
                            }
                            async with session.post(url, headers=headers, json=payload,
                                                    timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    organic_results = data.get("organic", [])
                                    if not organic_results:
                                        break
                                    for r in organic_results:
                                        all_results.append({
                                            "title": r.get("title", ""),
                                            "url": r.get("link", ""),
                                            "snippet": r.get("snippet", ""),
                                            "position": r.get("position", 0)
                                        })
                                    if len(organic_results) < 10:
                                        break
                                elif resp.status == 401:
                                    print(f"⚠️ Serper API 认证失败，请检查 API Key")
                                    return all_results
                                else:
                                    print(f"⚠️ Serper API 错误: {resp.status}")
                                    break

                            if len(all_results) >= max_results:
                                break
                            await asyncio.sleep(0.5)

                print(f"✅ Serper 搜索成功: 找到 {len(all_results)} 个结果")
                return all_results[:max_results]

            except asyncio.TimeoutError:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt < 2:
                    print(f"  ⚠️ Serper retry {attempt+1}: {e}")
                    await asyncio.sleep(2 ** attempt)

        print(f"⚠️ Serper 搜索失败，返回 {len(all_results)} 个结果")
        return all_results

    @staticmethod
    async def search_semantic_scholar(query: str, limit: int = 50, timeout: int = 60) -> List[Dict]:
        """异步 Semantic Scholar 搜索"""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "title,abstract,year,authors,url,citationCount,venue,publicationDate,isOpenAccess"
        }
        headers = {
            "User-Agent": random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            ]),
            "Accept": "application/json"
        }

        for attempt in range(4):
            try:
                async with _search_semaphore:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, params=params, headers=headers,
                                               timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                papers = data.get("data", [])
                                if papers:
                                    sorted_papers = sorted(papers, key=lambda x: x.get("citationCount", 0),
                                                           reverse=True)
                                    print(f"    ✅ 找到 {len(sorted_papers)} 篇学术文献")
                                    return sorted_papers[:limit]
                                return []
                            elif resp.status == 429:
                                wait_seconds = min(120, int(resp.headers.get('Retry-After', '300')))
                                print(f"    ⚠️ Semantic Scholar 速率限制，等待 {wait_seconds}s")
                                await asyncio.sleep(wait_seconds)
                            else:
                                print(f"    ⚠️ Semantic Scholar API 错误: {resp.status}")
                                return []
            except asyncio.TimeoutError:
                if attempt < 3:
                    await asyncio.sleep(min(2 ** attempt + 2, 60))
            except Exception as e:
                if attempt < 3:
                    await asyncio.sleep(min(2 ** attempt + 2, 60))

        return []

    @staticmethod
    async def search_google_scholar(query: str, max_results: int = 50, timeout: int = 30) -> List[Dict]:
        """异步 Google Scholar (via Serper)"""
        all_results = []
        url = "https://google.serper.dev/scholar"
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }

        for attempt in range(3):
            try:
                async with _search_semaphore:
                    requests_needed = min((max_results + 9) // 10, 5)
                    async with aiohttp.ClientSession() as session:
                        for page in range(requests_needed):
                            payload = {
                                "q": query,
                                "num": min(10, max_results - len(all_results)),
                                "start": page * 10
                            }
                            async with session.post(url, headers=headers, json=payload,
                                                    timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
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
                                    break
                            if len(all_results) >= max_results:
                                break
                            await asyncio.sleep(0.5)

                print(f"    ✅ Google Scholar 找到 {len(all_results)} 篇文献")
                return all_results[:max_results]
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        return all_results

    @staticmethod
    async def search_arxiv(query: str, max_results: int = 50, timeout: int = 30) -> List[Dict]:
        """异步 ArXiv 搜索"""
        try:
            base_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            query_string = urllib.parse.urlencode(params)
            url = f"{base_url}?{query_string}"

            async with _search_semaphore:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                        if resp.status == 200:
                            content = await resp.text()
                            root = ET.fromstring(content)
                            ns = {
                                'atom': 'http://www.w3.org/2005/Atom',
                                'arxiv': 'http://arxiv.org/schemas/atom'
                            }
                            results = []
                            for entry in root.findall('atom:entry', ns):
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
                        return []
        except Exception as e:
            print(f"    ⚠️ ArXiv 搜索失败: {str(e)[:100]}")
            return []

    @classmethod
    async def search_academic_sources(cls, query: str, limit: int = 50,
                                      skip_semantic_scholar: bool = False) -> List[Dict]:
        """异步综合学术搜索: SS + GS + ArXiv 并发"""
        print(f"  📚 开始学术文献搜索...")

        tasks = []
        if not skip_semantic_scholar:
            tasks.append(cls.search_semantic_scholar(query, limit=limit))
        tasks.append(cls.search_google_scholar(query, max_results=limit))
        tasks.append(cls.search_arxiv(query, max_results=limit))

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        for i, results in enumerate(results_list):
            if isinstance(results, Exception):
                print(f"    ⚠️ 搜索源 {i} 异常: {results}")
            elif results:
                all_results.extend(results)

        if all_results:
            print(f"  ✅ 学术搜索完成: 共 {len(all_results)} 篇文献")
        else:
            print(f"  ⚠️ 所有学术数据库均未返回结果")
        return all_results
