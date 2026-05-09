"""
混合检索模块 — BM25 + Sentence-BERT
"""

import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np

from config import BM25_CANDIDATE_K, SBERT_MODEL_NAME


# ==============================================================
# 工具函数
# ==============================================================

def _deduplicate(results: List[Dict]) -> List[Dict]:
    """按 (title, url) 去重，保留第一条"""
    seen = set()
    out = []
    for r in results:
        key = (r.get("title", "").strip().lower(), r.get("url", "").strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _tokenize(text: str, language: str = "en") -> List[str]:
    r"""
    统一分词入口
      - EN: 正则 \w+ + 停用词去除
      - ZH: jieba 分词 + 停用词去除
    """
    if isinstance(text, list):
        text = ' '.join(str(item) for item in text)
    if not text:
        return []

    en_stop = {
        'the', 'a', 'an', 'in', 'on', 'at', 'for', 'of', 'and', 'or',
        'but', 'is', 'are', 'was', 'were', 'be', 'been', 'to', 'that',
        'this', 'with', 'as', 'by', 'from', 'it', 'its', 'they', 'their',
        'have', 'has', 'had', 'not', 'no', 'do', 'does', 'did', 'will',
        'would', 'can', 'could', 'may', 'might', 'should', 'shall'
    }
    zh_stop = {
        '的', '了', '和', '在', '是', '一', '个', '了', '有', '不',
        '也', '都', '这', '上', '于', '与', '中', '为', '由', '其'
    }

    if language == "zh":
        try:
            import jieba
            tokens = list(jieba.cut(text))
            stop = en_stop | zh_stop
            return [t.strip() for t in tokens
                    if t.strip() and t.strip() not in stop and len(t.strip()) > 0
                    and not re.fullmatch(r'[^\w]+', t.strip())]
        except ImportError:
            chars = list(text.replace(' ', ''))
            return [c for c in chars if c not in zh_stop and re.match(r'\w', c)]
    else:
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in en_stop and len(w) > 1]


# ==============================================================
# BM25Scorer
# ==============================================================

class BM25Scorer:
    """标准 BM25 实现"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self._doc_count: int = 0
        self._avg_doc_len: float = 0.0
        self._doc_freqs: Dict[str, int] = defaultdict(int)
        self._doc_word_counts: List[Counter] = []
        self._doc_lengths: List[int] = []
        self._doc_languages: List[str] = []

    def fit(self, doc_texts: List[str], doc_languages: Optional[List[str]] = None):
        """训练: 遍历所有文档，计算统计量"""
        self._doc_count = len(doc_texts)
        self._doc_languages = doc_languages or ["en"] * self._doc_count
        self._doc_word_counts = []
        self._doc_lengths = []
        self._doc_freqs = defaultdict(int)

        for i, text in enumerate(doc_texts):
            lang = self._doc_languages[i]
            words = _tokenize(text, lang)
            wc = Counter(words)

            self._doc_word_counts.append(wc)
            self._doc_lengths.append(len(words))

            for term in wc:
                self._doc_freqs[term] += 1

        self._avg_doc_len = (
            sum(self._doc_lengths) / self._doc_count
            if self._doc_count > 0 else 1.0
        )

    def _idf(self, term: str) -> float:
        """标准 BM25 IDF 公式"""
        df = self._doc_freqs.get(term, 0)
        return math.log(
            (self._doc_count - df + 0.5) / (df + 0.5) + 1
        )

    def score(self, queries: Dict[str, str], doc_index: int) -> float:
        """对单个已训练文档打分"""
        if doc_index < 0 or doc_index >= self._doc_count:
            return 0.0

        doc_wc = self._doc_word_counts[doc_index]
        doc_len = self._doc_lengths[doc_index]
        if doc_len == 0:
            return 0.0

        doc_lang = self._doc_languages[doc_index]
        query_text = queries.get(doc_lang, queries.get("en", ""))
        if not query_text:
            return 0.0

        query_terms = _tokenize(query_text, doc_lang)
        if not query_terms:
            return 0.0

        score = 0.0
        for term in query_terms:
            tf = doc_wc.get(term, 0)
            if tf == 0:
                continue

            idf = self._idf(term)

            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * (doc_len / self._avg_doc_len))
            )
            score += idf * tf_norm

        return score

    def score_all(self, queries: Dict[str, str]) -> List[float]:
        """对所有训练文档批量打分"""
        return [
            self.score(queries, i)
            for i in range(self._doc_count)
        ]


# ==============================================================
# SBERTScorer
# ==============================================================

class SBERTScorer:
    """封装 sentence-transformers 模型"""

    def __init__(self, model_name: str = SBERT_MODEL_NAME):
        import os
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
        from sentence_transformers import SentenceTransformer
        print(f"  🤖 加载 Sentence-BERT 模型: {model_name}...")
        self.model = SentenceTransformer("./models/paraphrase-multilingual-MiniLM-L12-v2")
        print("  ✅ 模型加载完成")

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码为向量"""
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)

    def cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """计算余弦相似度"""
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        d_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
        return d_norms @ q_norm


# ==============================================================
# HybridRetriever
# ==============================================================

class HybridRetriever:
    """BM25 + SBERT RRF (Reciprocal Rank Fusion) 混合检索器"""

    def __init__(self, rrf_k: int = 60, sbert_model_name: str = SBERT_MODEL_NAME):
        self.rrf_k = rrf_k
        self.bm25 = BM25Scorer()
        self.sbert = SBERTScorer(sbert_model_name)

        self._doc_texts: List[str] = []
        self._doc_languages: List[str] = []
        self._sbert_doc_vecs: Optional[np.ndarray] = None

    def fit(self, doc_texts: List[str], doc_languages: Optional[List[str]] = None):
        """训练 BM25 + 预计算 SBERT 文档向量"""
        self._doc_texts = doc_texts
        self._doc_languages = doc_languages or ["en"] * len(doc_texts)

        print("    📊 BM25 训练中...")
        self.bm25.fit(doc_texts, self._doc_languages)

        print("    🤖 SBERT 编码文档中...")
        self._sbert_doc_vecs = self.sbert.encode(doc_texts)
        print(f"    ✅ 编码完成: {self._sbert_doc_vecs.shape}")

    def retrieve(
        self,
        queries: Dict[str, str],
        top_k: int = BM25_CANDIDATE_K
    ) -> List[Tuple[int, float, int, int]]:
        """
        RRF 融合检索，返回: [(doc_index, rrf_score, bm25_rank, sbert_rank), ...]
        RRF 公式: score(d) = 1/(k + bm25_rank) + 1/(k + sbert_rank)
        """
        if self._sbert_doc_vecs is None:
            raise RuntimeError("未调用 fit()，无法检索")

        n_docs = len(self._doc_texts)

        # ── BM25 排名 ──
        bm25_raw = np.array(self.bm25.score_all(queries))
        bm25_ranking = np.argsort(-bm25_raw)  # 按分数降序
        bm25_rank_map = {int(idx): rank + 1 for rank, idx in enumerate(bm25_ranking)}

        # ── SBERT 相似度排名 ──
        query_vecs: Dict[str, np.ndarray] = {}
        for lang, q_text in queries.items():
            query_vecs[lang] = self.sbert.encode([q_text])[0]

        fallback_lang = "en"
        sbert_sims = np.zeros(n_docs)
        for i in range(n_docs):
            doc_lang = self._doc_languages[i]
            q_vec = query_vecs.get(doc_lang, query_vecs.get(fallback_lang))
            if q_vec is None:
                continue
            sbert_sims[i] = self.sbert.cosine_similarity(
                q_vec, self._sbert_doc_vecs[i:i+1]
            )[0]

        sbert_ranking = np.argsort(-sbert_sims)
        sbert_rank_map = {int(idx): rank + 1 for rank, idx in enumerate(sbert_ranking)}

        # ── RRF 融合 ──
        rrf_scores = np.zeros(n_docs)
        for i in range(n_docs):
            rrf_scores[i] = (1.0 / (self.rrf_k + bm25_rank_map[i])
                           + 1.0 / (self.rrf_k + sbert_rank_map[i]))

        top_indices = np.argsort(-rrf_scores)[:top_k]

        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(rrf_scores[idx]),
                bm25_rank_map[int(idx)],
                sbert_rank_map[int(idx)],
            ))
        return results
