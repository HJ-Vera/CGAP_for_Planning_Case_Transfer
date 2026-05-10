"""Content fingerprint verification — detects A/B swap in LLM judge reasoning.

Post-hoc check: extract discriminative keywords from each report, then verify
that reasoning_chain for Option A overlaps more with content A than content B
(and vice versa). Flags potential confusion; never auto-corrects.
"""

import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class ContentVerificationResult:
    a_reasoning_a_overlap: float = 0.0
    a_reasoning_b_overlap: float = 0.0
    b_reasoning_b_overlap: float = 0.0
    b_reasoning_a_overlap: float = 0.0
    n_discriminative_a: int = 0
    n_discriminative_b: int = 0
    a_match_terms_a: list[str] = field(default_factory=list)
    a_match_terms_b: list[str] = field(default_factory=list)
    b_match_terms_b: list[str] = field(default_factory=list)
    b_match_terms_a: list[str] = field(default_factory=list)
    swap_flagged: bool = False
    swap_direction: str = "none"
    swap_confidence: float = 0.0
    insufficient_terms: bool = False


DOMAIN_STOPWORDS: set[str] = {
    "规划", "发展", "建议", "香港", "方案", "项目", "地区", "用地",
    "发展区", "公顷", "平方公里", "平方米", "策略", "研究", "分析",
    "评估", "目标", "措施", "城市规划", "土地利用", "发展策略",
    "基础设施", "交通", "社区", "环境", "公共", "政府", "政策",
    "管理", "实施", "建设", "改造", "更新", "开发", "提升", "优化",
    "加强", "推进", "促进", "支持", "确保", "包括", "以及", "通过",
    "需要", "进行", "提供", "建立", "方面", "问题", "报告", "参考",
    "案例", "经验", "模式", "框架", "内容", "结果", "影响",
    "原则", "结构", "方法", "体系", "制度", "机制", "标准", "指标",
}

ZH_STOP: set[str] = {
    "的", "了", "和", "在", "是", "一", "个", "有", "不", "也",
    "都", "这", "上", "于", "与", "中", "为", "由", "其", "及",
    "等", "可", "将", "被", "从", "而", "对", "使", "之", "或",
    "该", "每", "各", "所", "以", "如", "但", "而", "又", "且",
}

EN_STOP: set[str] = {
    "the", "a", "an", "in", "on", "at", "for", "of", "and", "or",
    "but", "is", "are", "was", "were", "be", "been", "to", "that",
    "this", "with", "as", "by", "from", "it", "its", "they", "their",
    "have", "has", "had", "not", "no", "do", "does", "did", "will",
    "would", "can", "could", "may", "might", "should", "shall",
}

ALL_STOP = ZH_STOP | EN_STOP | DOMAIN_STOPWORDS

MIN_TERM_LEN = 2
TOP_N_TERMS = 50
MIN_DISCRIMINATIVE = 10


def _tokenize(text: str) -> list[str]:
    try:
        import jieba
    except ImportError:
        return [t for t in re.findall(r"\w+", text) if len(t) >= MIN_TERM_LEN]
    tokens = list(jieba.cut(text))
    return [
        t.strip() for t in tokens
        if t.strip()
        and t.strip() not in ALL_STOP
        and len(t.strip()) >= MIN_TERM_LEN
        and not re.fullmatch(r"[^\w]+", t.strip())
    ]


class ContentVerifier:
    """Verify LLM judge reasoning matches the correct content."""

    def verify(
        self,
        content_a: str,
        content_b: str,
        reasoning_chain: dict,
        reasoning: dict | None = None,
    ) -> ContentVerificationResult:
        terms_a, terms_b = self._extract_discriminative_terms(content_a, content_b)

        res = ContentVerificationResult(
            n_discriminative_a=len(terms_a),
            n_discriminative_b=len(terms_b),
        )

        if len(terms_a) < MIN_DISCRIMINATIVE or len(terms_b) < MIN_DISCRIMINATIVE:
            res.insufficient_terms = True
            return res

        reasoning_a_text = self._flatten_reasoning(
            reasoning_chain.get("A", {}), reasoning, "A"
        )
        reasoning_b_text = self._flatten_reasoning(
            reasoning_chain.get("B", {}), reasoning, "B"
        )

        res.a_reasoning_a_overlap, res.a_match_terms_a = self._compute_overlap(terms_a, reasoning_a_text)
        res.a_reasoning_b_overlap, res.a_match_terms_b = self._compute_overlap(terms_b, reasoning_a_text)
        res.b_reasoning_b_overlap, res.b_match_terms_b = self._compute_overlap(terms_b, reasoning_b_text)
        res.b_reasoning_a_overlap, res.b_match_terms_a = self._compute_overlap(terms_a, reasoning_b_text)

        a_swapped = res.a_reasoning_b_overlap > res.a_reasoning_a_overlap
        b_swapped = res.b_reasoning_a_overlap > res.b_reasoning_b_overlap

        if a_swapped or b_swapped:
            res.swap_flagged = True
            res.swap_confidence = round(
                max(
                    res.a_reasoning_b_overlap - res.a_reasoning_a_overlap,
                    res.b_reasoning_a_overlap - res.b_reasoning_b_overlap,
                ),
                4,
            )
            if a_swapped and b_swapped:
                res.swap_direction = "both"
            elif a_swapped:
                res.swap_direction = "A->B"
            else:
                res.swap_direction = "B->A"
        else:
            res.swap_confidence = round(
                min(
                    res.a_reasoning_a_overlap - res.a_reasoning_b_overlap,
                    res.b_reasoning_b_overlap - res.b_reasoning_a_overlap,
                ),
                4,
            )

        return res

    def _extract_discriminative_terms(
        self, content_a: str, content_b: str
    ) -> tuple[list[str], list[str]]:
        tokens_a = _tokenize(content_a)
        tokens_b = _tokenize(content_b)
        counter_a = Counter(tokens_a)
        counter_b = Counter(tokens_b)

        disc_a: list[tuple[str, float]] = []
        for t, cnt in counter_a.items():
            score = cnt / (counter_b.get(t, 0) + 1)
            if score > 1.0:
                disc_a.append((t, score))
        disc_a.sort(key=lambda x: x[1], reverse=True)

        disc_b: list[tuple[str, float]] = []
        for t, cnt in counter_b.items():
            score = cnt / (counter_a.get(t, 0) + 1)
            if score > 1.0:
                disc_b.append((t, score))
        disc_b.sort(key=lambda x: x[1], reverse=True)

        return [t for t, _ in disc_a[:TOP_N_TERMS]], [t for t, _ in disc_b[:TOP_N_TERMS]]

    def _flatten_reasoning(
        self,
        chain: dict,
        reasoning: dict | None = None,
        option: str = "A",
    ) -> str:
        parts: list[str] = []

        def _collect(obj):
            if isinstance(obj, str):
                parts.append(obj)
            elif isinstance(obj, list):
                for item in obj:
                    _collect(item)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _collect(v)

        _collect(chain)

        if reasoning:
            for key in (f"{option}_strengths", f"{option}_weaknesses"):
                val = reasoning.get(key, "")
                if val:
                    parts.append(val)

        return " ".join(parts)

    def _compute_overlap(
        self, discriminative_terms: list[str], reasoning_text: str
    ) -> tuple[float, list[str]]:
        if not discriminative_terms:
            return 0.0, []
        reasoning_tokens = set(_tokenize(reasoning_text))
        matched = [t for t in discriminative_terms if t in reasoning_tokens]
        return len(matched) / len(discriminative_terms), matched
