"""Pairwise runner: executes A vs B comparison using objective + subjective skills."""

import json
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from ..judges._base_judge import BaseJudge
from ..parser.pairwise_parser import PairwiseParser
from ..policies.aggregation_policy import aggregate
from ..policies.content_verification import ContentVerifier
from ..policies.veto_policy import check_d8_no_override_d5


class PairwiseRunner:
    """Runs a full pairwise (A vs B) evaluation using two skills and a judge."""

    SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

    def __init__(self, judge: BaseJudge, verbose: bool = True):
        self.judge = judge
        self.parser = PairwiseParser()
        self.verbose = verbose
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(self.SKILLS_DIR)),
            autoescape=False,
        )

    def _log(self, msg: str, end: str = "\n"):
        if self.verbose:
            print(msg, end=end, flush=True)

    def _load_skill_config(self, skill_name: str) -> dict:
        config_path = self.SKILLS_DIR / skill_name / "skill.yaml"
        self._log(f"  📋 加载 Skill 配置: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        dims = [d["id"] for d in cfg["dimensions"]]
        self._log(f"     维度: {', '.join(dims)}")
        return cfg

    def _render_prompt(self, skill_name: str, context: dict) -> str:
        self._log(f"  🔧 渲染 Jinja2 模板: {skill_name}/prompt.jinja2 ...")
        template = self._jinja_env.get_template(f"{skill_name}/prompt.jinja2")
        result = template.render(**context)
        self._log(f"     渲染后长度: {len(result)} chars")
        return result

    def _run_skill(self, skill_name: str, query_id: str, query: str,
                   option_a: str, option_b: str) -> dict:
        self._log(f"\n{'─' * 50}")
        self._log(f"🔍 [{skill_name.upper()}] 开始评估 {query_id}")
        self._log(f"{'─' * 50}")

        skill_config = self._load_skill_config(skill_name)
        context = {
            "query_id": query_id,
            "planning_query": query,
            "option_a_content": option_a,
            "option_b_content": option_b,
            "dimensions": skill_config["dimensions"],
        }
        prompt = self._render_prompt(skill_name, context)

        self._log(f"  🤖 调用 Judge: {self.judge.model} ...")
        t0 = time.time()
        raw_output = self.judge.evaluate(prompt, response_schema={"type": "json_object"})
        elapsed = time.time() - t0
        self._log(f"     API 耗时: {elapsed:.0f}s, 输出长度: {len(raw_output)} chars")

        try:
            parsed = self.parser.parse(raw_output)
            scores_a = parsed.get("scores", {}).get("A", {})
            scores_b = parsed.get("scores", {}).get("B", {})
            self._log(f"  ✅ 解析成功")
            self._log(f"     A 分数: {json.dumps(scores_a, ensure_ascii=False)}")
            self._log(f"     B 分数: {json.dumps(scores_b, ensure_ascii=False)}")
            self._log(f"     overall_preference: {parsed.get('overall_preference', 'N/A')}")
            flags = parsed.get("hong_kong_specific_flags", [])
            if flags:
                self._log(f"     HK flags: {len(flags)} 条")
            return parsed
        except Exception:
            dump_path = os.path.join(
                os.environ.get("TEMP", "/tmp"),
                f"eval_debug_{skill_name}_{query_id}.txt"
            )
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write(f"=== PROMPT (len={len(prompt)}) ===\n{prompt}\n\n=== OUTPUT (len={len(raw_output)}) ===\n{raw_output}")
            self._log(f"  ❌ 解析失败! 完整 dump: {dump_path}")
            raise

    def run(self, query_id: str, query: str,
            option_a_content: str, option_b_content: str) -> dict:
        """Execute a full pairwise evaluation."""

        self._log(f"\n{'═' * 60}")
        self._log(f"📊 开始逐对评估: {query_id}")
        self._log(f"   查询: {query[:60]}...")
        self._log(f"   Option A 长度: {len(option_a_content)} chars")
        self._log(f"   Option B 长度: {len(option_b_content)} chars")
        self._log(f"{'═' * 60}")

        t0 = time.time()

        obj_result = self._run_skill(
            "objective", query_id, query, option_a_content, option_b_content
        )
        subj_result = self._run_skill(
            "subjective", query_id, query, option_a_content, option_b_content
        )

        self._log(f"\n{'─' * 50}")
        self._log(f"📐 聚合结果")
        self._log(f"{'─' * 50}")

        obj_scores_a = obj_result.get("scores", {}).get("A", {})
        obj_scores_b = obj_result.get("scores", {}).get("B", {})
        subj_scores_a = subj_result.get("scores", {}).get("A", {})
        subj_scores_b = subj_result.get("scores", {}).get("B", {})

        agg = aggregate(
            obj_scores_a=obj_scores_a,
            obj_scores_b=obj_scores_b,
            subj_scores_a=subj_scores_a,
            subj_scores_b=subj_scores_b,
            hk_flags_obj=obj_result.get("hong_kong_specific_flags"),
            hk_flags_subj=subj_result.get("hong_kong_specific_flags"),
            failure_flags_obj=obj_result.get("failure_mode_flags"),
            failure_flags_subj=subj_result.get("failure_mode_flags"),
            low_conf_obj=obj_result.get("low_confidence_flags"),
            low_conf_subj=subj_result.get("low_confidence_flags"),
            subj_d8_note=subj_result.get("d8_note"),
        )

        _drop = lambda d: {k: v for k, v in d.items() if k != "total"}
        all_scores_a = {**_drop(obj_scores_a), **_drop(subj_scores_a)}
        all_scores_b = {**_drop(obj_scores_b), **_drop(subj_scores_b)}

        total_a = sum(all_scores_a.values())
        total_b = sum(all_scores_b.values())

        # 合并 reasoning_chain（objective D1-D4 + subjective D5-D8）
        rc_a_obj = obj_result.get("reasoning_chain", {}).get("A", {})
        rc_a_subj = subj_result.get("reasoning_chain", {}).get("A", {})
        rc_b_obj = obj_result.get("reasoning_chain", {}).get("B", {})
        rc_b_subj = subj_result.get("reasoning_chain", {}).get("B", {})
        reasoning_chain = {
            "A": {**rc_a_obj, **rc_a_subj},
            "B": {**rc_b_obj, **rc_b_subj},
        }

        # 合并 reasoning
        def _merge_str(a: str, b: str) -> str:
            a, b = (a or "").strip(), (b or "").strip()
            return f"{a}；{b}" if a and b else a or b

        obj_reason = obj_result.get("reasoning", {})
        subj_reason = subj_result.get("reasoning", {})
        reasoning = {
            "decisive_factor": agg.decisive_factor,
            "A_strengths": _merge_str(obj_reason.get("A_strengths", ""), subj_reason.get("A_strengths", "")),
            "A_weaknesses": _merge_str(obj_reason.get("A_weaknesses", ""), subj_reason.get("A_weaknesses", "")),
            "B_strengths": _merge_str(obj_reason.get("B_strengths", ""), subj_reason.get("B_strengths", "")),
            "B_weaknesses": _merge_str(obj_reason.get("B_weaknesses", ""), subj_reason.get("B_weaknesses", "")),
        }

        total_elapsed = time.time() - t0
        self._log(f"  overall_preference: {agg.overall_preference}")
        self._log(f"  A 总分: {total_a} = {' + '.join(f'{k}={v}' for k,v in sorted(all_scores_a.items()))}")
        self._log(f"  B 总分: {total_b} = {' + '.join(f'{k}={v}' for k,v in sorted(all_scores_b.items()))}")
        if agg.veto_triggered:
            self._log(f"  🚫 D5否决触发: {agg.veto_reason}")
        if agg.d8_note:
            self._log(f"  ⚠ d8_note: {agg.d8_note}")
        self._log(f"  决定性因子: {agg.decisive_factor}")
        self._log(f"  总耗时: {total_elapsed:.0f}s")

        # 内容指纹校验
        verifier = ContentVerifier()
        verification = verifier.verify(
            content_a=option_a_content,
            content_b=option_b_content,
            reasoning_chain=reasoning_chain,
            reasoning=reasoning,
        )
        if verification.swap_flagged:
            self._log(f"  ⚠️ 内容指纹校验: A/B可能混淆 (方向={verification.swap_direction}, "
                      f"置信度={verification.swap_confidence:.3f})")
        elif not verification.insufficient_terms:
            self._log(f"  🔒 内容指纹校验通过 (A→A={verification.a_reasoning_a_overlap:.2f} "
                      f"A→B={verification.a_reasoning_b_overlap:.2f} "
                      f"B→B={verification.b_reasoning_b_overlap:.2f} "
                      f"B→A={verification.b_reasoning_a_overlap:.2f})")

        return {
            "query_id": query_id,
            "scores": {
                "A": {**all_scores_a, "total": total_a},
                "B": {**all_scores_b, "total": total_b},
            },
            "reasoning_chain": reasoning_chain,
            "overall_preference": agg.overall_preference,
            "reasoning": reasoning,
            "hong_kong_specific_flags": agg.hong_kong_specific_flags,
            "failure_mode_flags": agg.failure_mode_flags,
            "low_confidence_flags": agg.low_confidence_flags,
            "d8_note": agg.d8_note,
            "content_verification": {
                "swap_flagged": verification.swap_flagged,
                "swap_direction": verification.swap_direction,
                "swap_confidence": verification.swap_confidence,
                "a_reasoning_a_overlap": round(verification.a_reasoning_a_overlap, 4),
                "a_reasoning_b_overlap": round(verification.a_reasoning_b_overlap, 4),
                "b_reasoning_b_overlap": round(verification.b_reasoning_b_overlap, 4),
                "b_reasoning_a_overlap": round(verification.b_reasoning_a_overlap, 4),
                "n_discriminative_terms": {
                    "A": verification.n_discriminative_a,
                    "B": verification.n_discriminative_b,
                },
                "flagged_terms_sample": {
                    "reasoning_A_matched_B": verification.a_match_terms_b[:5],
                    "reasoning_B_matched_A": verification.b_match_terms_a[:5],
                },
            },
        }
