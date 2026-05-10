"""Pointwise runner: executes single-case evaluation using objective + subjective skills."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
import yaml

from ..judges._base_judge import BaseJudge
from ..parser.pointwise_parser import PointwiseParser


class PointwiseRunner:
    """Runs a pointwise (single option) evaluation using two skills and a judge."""

    SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

    def __init__(self, judge: BaseJudge):
        self.judge = judge
        self.parser = PointwiseParser()
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(self.SKILLS_DIR)),
            autoescape=False,
        )

    def _load_skill_config(self, skill_name: str) -> dict:
        config_path = self.SKILLS_DIR / skill_name / "skill.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _render_prompt(self, skill_name: str, context: dict) -> str:
        template = self._jinja_env.get_template(f"{skill_name}/prompt.jinja2")
        return template.render(**context)

    def _run_skill(self, skill_name: str, query_id: str, query: str,
                   option_content: str) -> dict:
        skill_config = self._load_skill_config(skill_name)
        context = {
            "query_id": query_id,
            "planning_query": query,
            "option_content": option_content,
            "dimensions": skill_config["dimensions"],
        }
        prompt = self._render_prompt(skill_name, context)
        raw_output = self.judge.evaluate(prompt)
        return self.parser.parse(raw_output)

    def run(self, query_id: str, query: str, option_content: str) -> dict:
        """Execute a pointwise evaluation for a single case.

        Returns a dict with combined scores from both skills.
        """
        obj_result = self._run_skill("objective", query_id, query, option_content)
        subj_result = self._run_skill("subjective", query_id, query, option_content)

        obj_scores = obj_result.get("scores", {})
        subj_scores = subj_result.get("scores", {})
        all_scores = {**obj_scores, **subj_scores}

        return {
            "query_id": query_id,
            "scores": {**all_scores, "total": sum(all_scores.values())},
            "objective_result": obj_result,
            "subjective_result": subj_result,
            "flags": {
                "hong_kong_specific": (
                    obj_result.get("hong_kong_specific_flags", [])
                    + subj_result.get("hong_kong_specific_flags", [])
                ),
                "failure_mode": (
                    obj_result.get("failure_mode_flags", [])
                    + subj_result.get("failure_mode_flags", [])
                ),
                "low_confidence": (
                    obj_result.get("low_confidence_flags", [])
                    + subj_result.get("low_confidence_flags", [])
                ),
            },
            "d8_note": subj_result.get("d8_note"),
        }
