"""Parser for pairwise (A vs B) evaluation output."""

from typing import Any
from .base_parser import BaseParser


class PairwiseParser(BaseParser):
    """Parses the JSON output from a pairwise skill evaluation into structured data."""

    SCORE_DIMENSIONS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]

    def parse(self, raw_output: str) -> dict:
        """Parse raw LLM output into a structured pairwise evaluation result.

        Returns a dict with:
            - query_id: str
            - scores: dict with A and B sub-dicts of dimension scores + total
            - reasoning_chain: dict with A and B sub-dicts of per-dimension reasoning
            - overall_preference: "A" | "B" | "tie"
            - reasoning: dict with decisive_factor, strengths/weaknesses
            - flags: hong_kong_specific_flags, failure_mode_flags, low_confidence_flags
            - d8_note: str or None
            - errors: list of validation error messages
        """
        data = self.extract_json(raw_output)
        errors = []

        required_fields = ["query_id", "scores", "reasoning_chain", "overall_preference"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        scores = data.get("scores", {})
        for option in ["A", "B"]:
            if option in scores:
                dim_errors = self.validate_score_range(scores[option], self.SCORE_DIMENSIONS)
                errors.extend([f"Option {option}: {e}" for e in dim_errors])

        overall = data.get("overall_preference", "")
        if overall not in ("A", "B", "tie"):
            errors.append(f"Invalid overall_preference: {overall}")

        return {
            "query_id": data.get("query_id", ""),
            "scores": scores,
            "reasoning_chain": data.get("reasoning_chain", {}),
            "overall_preference": overall,
            "reasoning": data.get("reasoning", {}),
            "hong_kong_specific_flags": data.get("hong_kong_specific_flags", []),
            "failure_mode_flags": data.get("failure_mode_flags", []),
            "low_confidence_flags": data.get("low_confidence_flags", []),
            "d8_note": data.get("d8_note"),
            "errors": errors,
        }

    def extract_scores_for_skill(self, parsed: dict, dim_keys: list[str]) -> dict:
        """Extract only the scores relevant to a specific skill from a parsed result.

        Args:
            parsed: The full parsed result from parse()
            dim_keys: List of dimension score keys (e.g. ["D1_precision", "D2_scenario"])

        Returns dict with A and B sub-dicts containing only the requested dimensions.
        """
        result = {"A": {}, "B": {}}
        scores = parsed.get("scores", {})
        for option in ["A", "B"]:
            if option in scores:
                for key in dim_keys:
                    if key in scores[option]:
                        result[option][key] = scores[option][key]
        return result
