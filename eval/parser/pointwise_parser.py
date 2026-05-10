"""Parser for pointwise (single-case) evaluation output."""

from typing import Any
from .base_parser import BaseParser


class PointwiseParser(BaseParser):
    """Parses the JSON output from a pointwise skill evaluation into structured data.

    Used for single-option evaluation (no A/B comparison).
    """

    SCORE_DIMENSIONS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]

    def parse(self, raw_output: str) -> dict:
        """Parse raw LLM output into a structured pointwise evaluation result.

        Returns a dict similar to pairwise but with a single "scores" entry
        and no overall_preference.
        """
        data = self.extract_json(raw_output)
        errors = []

        required_fields = ["query_id", "scores", "reasoning_chain"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        scores = data.get("scores", {})
        dim_errors = self.validate_score_range(scores, self.SCORE_DIMENSIONS)
        errors.extend(dim_errors)

        return {
            "query_id": data.get("query_id", ""),
            "scores": scores,
            "reasoning_chain": data.get("reasoning_chain", {}),
            "hong_kong_specific_flags": data.get("hong_kong_specific_flags", []),
            "failure_mode_flags": data.get("failure_mode_flags", []),
            "low_confidence_flags": data.get("low_confidence_flags", []),
            "d8_note": data.get("d8_note"),
            "errors": errors,
        }
