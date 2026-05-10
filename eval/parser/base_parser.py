"""Base parser for LLM-as-Judge evaluation output."""

import json
import re
from typing import Any


class BaseParser:
    """Common parsing utilities shared by pairwise and pointwise parsers."""

    @staticmethod
    def extract_json(text: str) -> dict:
        """Extract JSON object from LLM output that may contain extra text."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(
                f"Failed to parse JSON from output (len={len(text)}):\n{text[:500]}"
            )

    @staticmethod
    def validate_score_range(scores: dict, dims: list[str], min_val: int = 1, max_val: int = 5) -> list[str]:
        """Validate that all dimension scores are within the allowed range. Returns list of error messages."""
        errors = []
        for dim in dims:
            if dim not in scores:
                errors.append(f"Missing dimension: {dim}")
            elif not isinstance(scores[dim], int):
                errors.append(f"Non-integer score for {dim}: {scores[dim]}")
            elif scores[dim] < min_val or scores[dim] > max_val:
                errors.append(f"Score out of range [{min_val},{max_val}] for {dim}: {scores[dim]}")
        return errors

    @staticmethod
    def validate_required_fields(data: dict, required_fields: list[str]) -> list[str]:
        """Check that all required top-level fields are present. Returns list of missing field names."""
        return [f for f in required_fields if f not in data]
