"""Batch runner: runs evaluations across multiple test cases and saves results."""

import json
import os
from pathlib import Path
from typing import Any

from ..judges._base_judge import BaseJudge
from .pairwise_runner import PairwiseRunner
from .pointwise_runner import PointwiseRunner


class BatchRunner:
    """Runs batch evaluations and saves results to eval_output/."""

    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "eval_output"

    def __init__(self, judge: BaseJudge):
        self.judge = judge
        self.pairwise_runner = PairwiseRunner(judge)
        self.pointwise_runner = PointwiseRunner(judge)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def _load_dataset(self, dataset_path: str) -> list[dict]:
        """Load test cases from a JSON file.

        Expected format:
        [
            {
                "query_id": "Q-001",
                "query": "...",
                "option_a": {"content": "..."},
                "option_b": {"content": "..."}
            },
            ...
        ]
        """
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_pairwise_batch(self, dataset_path: str, output_name: str = "batch_results") -> str:
        """Run pairwise evaluations on a batch of test cases.

        Args:
            dataset_path: Path to the JSON dataset file.
            output_name: Base name for the output file (without extension).

        Returns:
            Path to the output JSON file.
        """
        cases = self._load_dataset(dataset_path)
        results = []

        for i, case in enumerate(cases):
            print(f"[{i+1}/{len(cases)}] Running: {case.get('query_id', 'unknown')}")
            result = self.pairwise_runner.run(
                query_id=case.get("query_id", f"Q-{i:03d}"),
                query=case["query"],
                option_a_content=case.get("option_a", {}).get("content", ""),
                option_b_content=case.get("option_b", {}).get("content", ""),
            )
            results.append(result)

        output_path = self.OUTPUT_DIR / f"{output_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(results)} results to {output_path}")
        return str(output_path)

    def run_pointwise_batch(self, dataset_path: str, output_name: str = "pointwise_results") -> str:
        """Run pointwise evaluations on a batch of single-option test cases.

        Args:
            dataset_path: Path to the JSON dataset file.
            output_name: Base name for the output file.

        Returns:
            Path to the output JSON file.
        """
        cases = self._load_dataset(dataset_path)
        results = []

        for i, case in enumerate(cases):
            print(f"[{i+1}/{len(cases)}] Running: {case.get('query_id', 'unknown')}")
            result = self.pointwise_runner.run(
                query_id=case.get("query_id", f"Q-{i:03d}"),
                query=case["query"],
                option_content=case.get("option", {}).get("content", ""),
            )
            results.append(result)

        output_path = self.OUTPUT_DIR / f"{output_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(results)} results to {output_path}")
        return str(output_path)
