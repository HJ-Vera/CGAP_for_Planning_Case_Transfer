"""DeepSeek judge implementation."""

import os
from typing import Any
from ._base_judge import BaseJudge


class DeepSeekJudge(BaseJudge):
    """Judge that uses the DeepSeek API for evaluation."""

    def __init__(self, model: str = "deepseek-chat", api_key: str | None = None, base_url: str | None = None, **kwargs: Any):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.base_url = base_url or "https://api.deepseek.com"
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")

    def evaluate(self, prompt: str, response_schema: dict | None = None) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        client = OpenAI(api_key=self._api_key, base_url=self.base_url)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.extra_config.get("temperature", 0.0),
            "max_tokens": self.extra_config.get("max_tokens", 50000),
        }

        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
