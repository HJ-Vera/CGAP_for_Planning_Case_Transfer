"""GLM (ZhipuAI) judge implementation."""

import os
from typing import Any
from ._base_judge import BaseJudge


class GLMJudge(BaseJudge):
    """Judge that uses the ZhipuAI GLM API for evaluation."""

    def __init__(self, model: str = "glm-4-flash", api_key: str | None = None,
                 base_url: str | None = None, **kwargs: Any):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._api_key = api_key or os.environ.get("ZHIPUAI_API_KEY", "")
        self._base_url = base_url

    def evaluate(self, prompt: str, response_schema: dict | None = None) -> str:
        try:
            from zhipuai import ZhipuAI
        except ImportError:
            raise ImportError("zhipuai package required: pip install zhipuai")

        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        client = ZhipuAI(**client_kwargs)

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
