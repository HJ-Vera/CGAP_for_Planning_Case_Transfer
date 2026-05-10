"""Abstract base class for LLM judges."""

from abc import ABC, abstractmethod
from typing import Any


class BaseJudge(ABC):
    """Abstract base for all LLM judge implementations.

    Each judge implementation handles the API call to a specific LLM provider,
    sending the prompt and returning the raw response text.
    """

    def __init__(self, model: str, api_key: str | None = None, **kwargs: Any):
        self.model = model
        self.api_key = api_key
        self.extra_config = kwargs

    @abstractmethod
    def evaluate(self, prompt: str, response_schema: dict | None = None) -> str:
        """Send prompt to the LLM and return the raw response text.

        Args:
            prompt: The full prompt text (rendered Jinja2 template).
            response_schema: Optional JSON schema for structured output.

        Returns:
            Raw response string from the LLM.
        """
        ...
