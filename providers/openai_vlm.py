"""OpenAI Responses API image adapter for the separate cloud benchmark."""

from __future__ import annotations

import importlib.metadata
import os
from typing import Any

from providers.base_cloud_vlm import (
    CloudRequest, CloudVLMProvider, _field, disable_sdk_debug_logging, jsonable,
    require_secret_safety,
)


class OpenAIProvider(CloudVLMProvider):
    provider = "openai"
    api_interface = "openai.responses"
    default_model = "gpt-5.6-sol"
    model_environment = "OPENAI_MODEL"
    transport_delays = (1.0, 2.0, 4.0, 8.0)

    def __init__(self, *, client: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._owns_client = client is None
        self.sdk_version = importlib.metadata.version("openai")
        if client is not None:
            self.client = client
            return
        require_secret_safety()
        disable_sdk_debug_logging()
        if not os.getenv("OPENAI_API_KEY"):
            self.client = None
            self._initialization_error = "MISSING_CREDENTIALS"
            return
        from openai import OpenAI

        # Supplying the environment value explicitly also avoids implicit alternate auth.
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=120.0)
        disable_sdk_debug_logging()

    @property
    def provider_config(self) -> dict[str, Any]:
        return {
            "sdk": "openai", "sdk_version": self.sdk_version,
            "api_interface": self.api_interface, "store": False, "tools": [],
            "generation_config": {
                "reasoning": {"effort": "none"}, "temperature": 0,
                "max_output_tokens": 2048,
            },
            "sdk_internal_retries": 0,
            "transport_retry_delays_seconds": list(self.transport_delays),
            "request_timeout_seconds": 120.0,
            "latency_scope": "API transport including retries/backoff; excludes request pacing",
        }

    def build_payload(self, request: CloudRequest) -> dict[str, Any]:
        image, mime_type = request.image_base64()
        return {
            "model": self.model,
            "input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": f"data:{mime_type};base64,{image}"},
                {"type": "input_text", "text": request.prompt},
            ]}],
            "text": {"format": {
                "type": "json_schema", "name": "lensguard_action", "strict": True,
                "schema": request.response_schema,
            }},
            "reasoning": {"effort": "none"}, "temperature": 0,
            "max_output_tokens": 2048, "tools": [], "store": False,
        }

    def _send(self, payload: dict[str, Any]) -> Any:
        return self.client.responses.create(**payload)

    def _usage(self, response: Any) -> dict[str, Any]:
        usage = jsonable(_field(response, "usage")) or {}
        return {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
            "cached_input_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens"),
            "total_tokens": usage.get("total_tokens"), "native": usage,
        }
