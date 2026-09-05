"""Sequential Google GenAI Interactions adapter with auditable quota handling."""

from __future__ import annotations

import importlib.metadata
import math
import os
import random
from collections.abc import Callable
from typing import Any

from providers.base import ProviderConfigurationError, interactions_without_internal_retry
from providers.base_cloud_vlm import (
    CloudRequest, CloudVLMProvider, _field, disable_sdk_debug_logging, jsonable,
    require_secret_safety,
)


class GeminiProvider(CloudVLMProvider):
    provider = "gemini"
    api_interface = "google.genai.interactions"
    default_model = "gemini-3.1-flash-lite"
    model_environment = "GEMINI_MODEL"
    transport_delays = (30.0, 60.0, 120.0, 240.0)

    def __init__(
        self, *, client: Any | None = None,
        jitter: Callable[[], float] = lambda: random.uniform(0.0, 2.0),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        try:
            self.request_delay_seconds = float(os.getenv("GEMINI_REQUEST_DELAY_SECONDS", "8"))
        except ValueError:
            raise ProviderConfigurationError("Gemini request delay must be a finite nonnegative number") from None
        if not math.isfinite(self.request_delay_seconds) or self.request_delay_seconds < 0:
            raise ProviderConfigurationError("Gemini request delay must be a finite nonnegative number")
        self._jitter = jitter
        self._next_request_at = 0.0
        self.next_request_delay_seconds = 0.0
        self.sdk_version = importlib.metadata.version("google-genai")
        self._owns_client = client is None
        if client is not None:
            self.client = client
        else:
            require_secret_safety()
            disable_sdk_debug_logging()
            if not os.getenv("GEMINI_API_KEY"):
                self.client = None
                self._initialization_error = "MISSING_CREDENTIALS"
                return
            from google import genai
            from google.genai import types

            self.client = genai.Client(
                api_key=os.environ["GEMINI_API_KEY"],
                http_options=types.HttpOptions(
                    api_version="v1beta", timeout=120_000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            disable_sdk_debug_logging()
        self._interactions = interactions_without_internal_retry(self.client)

    @property
    def provider_config(self) -> dict[str, Any]:
        return {
            "sdk": "google-genai", "sdk_version": self.sdk_version,
            "api_interface": self.api_interface, "api_version": "v1beta",
            "store": False, "tools": [],
            "generation_config": {
                "seed": 0, "thinking_level": "minimal", "thinking_summaries": "none",
                "max_output_tokens": 2048,
            },
            "temperature": "not exposed by pinned Interactions SDK",
            "sdk_internal_retries": 0,
            "transport_retry_delays_seconds": list(self.transport_delays),
            "request_timeout_seconds": 120.0,
            "request_delay_seconds": self.request_delay_seconds,
            "request_jitter_range_seconds": [0.0, 2.0], "concurrency": 1,
            "latency_scope": "API transport including retries/backoff; excludes request pacing",
        }

    def build_payload(self, request: CloudRequest) -> dict[str, Any]:
        image, mime_type = request.image_base64()
        return {
            "model": self.model,
            "input": [
                {"type": "image", "mime_type": mime_type, "data": image},
                {"type": "text", "text": request.prompt},
            ],
            "response_format": {
                "type": "text", "mime_type": "application/json",
                "schema": request.response_schema,
            },
            "generation_config": dict(self.provider_config["generation_config"]),
            "tools": [], "store": False, "api_version": "v1beta",
        }

    def _send(self, payload: dict[str, Any]) -> Any:
        return self._interactions.create(**payload)

    def _before_request(self) -> None:
        remaining = self._next_request_at - self._clock()
        if remaining > 0:
            self._bounded_sleep(remaining)

    def _after_success(self) -> None:
        jitter = self._jitter()
        if not math.isfinite(jitter) or not 0 <= jitter <= 2:
            raise ProviderConfigurationError("Gemini pacing jitter must be within 0 to 2 seconds")
        self.next_request_delay_seconds = self.request_delay_seconds + jitter
        self._next_request_at = self._clock() + self.next_request_delay_seconds

    def _usage(self, response: Any) -> dict[str, Any]:
        usage = jsonable(_field(response, "usage")) or {}
        return {
            "input_tokens": usage.get("total_input_tokens"),
            "output_tokens": usage.get("total_output_tokens"),
            "reasoning_tokens": usage.get("total_thought_tokens"),
            "cached_input_tokens": usage.get("total_cached_tokens"),
            "total_tokens": usage.get("total_tokens"), "native": usage,
        }
