"""Exactly scoped local VLM providers for LensGuard Phase 2.5."""

from __future__ import annotations

from typing import Any

from providers.base import ProviderConfigurationError

from .base_local_vlm import (
    BaseLocalVLMProvider,
    LOCAL_ATTENTION_BACKEND,
    LOCAL_BATCH_SIZE,
    LOCAL_DTYPE,
    LOCAL_PROVIDER_INTERFACE_VERSION,
    LOCAL_QUANTIZATION,
    LOCAL_SCHEMA_TRANSPORT_VERSION,
    LOCAL_STRUCTURED_DECODING_MODE,
    LocalOutputContractDiagnostics,
    LocalVLMOutOfMemoryError,
    ZERO_SHOT_V1,
    ZERO_SHOT_V2,
    build_zero_shot_prompt,
    extract_single_json_object,
    parse_local_output,
)
from .gemma3_provider import Gemma3Provider
from .minicpm_provider import MiniCPMProvider
from .qwen3vl_provider import Qwen3VLProvider


LOCAL_MODEL_PROVIDERS: dict[str, type[BaseLocalVLMProvider]] = {
    Gemma3Provider.MODEL_SPEC.alias: Gemma3Provider,
    Qwen3VLProvider.MODEL_SPEC.alias: Qwen3VLProvider,
    MiniCPMProvider.MODEL_SPEC.alias: MiniCPMProvider,
}
LOCAL_MODEL_REPOSITORIES: dict[str, str] = {
    alias: provider.MODEL_SPEC.repository_id
    for alias, provider in LOCAL_MODEL_PROVIDERS.items()
}


def create_local_provider(model_alias: str, **kwargs: Any) -> BaseLocalVLMProvider:
    """Instantiate one of the three declared aliases; repository IDs are not aliases."""

    try:
        provider = LOCAL_MODEL_PROVIDERS[model_alias]
    except KeyError as error:
        expected = ", ".join(LOCAL_MODEL_PROVIDERS)
        raise ProviderConfigurationError(
            f"Unsupported Phase 2.5 local model alias {model_alias!r}; choose one of: {expected}"
        ) from error
    return provider(**kwargs)


__all__ = [
    "BaseLocalVLMProvider",
    "Gemma3Provider",
    "LOCAL_ATTENTION_BACKEND",
    "LOCAL_BATCH_SIZE",
    "LOCAL_DTYPE",
    "LOCAL_MODEL_PROVIDERS",
    "LOCAL_MODEL_REPOSITORIES",
    "LOCAL_PROVIDER_INTERFACE_VERSION",
    "LOCAL_QUANTIZATION",
    "LOCAL_SCHEMA_TRANSPORT_VERSION",
    "LOCAL_STRUCTURED_DECODING_MODE",
    "LocalOutputContractDiagnostics",
    "LocalVLMOutOfMemoryError",
    "MiniCPMProvider",
    "Qwen3VLProvider",
    "ZERO_SHOT_V1",
    "ZERO_SHOT_V2",
    "build_zero_shot_prompt",
    "create_local_provider",
    "extract_single_json_object",
    "parse_local_output",
]
