"""MiniCPM-V 4.5 adapter for the Phase 2.5 local provider contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PIL import Image

from providers.base import ProviderDependencyError, ProviderResponseError

from .base_local_vlm import (
    BaseLocalVLMProvider,
    LocalGeneration,
    LocalModelSpec,
    PreparedLocalInput,
)


class MiniCPMProvider(BaseLocalVLMProvider):
    """Deterministic BF16/SDPA provider for ``openbmb/MiniCPM-V-4_5``."""

    MODEL_SPEC = LocalModelSpec(
        alias="minicpm-v4.5",
        repository_id="openbmb/MiniCPM-V-4_5",
        role="edge-oriented multimodal baseline",
    )
    TRUST_REMOTE_CODE = True
    EFFECTIVE_ATTENTION_BACKEND = "llm_sdpa_vision_eager"

    def __init__(self, *, tokenizer: Any | None = None, **kwargs: Any) -> None:
        injected_processor = kwargs.get("processor")
        self.tokenizer = tokenizer
        if kwargs.get("model") is not None and self.tokenizer is None:
            # A combined injected fake may implement both interfaces. Real
            # loading always resolves two distinct repository components.
            self.tokenizer = injected_processor
        super().__init__(**kwargs)
        if self.is_loaded and self.tokenizer is None:
            raise ProviderDependencyError(
                "Injected MiniCPM provider requires a tokenizer or a processor with encode()"
            )

    @property
    def tokenizer_revision(self) -> str:
        value = getattr(self.tokenizer, "_commit_hash", None)
        if not isinstance(value, str) or not value.strip():
            init_kwargs = getattr(self.tokenizer, "init_kwargs", None)
            value = init_kwargs.get("_commit_hash") if isinstance(init_kwargs, Mapping) else None
        return (
            value.strip()
            if isinstance(value, str) and value.strip()
            else self.requested_revision or "unresolved"
        )

    @property
    def experiment_config(self) -> dict[str, Any]:
        return {
            **super().experiment_config,
            "tokenizer_revision": self.tokenizer_revision,
            "attention_backend_detail": {
                "language_model": "sdpa",
                "vision_model": "eager",
            },
        }

    def _load_components(self, torch_module: Any, transformers: Any) -> tuple[Any, Any]:
        model_class = getattr(transformers, "AutoModel", None)
        tokenizer_class = getattr(transformers, "AutoTokenizer", None)
        processor_class = getattr(transformers, "AutoProcessor", None)
        if model_class is None or tokenizer_class is None or processor_class is None:
            raise ProviderDependencyError(
                "The MiniCPM environment requires Transformers AutoModel, AutoTokenizer, "
                "and AutoProcessor"
            )
        revision = {"revision": self.requested_revision} if self.requested_revision else {}
        self.tokenizer = tokenizer_class.from_pretrained(
            self.repository_id,
            trust_remote_code=True,
            **revision,
        )
        processor = processor_class.from_pretrained(
            self.repository_id,
            trust_remote_code=True,
            **revision,
        )
        model = model_class.from_pretrained(
            self.repository_id,
            trust_remote_code=True,
            torch_dtype=torch_module.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            **revision,
        )
        return model, processor

    def _prepare_input(self, prompt: str, image: Image.Image) -> PreparedLocalInput:
        # MiniCPM's repository-defined ``chat`` owns its internal visual processor.
        # Keep this non-standard behavior entirely inside this family adapter.
        encoder = getattr(self.tokenizer, "encode", None)
        text_tokens: int | None = None
        if callable(encoder):
            try:
                text_tokens = len(encoder(prompt, add_special_tokens=True))
            except (TypeError, ValueError):
                text_tokens = None
        return PreparedLocalInput(
            payload={
                "image": None,
                "msgs": [{"role": "user", "content": [image, prompt]}],
            },
            # The repository chat method does not expose its visual-token count.
            # Keep the aggregate field unknown instead of understating it with
            # the text-only tokenizer count.
            input_token_count=None,
            processed_image_width=None,
            processed_image_height=None,
            metadata={
                "chat_template_adapter": "minicpm-v4.5-remote-chat-v1",
                "system_instruction_transport": "shared_user_message_wrapper",
                "input_token_count_scope": "text_only_excludes_internal_visual_tokens",
                "text_input_token_count": text_tokens,
                "processed_image_dimensions_unavailable": True,
            },
        )

    def _generate(self, prepared: PreparedLocalInput) -> LocalGeneration:
        chatter = getattr(self.model, "chat", None)
        if not callable(chatter):
            raise ProviderDependencyError("MiniCPM remote-code model does not expose chat()")
        raw = chatter(
            **prepared.payload,
            tokenizer=self.tokenizer,
            processor=self.processor,
            enable_thinking=False,
            sampling=False,
            stream=False,
            num_beams=1,
            repetition_penalty=1.0,
            max_new_tokens=self.max_new_tokens,
        )
        if not isinstance(raw, str):
            raise ProviderResponseError("MiniCPM chat() did not return one decoded text string")
        encoder = getattr(self.tokenizer, "encode", None)
        output_tokens: int | None = None
        if callable(encoder):
            try:
                output_tokens = len(encoder(raw, add_special_tokens=False))
            except (TypeError, ValueError):
                output_tokens = None
        return LocalGeneration(
            raw_text=raw,
            output_token_count=output_tokens,
            generated_tokens=output_tokens,
            metadata={
                "generation_mode": "repository_remote_code_chat",
                "generation_latency_scope": "remote_chat_internal_preprocess_generate_decode",
                "output_token_count_scope": "decoded_text_retokenized",
                "tokenizer_revision": self.tokenizer_revision,
                "attention_backend_detail": {
                    "language_model": "sdpa",
                    "vision_model": "eager",
                },
            },
        )

    def close(self) -> None:
        self.tokenizer = None
        super().close()


__all__ = ["MiniCPMProvider"]
