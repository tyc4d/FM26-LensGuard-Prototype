"""Gemma 3 4B IT adapter for the Phase 2.5 local provider contract."""

from __future__ import annotations

from typing import Any

from PIL import Image

from providers.base import ProviderDependencyError

from .base_local_vlm import (
    BaseLocalVLMProvider,
    LocalGeneration,
    LocalModelSpec,
    PreparedLocalInput,
    decoder_only_generation,
    input_token_count,
    move_inputs_to_device,
    processed_image_dimensions,
)


class Gemma3Provider(BaseLocalVLMProvider):
    """Deterministic BF16/SDPA provider for ``google/gemma-3-4b-it``."""

    MODEL_SPEC = LocalModelSpec(
        alias="gemma3-4b",
        repository_id="google/gemma-3-4b-it",
        role="small-model baseline",
    )

    def _load_components(self, torch_module: Any, transformers: Any) -> tuple[Any, Any]:
        model_class = getattr(transformers, "Gemma3ForConditionalGeneration", None)
        processor_class = getattr(transformers, "AutoProcessor", None)
        if model_class is None or processor_class is None:
            raise ProviderDependencyError(
                "The Gemma environment requires Transformers with "
                "Gemma3ForConditionalGeneration and AutoProcessor"
            )
        revision = {"revision": self.requested_revision} if self.requested_revision else {}
        processor = processor_class.from_pretrained(self.repository_id, **revision)
        model = model_class.from_pretrained(
            self.repository_id,
            torch_dtype=torch_module.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            **revision,
        )
        return model, processor

    def _prepare_input(self, prompt: str, image: Image.Image) -> PreparedLocalInput:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        templater = getattr(self.processor, "apply_chat_template", None)
        if not callable(templater):
            raise ProviderDependencyError("Gemma processor does not expose apply_chat_template()")
        inputs = templater(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = move_inputs_to_device(
            inputs,
            self.device,
            dtype=self._torch_module().bfloat16,
        )
        width, height = processed_image_dimensions(inputs)
        return PreparedLocalInput(
            payload=inputs,
            input_token_count=input_token_count(inputs),
            processed_image_width=width,
            processed_image_height=height,
            metadata={
                "chat_template_adapter": "gemma3-single-user-multimodal-v1",
                "system_instruction_transport": "shared_user_message_wrapper",
            },
        )

    def _generate(self, prepared: PreparedLocalInput) -> LocalGeneration:
        return decoder_only_generation(
            model=self.model,
            processor=self.processor,
            prepared=prepared,
            max_new_tokens=self.max_new_tokens,
        )


__all__ = ["Gemma3Provider"]
