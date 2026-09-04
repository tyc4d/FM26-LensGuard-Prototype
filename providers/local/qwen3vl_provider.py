"""Qwen3-VL 8B Instruct adapter for the Phase 2.5 local provider contract."""

from __future__ import annotations

from collections.abc import Mapping
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
    tensor_shape,
)


class Qwen3VLProvider(BaseLocalVLMProvider):
    """Deterministic BF16/SDPA provider for Qwen3-VL 8B Instruct."""

    MODEL_SPEC = LocalModelSpec(
        alias="qwen3vl-8b",
        repository_id="Qwen/Qwen3-VL-8B-Instruct",
        role="strong local quality baseline",
    )

    def _load_components(self, torch_module: Any, transformers: Any) -> tuple[Any, Any]:
        model_class = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
        processor_class = getattr(transformers, "AutoProcessor", None)
        if model_class is None or processor_class is None:
            raise ProviderDependencyError(
                "The Qwen environment requires Transformers with "
                "Qwen3VLForConditionalGeneration and AutoProcessor"
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

    def _qwen_processed_dimensions(self, inputs: Any) -> tuple[int | None, int | None]:
        width, height = processed_image_dimensions(inputs)
        if width is not None and height is not None:
            return width, height
        grid = inputs.get("image_grid_thw") if isinstance(inputs, Mapping) else None
        shape = tensor_shape(grid)
        if not shape or len(shape) < 2:
            return None, None
        try:
            first = grid[0]
            grid_height = int(first[-2])
            grid_width = int(first[-1])
            image_processor = getattr(self.processor, "image_processor", None)
            patch_size = int(getattr(image_processor, "patch_size", 0))
            if patch_size > 0:
                return grid_width * patch_size, grid_height * patch_size
        except (IndexError, TypeError, ValueError):
            pass
        return None, None

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
            raise ProviderDependencyError("Qwen processor does not expose apply_chat_template()")
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
        width, height = self._qwen_processed_dimensions(inputs)
        return PreparedLocalInput(
            payload=inputs,
            input_token_count=input_token_count(inputs),
            processed_image_width=width,
            processed_image_height=height,
            metadata={
                "chat_template_adapter": "qwen3vl-single-user-multimodal-v1",
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


__all__ = ["Qwen3VLProvider"]
