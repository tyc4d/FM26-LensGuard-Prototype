"""Cosmos Reason1 transport using its native Qwen2.5-VL architecture."""
from .base_local_vlm import (
    BaseLocalVLMProvider, LocalModelSpec, PreparedLocalInput,
    decoder_only_generation, input_token_count, move_inputs_to_device,
)
from .nvidia_models import NVIDIA_MODEL_PROFILES


class CosmosReason1Provider(BaseLocalVLMProvider):
    PROFILE = NVIDIA_MODEL_PROFILES['cosmos-reason1-7b']
    MODEL_SPEC = LocalModelSpec(
        PROFILE['family_alias'], PROFILE['model_id'], 'physical/embodied reasoning baseline')

    def __init__(self, **kwargs):
        kwargs.setdefault('revision', self.PROFILE['revision'])
        super().__init__(**kwargs)

    def _load_components(self, torch_module, transformers):
        processor = transformers.AutoProcessor.from_pretrained(
            self.repository_id, revision=self.requested_revision,
            min_pixels=self.PROFILE['min_pixels'], max_pixels=self.PROFILE['max_pixels'])
        model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.repository_id, revision=self.requested_revision,
            torch_dtype=torch_module.bfloat16, attn_implementation='sdpa',
            low_cpu_mem_usage=True)
        return model, processor

    def _prepare_input(self, prompt, image):
        content = [] if image is None else [{'type': 'image', 'image': image}]
        content.append({'type': 'text', 'text': prompt})
        inputs = self.processor.apply_chat_template(
            [{'role': 'user', 'content': content}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors='pt')
        inputs = move_inputs_to_device(inputs, self.device, dtype=self._torch_module().bfloat16)
        return PreparedLocalInput(
            payload=inputs, input_token_count=input_token_count(inputs),
            metadata={'chat_template_adapter': 'cosmos-reason1-qwen2.5-vl-v1',
                      'visual_max_pixels': self.PROFILE['max_pixels'],
                      'text_only': image is None})

    def _prepare_text_input(self, prompt):
        return self._prepare_input(prompt, None)

    def _generate(self, prepared):
        return decoder_only_generation(model=self.model, processor=self.processor,
            prepared=prepared, max_new_tokens=self.max_new_tokens)
