"""NVIDIA's native tiled RGB processor and stateless Nemotron chat transport.

Remote code is required for both Nemotron and C-RADIO. Resolve the nested vision
configuration locally at a pinned revision: upstream configuration.py otherwise
imports C-RADIO's moving main branch even when Nemotron itself is pinned.
"""
import json

from providers.base import ProviderConfigurationError, ProviderResponseError
from .base_local_vlm import (
    BaseLocalVLMProvider, LocalGeneration, LocalModelSpec, PreparedLocalInput,
    move_inputs_to_device,
)
from .nvidia_models import NVIDIA_MODEL_PROFILES


class NemotronNanoVLProvider(BaseLocalVLMProvider):
    DEMO_SEMANTIC_CONTRACT = 'observations-v1'
    PROFILE = NVIDIA_MODEL_PROFILES['nemotron-nano-vl-8b']
    MODEL_SPEC = LocalModelSpec(
        PROFILE['family_alias'], PROFILE['model_id'], 'NVIDIA general vision/language baseline')
    TRUST_REMOTE_CODE = True
    EFFECTIVE_ATTENTION_BACKEND = 'llm_sdpa_vision_timm'

    def __init__(self, *, tokenizer=None, **kwargs):
        kwargs.setdefault('revision', self.PROFILE['revision'])
        self.tokenizer = tokenizer
        super().__init__(**kwargs)
        # NVIDIA chat() uses .cuda() internally and only supports the default GPU.
        if self.device not in ('cuda', 'cuda:0'):
            raise ProviderConfigurationError('Nemotron native chat requires CUDA device 0')

    def _load_components(self, torch_module, transformers):
        from huggingface_hub import hf_hub_download
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config_path = hf_hub_download(
            self.repository_id, 'config.json', revision=self.requested_revision)
        with open(config_path) as file:
            data = json.load(file)
        vision_data = data.pop('vision_config')
        radio_kwargs = {'revision': self.PROFILE['radio_revision']}
        radio_config_class = get_class_from_dynamic_module(
            'hf_model.RADIOConfig', self.PROFILE['radio_model_id'], **radio_kwargs)
        radio_model_class = get_class_from_dynamic_module(
            'hf_model.RADIOModel', self.PROFILE['radio_model_id'], **radio_kwargs)
        # Use Transformers' normal custom-class registration. Removing auto_map
        # prevents upstream AutoModel.from_config from fetching moving remote code.
        transformers.AutoModel.register(radio_config_class, radio_model_class, exist_ok=True)
        vision_data.pop('auto_map', None)
        vision = radio_config_class(**vision_data)
        vision._commit_hash = self.PROFILE['radio_revision']
        vision.use_flash_attn = False
        config_class = get_class_from_dynamic_module(
            'configuration.Llama_Nemotron_Nano_VL_Config', self.repository_id,
            revision=self.requested_revision)
        # No vision_config here: avoid upstream's unconditional main-branch import.
        data.pop('attn_implementation', None)
        config = config_class(**data, attn_implementation='eager')
        config.vision_config = vision
        config.llm_config._attn_implementation = 'sdpa'
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.repository_id, revision=self.requested_revision)
        processor = transformers.AutoImageProcessor.from_pretrained(
            self.repository_id, revision=self.requested_revision, trust_remote_code=True,
            device=self.device, max_num_tiles=self.PROFILE['max_num_tiles'])
        model = transformers.AutoModel.from_pretrained(
            self.repository_id, revision=self.requested_revision, config=config,
            trust_remote_code=True, torch_dtype=torch_module.bfloat16, low_cpu_mem_usage=True)
        return model, processor

    def _prepare_input(self, prompt, image):
        features = self.processor(image, return_tensors='pt')
        features = move_inputs_to_device(
            features, self.device, dtype=self._torch_module().bfloat16)
        return PreparedLocalInput(
            payload={'question': prompt, **features},
            processed_image_width=512, processed_image_height=512,
            metadata={'chat_template_adapter': 'nemotron-native-chat-v1',
                      'max_num_tiles': self.PROFILE['max_num_tiles'],
                      'processed_dimensions_scope': 'per_tile',
                      'input_token_count_scope': 'unavailable_inside_remote_chat'})

    def _prepare_text_input(self, prompt):
        return PreparedLocalInput(
            payload={'question': prompt, 'pixel_values': None},
            metadata={'chat_template_adapter': 'nemotron-native-chat-text-v1', 'text_only': True})

    def _generate(self, prepared):
        raw = self.model.chat(
            tokenizer=self.tokenizer, **prepared.payload,
            generation_config={'max_new_tokens': self.max_new_tokens, 'do_sample': False,
                               'eos_token_id': self.tokenizer.eos_token_id},
            history=None, return_history=False)
        if not isinstance(raw, str):
            raise ProviderResponseError('Nemotron chat() did not return a decoded string')
        count = len(self.tokenizer.encode(raw, add_special_tokens=False))
        return LocalGeneration(raw_text=raw, output_token_count=count, generated_tokens=count,
            metadata={'generation_mode': 'repository_remote_code_chat',
                      'output_token_count_scope': 'decoded_text_retokenized'})

    def close(self):
        self.tokenizer = None
        super().close()
