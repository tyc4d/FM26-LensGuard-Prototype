"""Pinned NVIDIA model/runtime configuration; no authorization settings."""

NVIDIA_MODEL_PROFILES = {
    'nemotron-nano-vl-8b': {
        'family_alias': 'nemotron-nano-vl-8b',
        'model_id': 'nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1',
        'revision': '437f4e28b989cc2d9a16b6767cc930cdf48797ff',
        'transformers_version': '4.53.3',
        'max_new_tokens': 1024,
        'minimum_free_mib': 21000,
        'max_num_tiles': 4,
        'radio_model_id': 'nvidia/C-RADIOv2-H',
        'radio_revision': '0d8f4c18c877166eda07ddae1386bcad256b7a6a',
    },
    'cosmos-reason1-7b': {
        'family_alias': 'cosmos-reason1-7b',
        'model_id': 'nvidia/Cosmos-Reason1-7B',
        'revision': '375e24000b24baed78f4618d3dd779e47cd96323',
        'transformers_version': '5.16.1',
        'max_new_tokens': 1024,
        'minimum_free_mib': 21000,
        'min_pixels': 256 * 28 * 28,
        'max_pixels': 1280 * 28 * 28,
    },
}
