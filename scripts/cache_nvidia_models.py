"""Download pinned NVIDIA assets before starting the offline resident service."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from providers.local.nvidia_models import NVIDIA_MODEL_PROFILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=list(NVIDIA_MODEL_PROFILES))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    profile = NVIDIA_MODEL_PROFILES[args.model]
    print(snapshot_download(profile['model_id'], revision=profile['revision'],
        allow_patterns=['*.json', '*.py', '*.safetensors'], max_workers=3))
    if 'radio_model_id' in profile:
        # Nemotron includes vision weights; only RADIO's pinned code/config is needed.
        print(snapshot_download(profile['radio_model_id'], revision=profile['radio_revision'],
            allow_patterns=['*.json', '*.py'], max_workers=3))


if __name__ == '__main__':
    main()
