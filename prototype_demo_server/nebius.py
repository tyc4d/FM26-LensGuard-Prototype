"""Cloud transport using the same task, perception, citation and policy contracts."""
import base64
import os
from pathlib import Path
from time import perf_counter

import httpx

from .runtime import LocalRuntime

PROFILE = 'nebius-glm-5-3-flash'


class NebiusProvider:
    DEMO_SEMANTIC_CONTRACT = 'observations-v1'

    def __init__(self, *, api_key, model, base_url, transport=None, timeout=240):
        if not api_key:
            raise ValueError('NEBIUS_API_KEY is required for cloud inference.')
        self.model = model
        self.timeout = timeout
        self.deadline = None
        self.client = httpx.Client(
            base_url=base_url.rstrip('/') + '/',
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=httpx.Timeout(timeout, connect=10), transport=transport,
        )

    def _read_image(self, path):
        data = Path(path).read_bytes()
        mime = 'image/png' if data.startswith(b'\x89PNG\r\n\x1a\n') else 'image/jpeg'
        return f'data:{mime};base64,{base64.b64encode(data).decode("ascii")}', None, None

    def generate_remote(self, prompt, image=None):
        content = [{'type': 'input_text', 'text': prompt}]
        if image is not None:
            content.append({'type': 'input_image', 'image_url': image, 'detail': 'auto'})
        started = perf_counter()
        remaining = self.timeout if self.deadline is None else min(self.timeout, self.deadline - started)
        if remaining <= 0:
            raise RuntimeError('NEBIUS_TIMEOUT: analysis exceeded its total time limit.')
        try:
            response = self.client.post('responses', json={
                'model': self.model,
                'input': [{'role': 'user', 'content': content}],
                'instructions': 'Return only the requested JSON object. Keep reasoning concise. '
                    'When transcribing telephone numbers, use ASCII hyphens for separators '
                    'and preserve every visible digit.',
                'max_output_tokens': 8192,
                'reasoning': {'effort': 'low'},
                'store': False,
            }, timeout=httpx.Timeout(remaining, connect=min(10, remaining)))
        except httpx.TimeoutException:
            raise RuntimeError('NEBIUS_TIMEOUT: cloud inference timed out.') from None
        except httpx.HTTPError:
            raise RuntimeError('NEBIUS_UNAVAILABLE: cloud connection failed.') from None
        if response.is_error:
            raise RuntimeError(f'NEBIUS_HTTP_{response.status_code}: cloud request failed.')
        try:
            payload = response.json()
            if payload.get('status') != 'completed' or payload.get('error'):
                raise ValueError('Incomplete cloud response')
            raw = '\n'.join(part['text'] for item in payload['output']
                if item.get('type') == 'message' and item.get('role') == 'assistant'
                for part in item['content'] if part.get('type') == 'output_text')
            if not raw.strip():
                raise ValueError('Incomplete cloud response')
            usage = payload.get('usage') or {}
            if not isinstance(usage, dict):
                raise ValueError('Invalid usage metadata')
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise RuntimeError('NEBIUS_INVALID_RESPONSE: missing or incomplete model output.') from None
        return raw, {
            'preprocessing_ms': 0, 'generation_ms': (perf_counter() - started) * 1000,
            'input_tokens': usage.get('input_tokens'), 'output_tokens': usage.get('output_tokens'),
            'adapter': {'provider': 'nebius', 'model': self.model},
        }

    def close(self):
        self.client.close()


class NebiusRuntime(LocalRuntime):
    device = 'cloud'

    def __init__(self, *, provider=None):
        model = os.environ.get('NEBIUS_MODEL', 'zai-org/GLM-5.3-Flash')
        self.provider = provider if provider is not None else NebiusProvider(
            api_key=os.environ.get('NEBIUS_API_KEY'), model=model,
            base_url=os.environ.get('NEBIUS_BASE_URL', 'https://api.tokenfactory.us-central1.nebius.com/v1/'),
        )
        self.spec = {'model_id': self.provider.model, 'family_alias': PROFILE, 'revision': 'provider-managed'}
        self.model_profile = PROFILE
        self.loaded = True
        self.gpu = None
        self.progress_request_id = None
        self.progress_stage = None

    @property
    def health_metadata(self):
        return {'default_model': PROFILE, 'inference_progress': {
            'request_id': self.progress_request_id, 'stage': self.progress_stage}, 'models': [
            {'id': PROFILE, 'name': f'Nebius · {self.provider.model}', 'available': True},
        ]}

    def report_stage(self, stage):
        self.progress_stage = stage

    def ensure_loaded(self):
        pass

    def gpu_memory(self):
        return None

    def infer_for_demo(self, path, user_request, guard_enabled=True):
        self.provider.deadline = perf_counter() + 240
        try:
            return self.infer(path, user_request, guard_enabled)
        finally:
            self.provider.deadline = None
