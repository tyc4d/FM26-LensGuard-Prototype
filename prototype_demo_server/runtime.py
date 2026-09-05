"""Resident adapter using the frozen Phase 3.5 action-only invocation."""
import importlib.metadata
import logging
import os
import subprocess
from time import perf_counter

from physical_direct_local import LOCAL_MODELS
from providers.local import create_local_provider
from providers.local.phase3_5_adapter import Phase35Operation, invoke_phase3_5

log = logging.getLogger(__name__)
PROFILES = {LOCAL_MODELS[key]['family_alias']: LOCAL_MODELS[key] for key in ('gemma', 'qwen')}
DEFAULT_MODEL = 'qwen3vl-8b'
SPEC = PROFILES[DEFAULT_MODEL]


def gpu_preflight(loaded=False, model=DEFAULT_MODEL):
    """Read-only, fail-closed check. Never reclaim another process's GPU memory."""
    def query(*args):
        return subprocess.check_output(['nvidia-smi', *args], text=True, timeout=10).strip()
    row = query('--id=0', '--query-gpu=name,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits')
    name, total, used, free = [part.strip() for part in row.split(',')]
    processes = query('--id=0', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits')
    snapshot = dict(name=name, total_mib=int(total), used_mib=int(used), free_mib=int(free), processes=processes, model_profile=model)
    log.info('GPU preflight: %s', snapshot)
    foreign = [line for line in processes.splitlines() if line.split(',')[0].strip() != str(os.getpid())]
    if foreign:
        raise RuntimeError('GPU_BUSY: another compute process is active. Phase 3.6 is never interrupted.')
    required = 4096 if loaded else (21000 if model == 'qwen3vl-8b' else 14000)
    if int(free) < required:
        raise RuntimeError(f'GPU_MEMORY_INSUFFICIENT: {model} requires {required:,} MiB free.')
    return snapshot


class LocalRuntime:
    def __init__(self, model=DEFAULT_MODEL):
        self.spec = PROFILES[model]
        self.model_profile = model
        self.provider = None
        self.loaded = False
        self.gpu = None

    def infer(self, path, user_request):
        self.gpu = gpu_preflight(self.loaded, self.model_profile)
        if not self.loaded:
            for package, expected in [('torch', '2.10.0+cu128'), ('transformers', self.spec['transformers_version'])]:
                if importlib.metadata.version(package) != expected:
                    raise RuntimeError(f'RUNTIME_MISMATCH: {package} must remain {expected}')
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            # Same offline enforcement and pinned profile used by physical DIRECT.
            from physical_direct_local import _force_offline
            _force_offline()
            if self.provider is None:
                self.provider = create_local_provider(self.model_profile, revision=self.spec['revision'], max_new_tokens=1024, device='cuda', enable_nvml=True)
            self.provider.load()
            if self.provider.model_revision != self.spec['revision'] or self.provider.processor_revision != self.spec['revision']:
                raise RuntimeError('REVISION_MISMATCH: refusing a different model/processor revision')
            self.loaded = True
        started = perf_counter()
        invocation = invoke_phase3_5(self.provider, operation=Phase35Operation.ACTION_ONLY, user_prompt=user_request, image_path=path)
        elapsed = (perf_counter() - started) * 1000
        metadata = invocation.response_metadata['local_inference']
        return {
            'raw_text': invocation.raw_response or '',
            'parsed_action': invocation.parsed.model_dump(mode='json') if invocation.parsed else None,
            'candidate_action': invocation.json_payload,
            'diagnostics': invocation.diagnostics.model_dump(),
            'timing': {'inference_ms': invocation.latency_ms, 'generation_ms': metadata['generation_latency_ms'], 'parsing_and_metadata_ms': max(0, elapsed - invocation.latency_ms)},
            'metadata': metadata,
        }

    def close(self):
        if self.provider is not None:
            self.provider.close()
        self.loaded = False
