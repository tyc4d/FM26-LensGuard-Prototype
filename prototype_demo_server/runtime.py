"""Resident demo: isolated user task, scene transcription, cited selection."""
import importlib.metadata
import logging
import os
import subprocess
from time import perf_counter
from .perception import extract_scene
from .task_boundary import understand_task, select_evidence
from .native_demo import native_proposal

from physical_direct_local import LOCAL_MODELS
from providers.local import create_local_provider

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

    def infer_for_demo(self, path, user_request, guard_enabled=True):
        return self.infer(path, user_request, guard_enabled)

    def infer(self, path, user_request, guard_enabled=True):
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
        if not guard_enabled:
            native = native_proposal(self.provider, path, user_request)
            return {'raw_text': native['raw_text'], 'parsed_action': native['value'],
                    'candidate_action': native['value'], 'semantic_regions': [],
                    'diagnostics': {'parse_success': native['value'] is not None, 'error_message': native['error']},
                    'timing': {'inference_ms': native['elapsed_ms']},
                    'metadata': {'mode': 'unprotected_proposal', 'tools_available': False},
                    'unprotected': True}
        # User-only task, before pixels; all calls start a fresh chat context.
        task = understand_task(self.provider, user_request)
        scene = extract_scene(self.provider, path)
        selection = (select_evidence(self.provider, path, user_request, task['value'], scene['regions'])
                     if task['value'] is not None and scene['error'] is None
                     else {'value': None, 'raw_text': '', 'error': 'Task or perception unavailable', 'elapsed_ms': 0})
        elapsed = (perf_counter() - started) * 1000
        return {
            'raw_text': selection['raw_text'],
            'semantic_regions': scene['regions'],
            'parsed_action': None, 'candidate_action': None,
            'diagnostics': {'parse_success': selection['value'] is not None, 'error_message': selection['error']},
            'timing': {'task_ms': task['elapsed_ms'], 'perception_ms': scene['perception_ms'],
                       'selection_ms': selection['elapsed_ms'], 'inference_ms': elapsed},
            'boundary': {'task': task['value'], 'selection':
                {**selection['value'], 'operation': task['value']['operation'], 'kind': task['value']['kind']}
                if selection['value'] is not None else None},
            'metadata': {'task_interpretation': task, 'perception': scene, 'selection': selection,
                         'task_input': 'user_text_only', 'tools_available': False},
        }

    def close(self):
        if self.provider is not None:
            self.provider.close()
        self.loaded = False
