"""One recorded ON/OFF pass through the resident prototype; no evaluation labels in requests."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image, ImageOps

from physical_direct_contracts import TASK_PROMPTS

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'results_physical_pilot/reviewed_prototype_v1'
ARMS = ('guard_on', 'guard_off')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def exclusive(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())


def process_identity(pid):
    proc = Path('/proc') / str(pid)
    command = proc.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode().strip()
    if 'prototype_demo_server' not in command or proc.joinpath('cwd').resolve() != ROOT:
        raise ValueError('PID is not the prototype service in this checkout')
    # Split after the process name, which can itself contain spaces.
    fields = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'command': command, 'start_ticks': fields[19]}


def source_files():
    return sorted({*ROOT.glob('prototype_demo_server/*.py'), *ROOT.glob('providers/local/*.py'),
                   *ROOT.glob('firewall/*.py'), ROOT / 'physical_direct_local.py',
                   ROOT / 'physical_direct_contracts.py', ROOT / 'physical_reviewed_inference.py',
                   ROOT / 'config/physical_direct_prompts_v1.yaml'})


def prepare_image(data):
    """Both arms receive identical oriented derivatives within the resident API limits."""
    with Image.open(io.BytesIO(data)) as original:
        oriented = ImageOps.exif_transpose(original)
        original_size = oriented.size
        oriented.thumbnail((3840, 3840), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        oriented.convert('RGB').save(output, 'JPEG', quality=95, subsampling=0)
        payload = output.getvalue()
        assert len(payload) <= 10 * 1024 * 1024 and oriented.width * oriented.height <= 16_000_000
        return payload, {'oriented_original_size': original_size, 'input_size': oriented.size,
                         'input_bytes': len(payload), 'input_sha256': digest(payload)}


def prepare(client, base_url, pid, run=RUN):
    if (run / 'plan.json').exists():
        raise FileExistsError('Plan exists; use --resume for preserved trial identities')
    health = client.get(base_url + '/health').raise_for_status().json()
    if health['status'] != 'ready' or health['model_profile'] != 'qwen3vl-8b':
        raise ValueError('Expected the ready, resident Qwen3-VL 8B service')
    identity = process_identity(pid)
    source = ROOT / 'results_physical_pilot/direct_v1/input_manifest.json'
    manifest_bytes = source.read_bytes()
    manifest = json.loads(manifest_bytes)
    frozen = json.loads((run / 'evaluation_only/ground_truth_v1_manifest.json').read_bytes())
    annotations = (run / 'evaluation_only/ground_truth_v1.jsonl').read_bytes()
    if digest(annotations) != frozen['sha256'] or digest(manifest_bytes) != frozen['input_manifest_sha256']:
        raise ValueError('Frozen annotations and canonical image manifest differ')
    archive_path = ROOT / 'TestData.zip'
    if digest(archive_path.read_bytes()) != frozen['dataset_zip_sha256']:
        raise ValueError('Archive differs from frozen human annotation input')
    snapshots = {}
    for path in source_files():
        relative = path.relative_to(ROOT).as_posix()
        payload = path.read_bytes()
        snapshots[relative] = digest(payload)
        exclusive(run / 'runtime_snapshot' / relative, payload)
    exclusive(run / 'input_manifest.json', manifest_bytes)
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        for index, row in enumerate(manifest['records']):
            original = archive.read(row['archive_member'])
            if digest(original) != row['sha256']:
                raise ValueError('Original image hash mismatch: ' + row['image_id'])
            payload, details = prepare_image(original)
            relative = f'inputs/frame_{index + 1:03}.jpg'
            exclusive(run / relative, payload)
            # This allowlist deliberately excludes control labels, notes, and human answers.
            records.append({'image_id': row['image_id'], 'original_sha256': row['sha256'],
                            'scenario': row['scenario_family'], 'user_request': TASK_PROMPTS[row['scenario_family']],
                            'input_path': relative, **details,
                            'arm_order': list(ARMS if index % 2 == 0 else reversed(ARMS))})
    plan = {'experiment_id': 'physical-reviewed-prototype-v1', 'created_at': now(),
            'base_url': base_url, 'runtime_process': identity, 'health_before': health,
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'runtime_source_sha256': snapshots, 'input_manifest_sha256': digest(manifest_bytes),
            'annotation_sha256': digest(annotations), 'annotation_source_revision': frozen['source_revision'],
            'arm_definition': {'guard_on': 'current user-task-cited-evidence-v1 prototype pipeline',
                               'guard_off': 'current prototype native proposal, without guard'},
            'preprocessing': 'EXIF transpose; longest edge <=3840; JPEG quality=95, subsampling=0; identical per image in both arms',
            'decoding': 'resident pinned Qwen profile, greedy BF16, up to 1024 new tokens per stage',
            'scientific_attempts_per_image_arm': 1, 'semantic_retries': 0,
            'historical_direct_comparison': 'Separate results only: frozen DIRECT uses a different output contract and original inputs.',
            'labels_sent_to_model': False, 'external_tools_available': False,
            'trial_count': len(records) * 2, 'records': records}
    exclusive(run / 'plan.json', encoded(plan))
    return plan


def check_runtime(plan):
    if process_identity(plan['runtime_process']['pid']) != plan['runtime_process']:
        raise ValueError('Runtime process changed; audit before continuing')
    for relative, sha in plan['runtime_source_sha256'].items():
        if digest((ROOT / relative).read_bytes()) != sha:
            raise ValueError('Runtime source changed during experiment: ' + relative)


def request_trial(client, base_url, row, arm, image):
    # Do not add evaluation metadata or annotation content to this request.
    return client.post(base_url + '/v1/analyze',
                       files={'image': ('frame.jpg', image, 'image/jpeg')},
                       data={'user_request': row['user_request'], 'mode': 'action_only',
                             'guard_enabled': 'true' if arm == 'guard_on' else 'false'})


def trial_paths(plan, run, row, arm):
    relative = f"{arm}/{row['image_id']}.json"
    rejected = plan.get('recover_preflight_rejections', {}).get(relative)
    if rejected is not None:
        original = run / 'raw' / relative
        data = original.read_bytes()
        record = json.loads(data)
        if (digest(data) != rejected or record.get('http_status') != 503
                or not (record.get('response') or {}).get('detail', '').startswith(
                    'RUNTIME_UNAVAILABLE: GPU_MEMORY_INSUFFICIENT:')):
            raise ValueError('Only a preserved pre-inference VRAM rejection may be recovered')
        return run / 'raw_recovered' / relative, run / 'started_recovered' / relative
    return run / 'raw' / relative, run / 'started' / relative


def effective_plan(run=RUN):
    plan = json.loads((run / 'plan.json').read_bytes())
    continuation = run / 'continuation.json'
    if continuation.exists():
        update = json.loads(continuation.read_bytes())
        if update['parent_plan_sha256'] != digest((run / 'plan.json').read_bytes()):
            raise ValueError('Continuation is not bound to the immutable original plan')
        plan.update({key: update[key] for key in (
            'runtime_process', 'runtime_source_sha256', 'recover_preflight_rejections')})
    return plan


def run_trials(client, plan, run=RUN, limit=None):
    check_runtime(plan)
    done = 0
    for row in plan['records']:
        image = (run / row['input_path']).read_bytes()
        if digest(image) != row['input_sha256']:
            raise ValueError('Prepared input changed')
        for arm in row['arm_order']:
            raw, started = trial_paths(plan, run, row, arm)
            if raw.exists():
                preserved = json.loads(raw.read_bytes())
                if preserved['image_id'] != row['image_id'] or preserved['arm'] != arm:
                    raise ValueError('Preserved response identity differs')
                if preserved.get('transport_error') or preserved.get('http_status') != 200:
                    raise ValueError('Unsuccessful prior request; audit before resuming')
                continue
            if started.exists():
                raise ValueError('Started request without raw response; do not repeat: ' + str(started))
            check_runtime(plan)
            health = client.get(plan['base_url'] + '/health').raise_for_status().json()
            if health['status'] not in ('ready', 'unloaded'):
                raise ValueError('Resident service is not ready; no trial was sent')
            exclusive(started, encoded({'started_at': now(), 'image_id': row['image_id'], 'arm': arm,
                                       'input_sha256': row['input_sha256']}))
            record = {'image_id': row['image_id'], 'arm': arm, 'started_at': now(),
                      'input_sha256': row['input_sha256'], 'user_request': row['user_request']}
            begin = time.perf_counter()
            try:
                response = request_trial(client, plan['base_url'], row, arm, image)
                record.update(http_status=response.status_code, response_text=response.text,
                              response_headers=dict(response.headers))
                try:
                    record['response'] = response.json()
                except ValueError:
                    record['response'] = None
            except httpx.HTTPError as error:
                record['transport_error'] = type(error).__name__ + ': ' + str(error)
            record.update(completed_at=now(), client_elapsed_ms=(time.perf_counter() - begin) * 1000)
            exclusive(raw, encoded(record))
            result = record.get('response') or {}
            output = result.get('output') or {}
            policy = result.get('policy') or {}
            print(json.dumps({'image': row['image_id'], 'arm': arm, 'http': record.get('http_status'),
                              'seconds': round(record['client_elapsed_ms'] / 1000, 2),
                              'decision': policy.get('result'), 'rule': policy.get('rule_id'),
                              'action': output.get('proposed_action')}, ensure_ascii=False), flush=True)
            if record.get('http_status') != 200:
                raise RuntimeError('Request failed; raw evidence preserved; no automatic retry')
            if result.get('model') != {'profile': 'qwen3vl-8b',
                    'model_id': 'Qwen/Qwen3-VL-8B-Instruct', 'revision': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'}:
                raise ValueError('Response model identity differs from the pinned prototype')
            done += 1
            if limit is not None and done >= limit:
                return
    check_runtime(plan)
    paths = [trial_paths(plan, run, row, arm)[0] for row in plan['records'] for arm in row['arm_order']]
    if len(paths) != plan['trial_count']:
        raise ValueError('Incomplete trial count')
    completion = {'completed_at': now(), 'trials': len(paths), 'plan_sha256': digest((run / 'plan.json').read_bytes()),
                  'raw_sha256': {p.relative_to(run).as_posix(): digest(p.read_bytes()) for p in paths},
                  'prior_preflight_rejections': plan.get('recover_preflight_rejections', {}),
                  'health_after': client.get(plan['base_url'] + '/health').raise_for_status().json()}
    if (run / 'continuation.json').exists():
        completion['continuation_sha256'] = digest((run / 'continuation.json').read_bytes())
    if not (run / 'completion.json').exists():
        exclusive(run / 'completion.json', encoded(completion))
    print('COMPLETE: ' + str(len(paths)) + ' recorded ON/OFF trials', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-pid', type=int, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8010')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / '.run.running').open('a') as lock, httpx.Client(timeout=240) as client:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            plan = effective_plan()
            if args.base_url != plan['base_url'] or args.runtime_pid != plan['runtime_process']['pid']:
                raise ValueError('Resume must use the original runtime and endpoint')
        else:
            plan = prepare(client, args.base_url, args.runtime_pid)
        print('Plan ready: ' + str(plan['trial_count']) + ' trials; no ground truth in requests', flush=True)
        if not args.prepare_only:
            run_trials(client, plan, limit=args.limit)


if __name__ == '__main__':
    main()
