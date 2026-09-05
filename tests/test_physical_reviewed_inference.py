"""Transport isolation and one-attempt evidence preservation; no live model calls."""
import io
import json

import httpx
import pytest
from PIL import Image

import physical_reviewed_inference as runner


def test_input_derivative_retains_orientation_and_fits_api():
    image = Image.new('RGB', (4500, 3000), 'blue')
    exif = image.getexif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG', exif=exif)
    source = buffer.getvalue()
    derivative, details = runner.prepare_image(source)
    with Image.open(io.BytesIO(derivative)) as decoded:
        assert decoded.size == (2560, 3840)
        assert decoded.getexif().get(274, 1) == 1
    assert details['oriented_original_size'] == (3000, 4500)
    assert details['input_sha256'] == runner.digest(derivative)
    assert source == buffer.getvalue()


def test_transport_has_only_pixels_task_and_arm_not_annotation_fields():
    calls = []
    class Client:
        def post(self, path, **kwargs):
            calls.append((path, kwargs))
    row = {'user_request': 'User-authorized task', 'image_id': 'secret ID',
           'notes': 'secret label', 'ground_truth_value': 'secret answer',
           'attack_mode': 'OVERLAY', 'attacker_value': 'secret attacker value'}
    for arm in runner.ARMS:
        runner.request_trial(Client(), 'http://localhost', row, arm, b'same pixels')
    for path, kwargs in calls:
        assert path.endswith('/v1/analyze')
        assert kwargs['files'] == {'image': ('frame.jpg', b'same pixels', 'image/jpeg')}
        assert set(kwargs['data']) == {'user_request', 'mode', 'guard_enabled'}
        assert 'secret' not in json.dumps(kwargs['data'])
    assert calls[0][1]['data']['guard_enabled'] == 'true'
    assert calls[1][1]['data']['guard_enabled'] == 'false'


def fake_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'check_runtime', lambda plan: None)
    (tmp_path / 'image.jpg').write_bytes(b'pixels')
    row = {'image_id': 'one', 'input_path': 'image.jpg', 'input_sha256': runner.digest(b'pixels'),
           'user_request': 'task', 'arm_order': list(runner.ARMS)}
    plan = {'records': [row], 'base_url': 'http://test', 'trial_count': 2}
    (tmp_path / 'plan.json').write_bytes(runner.encoded(plan))
    return plan


def test_resume_preserves_completed_trials_and_rejects_uncertain_attempts(tmp_path, monkeypatch):
    plan = fake_plan(tmp_path, monkeypatch)
    posts = []
    def transport(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'status': 'ready'})
        posts.append(request.content)
        return httpx.Response(200, json={'model': {'profile': 'qwen3vl-8b',
            'model_id': 'Qwen/Qwen3-VL-8B-Instruct', 'revision': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'},
            'output': {'raw_text': 'unrepaired model output'}, 'policy': None})
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        runner.run_trials(client, plan, tmp_path, limit=1)
        original = (tmp_path / 'raw/guard_on/one.json').read_bytes()
        marker = tmp_path / 'started/guard_off/one.json'
        runner.exclusive(marker, b'{}')
        with pytest.raises(ValueError, match='Started request'):
            runner.run_trials(client, plan, tmp_path)
        assert len(posts) == 1
        assert (tmp_path / 'raw/guard_on/one.json').read_bytes() == original


def test_completed_run_does_not_generate_again(tmp_path, monkeypatch):
    plan = fake_plan(tmp_path, monkeypatch)
    posts = []
    def transport(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'status': 'ready'})
        posts.append(request.content)
        return httpx.Response(200, json={'model': {'profile': 'qwen3vl-8b',
            'model_id': 'Qwen/Qwen3-VL-8B-Instruct', 'revision': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'},
            'output': {}, 'policy': None})
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        runner.run_trials(client, plan, tmp_path)
        completion = (tmp_path / 'completion.json').read_bytes()
        runner.run_trials(client, plan, tmp_path)
    assert len(posts) == 2
    assert (tmp_path / 'completion.json').read_bytes() == completion


def test_only_pre_generation_memory_rejection_can_be_recovered(tmp_path):
    row = {'image_id': 'one'}
    payload = runner.encoded({'http_status': 503, 'response': {
        'detail': 'RUNTIME_UNAVAILABLE: GPU_MEMORY_INSUFFICIENT: qwen3vl-8b requires 4,096 MiB free.'}})
    original = tmp_path / 'raw/guard_off/one.json'
    runner.exclusive(original, payload)
    plan = {'recover_preflight_rejections': {'guard_off/one.json': runner.digest(payload)}}
    raw, marker = runner.trial_paths(plan, tmp_path, row, 'guard_off')
    assert raw == tmp_path / 'raw_recovered/guard_off/one.json'
    assert marker == tmp_path / 'started_recovered/guard_off/one.json'
    assert original.read_bytes() == payload
    other = runner.encoded({'http_status': 200, 'response': {'output': {'parsed': False}}})
    original.write_bytes(other)
    plan['recover_preflight_rejections']['guard_off/one.json'] = runner.digest(other)
    with pytest.raises(ValueError, match='pre-inference VRAM rejection'):
        runner.trial_paths(plan, tmp_path, row, 'guard_off')
