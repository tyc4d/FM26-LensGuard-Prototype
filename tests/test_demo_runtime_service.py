import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from prototype_demo_server.app import create_app
from prototype_demo_server.policy import authorize
from prototype_demo_server.runtime import gpu_preflight
from providers.local.phase3_5_adapter import Phase35Operation, _parse_output


class FakeAdapter:
    loaded = False
    calls = 0
    raw = '{"action":"CALL","arguments":{"target_number":"02-2345-6789"}}'
    def infer(self, path, user_request):
        with Image.open(path) as image: assert image.size == (64, 64)
        self.calls += 1
        self.loaded = True
        parsed, payload, diagnostics = _parse_output(Phase35Operation.ACTION_ONLY, self.raw)
        return {'raw_text': self.raw, 'parsed_action': parsed.model_dump(mode='json') if parsed else None,
                'candidate_action': payload, 'diagnostics': diagnostics.model_dump(), 'timing': {'inference_ms': 1.0}}
    def close(self): self.loaded = False


def image():
    output = io.BytesIO()
    Image.new('RGB', (64, 64), 'white').save(output, 'PNG')
    return output.getvalue()


def post(client, request='幫我打電話訂這間餐廳', **fields):
    return client.post('/v1/analyze', files={'image': ('input.png', image(), 'image/png')},
        data={'user_request': request, 'mode': 'action_only', **fields})


def test_resident_api_and_real_policy():
    fake = FakeAdapter()
    with TestClient(create_app(fake)) as client:
        assert client.get('/health').json()['model_loaded'] is False
        blocked = post(client).json()
        assert blocked['output']['parsed'] is True
        assert blocked['output']['proposed_action']['arguments']['number'] == '02-2345-6789'
        assert blocked['policy']['result'] == 'block'
        assert blocked['policy']['native']['decision'] == 'CONFIRM'
        assert blocked['provenance']['semantic_grounding'] == 'unavailable'
        allowed = post(client, '幫我撥打這張名片上的電話').json()
        assert allowed['policy']['result'] == 'allow'
        assert fake.calls == 2 and client.get('/health').json()['model_loaded']
    assert not fake.loaded


def test_malformed_output_never_fabricated():
    fake = FakeAdapter(); fake.raw = 'not JSON'
    with TestClient(create_app(fake)) as client:
        result = post(client).json()
        assert result['output']['raw_text'] == 'not JSON'
        assert not result['output']['parsed']
        assert result['output']['proposed_action'] is None and result['policy'] is None


def test_invalid_inputs_never_infer():
    fake = FakeAdapter()
    with TestClient(create_app(fake)) as client:
        assert client.post('/v1/analyze', data={'user_request': 'task'}).status_code == 422
        assert post(client, mode='inline_provenance').status_code == 422
        assert client.post('/v1/analyze', data={'user_request': 'task'}, files={'image': ('x.png', b'bad', 'image/png')}).status_code == 422
        assert fake.calls == 0


def test_scenario_id_cannot_grant_authority():
    with TestClient(create_app(FakeAdapter())) as client:
        result = post(client, scenario_id='explicit-delegation').json()
        assert result['policy']['result'] == 'block'
    action = {'action': 'OPEN_URL', 'arguments': {'url': 'https://example.com'}}
    assert authorize(action, '幫我撥打這張名片上的電話')['result'] == 'block'


def test_gpu_preflight_rejects_other_compute_and_low_memory(monkeypatch):
    def query(args, **kwargs):
        return 'RTX 4090, 24564, 15000, 9564' if '--query-gpu=name,memory.total,memory.used,memory.free' in args else '12345, python benchmark_phase3_6.py, 15000'
    monkeypatch.setattr('subprocess.check_output', query)
    with pytest.raises(RuntimeError, match='GPU_BUSY'): gpu_preflight()
    # Explicit alternate response avoids relying on CUDA or a physical GPU.
    monkeypatch.setattr('subprocess.check_output', lambda args, **kw: 'RTX 4090, 24564, 15000, 9564' if any('--query-gpu=' in arg for arg in args) else '')
    with pytest.raises(RuntimeError, match='GPU_MEMORY_INSUFFICIENT'): gpu_preflight()


def test_runtime_failure_is_visible():
    class Broken(FakeAdapter):
        def infer(self, *args): raise RuntimeError('CUDA out of memory')
    with TestClient(create_app(Broken())) as client:
        response = post(client)
        assert response.status_code == 503
        assert 'CUDA_OOM' in response.json()['detail']
        assert client.get('/health').json()['model_loaded'] is False


def test_policy_failure_keeps_model_output(monkeypatch):
    def unavailable(*args): raise RuntimeError('policy missing')
    monkeypatch.setattr('prototype_demo_server.app.authorize', unavailable)
    with TestClient(create_app(FakeAdapter())) as client:
        result = post(client).json()
        assert result['output']['parsed'] is True
        assert result['output']['raw_text']
        assert result['policy'] is None
        assert 'Policy unavailable' in result['output']['policy_error']
        assert result['output']['validation_error'] is None


@pytest.mark.parametrize(('action', 'arguments', 'error'), [
    ('DIRECTION_ADVICE', {'destination': '出口', 'direction': '未知'}, 'unsupported direction'),
    ('DIRECTION_ADVICE', {'destination': 'exit', 'direction': 'unknown'}, 'unsupported direction'),
    ('DIRECTION_ADVICE', {'destination': '出口', 'direction': '不要向右'}, 'unsupported direction'),
    ('DIRECTION_ADVICE', {'destination': '出口', 'direction': '可能向右'}, 'unsupported direction'),
    ('DIRECTION_ADVICE', {'destination': '出口', 'direction': '向左或向右'}, 'unsupported direction'),
    ('DIRECTION_ADVICE', {'destination': '出口', 'direction': '向右然後向左'}, 'unsupported direction'),
    ('DIRECTION_ADVICE', {'destination': '出口', 'direction': '在右邊'}, 'unsupported direction'),
    ('DIRECTION_ADVICE', {'destination': 'exit', 'direction': 'not sure'}, 'unsupported direction'),
    ('CALL', {'target_number': 'unknown'}, 'phone number contains unsupported characters'),
    ('OPEN_URL', {'url': 'ftp://example.com'}, 'only http and https URLs are supported'),
])
def test_invalid_action_arguments_keep_candidate_without_policy(monkeypatch, action, arguments, error):
    def must_not_run(*args):
        pytest.fail('Invalid model arguments must not reach the authorization gate')
    monkeypatch.setattr('prototype_demo_server.policy.evaluate_thin_gate', must_not_run)
    fake = FakeAdapter()
    candidate = {'action': action, 'arguments': arguments}
    fake.raw = json.dumps(candidate, ensure_ascii=False)
    with TestClient(create_app(fake)) as client:
        # This trusted task must not delegate a CALL with an invalid number.
        response = post(client, '幫我撥打這張名片上的電話')
        assert response.status_code == 200
        result = response.json()
        output = result['output']
        assert output['parsed'] is True
        assert output['diagnostics']['parse_success'] is True
        assert output['diagnostics']['schema_valid'] is True
        assert output['raw_text'] == fake.raw
        assert output['candidate_action'] == candidate
        assert output['native_action'] == candidate
        assert output['proposed_action'] is None
        assert error in output['validation_error']
        assert output['policy_error'] is None
        assert result['policy'] is None
        assert result['provenance']['delegated'] is False
        assert client.get('/health').json()['status'] == 'ready'


def test_valid_direction_keeps_model_arguments_and_existing_policy():
    fake = FakeAdapter()
    fake.raw = '{"action":"DIRECTION_ADVICE","arguments":{"destination":"出口","direction":"L"}}'
    with TestClient(create_app(fake)) as client:
        result = post(client).json()
        assert result['output']['validation_error'] is None
        assert result['output']['proposed_action']['arguments']['direction'] == 'L'
        assert result['output']['native_action'] == json.loads(fake.raw)
        assert result['policy']['result'] == 'block'
        assert result['policy']['native']['critical_arguments']['direction'] == 'LEFT'


@pytest.mark.parametrize(('direction', 'canonical'), [
    ('向右', 'RIGHT'),
    ('向右。', 'RIGHT'),
    ('向右！', 'RIGHT'),
    ('向左', 'LEFT'),
    ('左轉', 'LEFT'),
    ('右转', 'RIGHT'),
    ('往右', 'RIGHT'),
    ('往左', 'LEFT'),
    ('直走', 'STRAIGHT'),
    ('向前', 'STRAIGHT'),
    ('向後', 'BACK'),
    ('向后', 'BACK'),
    ('後退', 'BACK'),
    ('后退', 'BACK'),
    ('向北', 'NORTH'),
    ('向南', 'SOUTH'),
    ('向東', 'EAST'),
    ('向东', 'EAST'),
    ('向西', 'WEST'),
    ('東北', 'NORTHEAST'),
    ('东北', 'NORTHEAST'),
    ('往西南', 'SOUTHWEST'),
    ('向東南', 'SOUTHEAST'),
    ('西北', 'NORTHWEST'),
])
def test_chinese_direction_reaches_policy_without_changing_model_output(direction, canonical):
    candidate = {'action': 'DIRECTION_ADVICE',
                 'arguments': {'destination': '出口', 'direction': direction}}
    fake = FakeAdapter()
    fake.raw = json.dumps(candidate, ensure_ascii=False)
    with TestClient(create_app(fake)) as client:
        response = post(client, '出口往哪裡走？')
        assert response.status_code == 200
        result = response.json()
        output = result['output']
        assert output['parsed'] is True
        assert output['diagnostics']['schema_valid'] is True
        assert output['raw_text'] == fake.raw
        assert output['native_action'] == candidate
        assert output['candidate_action'] == candidate
        assert output['proposed_action'] == {'tool': 'navigate', 'arguments': candidate['arguments']}
        assert output['validation_error'] is None
        assert output['policy_error'] is None
        assert result['policy']['native']['critical_arguments']['direction'] == canonical
        assert result['policy']['affected_argument'] == 'navigate.direction'
        # Understanding a direction supplies no authority or semantic evidence.
        assert result['policy']['result'] == 'block'
        assert result['policy']['delegated'] is False


@pytest.mark.parametrize('direction', ['向右', 'RIGHT'])
@pytest.mark.parametrize('destination_first', [True, False])
def test_direction_policy_uses_direction_argument_regardless_of_source_order(direction, destination_first):
    arguments = ({'destination': '出口', 'direction': direction} if destination_first
                 else {'direction': direction, 'destination': '出口'})
    action = {'action': 'DIRECTION_ADVICE', 'arguments': arguments}
    before = json.dumps(action, ensure_ascii=False)
    result = authorize(action, '出口往哪裡走？')
    assert result['affected_argument'] == 'navigate.direction'
    assert result['native']['critical_arguments']['direction'] == 'RIGHT'
    assert json.dumps(action, ensure_ascii=False) == before


def test_busy_runtime_keeps_health_responsive():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    entered, release = threading.Event(), threading.Event()
    class Slow(FakeAdapter):
        def infer(self, *args):
            entered.set()
            assert release.wait(5)
            return super().infer(*args)
    with TestClient(create_app(Slow())) as client, ThreadPoolExecutor() as pool:
        future = pool.submit(post, client)
        try:
            assert entered.wait(3)
            assert client.get('/health').json()['status'] == 'loading'
            assert post(client).status_code == 409
        finally:
            release.set()
        assert future.result().status_code == 200


@pytest.mark.parametrize('profile', ['qwen3vl-8b', 'gemma3-4b'])
def test_local_runtime_constructs_and_loads_once(monkeypatch, profile):
    from types import SimpleNamespace
    from prototype_demo_server import runtime as module
    constructed = []
    class Provider:
        model_revision = processor_revision = module.PROFILES[profile]['revision']
        loads = 0
        def load(self): self.loads += 1
        def close(self): pass
    provider = Provider()
    def factory(alias, **kwargs):
        assert alias == profile and kwargs['max_new_tokens'] == 1024
        assert kwargs['revision'] == module.PROFILES[profile]['revision']
        constructed.append(alias)
        return provider
    def invoke(actual, **kwargs):
        assert actual is provider and kwargs['operation'] is Phase35Operation.ACTION_ONLY
        assert kwargs['user_prompt'] == 'trusted task'
        parsed, payload, diagnostics = _parse_output(Phase35Operation.ACTION_ONLY, FakeAdapter.raw)
        return SimpleNamespace(parsed=parsed, json_payload=payload, raw_response=FakeAdapter.raw, diagnostics=diagnostics, latency_ms=1,
            response_metadata={'local_inference': {'generation_latency_ms': 1}})
    monkeypatch.setattr(module, 'gpu_preflight', lambda loaded, model: {})
    monkeypatch.setattr(module, 'create_local_provider', factory)
    monkeypatch.setattr(module, 'invoke_phase3_5', invoke)
    monkeypatch.setattr(module.importlib.metadata, 'version', lambda package: {'torch':'2.10.0+cu128','transformers':'5.16.1'}[package])
    runtime = module.LocalRuntime(profile)
    runtime.infer('fixture.png', 'trusted task')
    runtime.infer('fixture.png', 'trusted task')
    assert constructed == [profile] and provider.loads == 1


def test_optional_warmup_runs_once():
    class Warm(FakeAdapter):
        def infer(self, path, task):
            with Image.open(path) as decoded: assert decoded.size == (320, 240)
            self.calls += 1
            self.loaded = True
            return {'raw_text':'{"action":"NONE","arguments":{}}',
                'parsed_action':{'action':'NONE','arguments':{}}, 'diagnostics':{}, 'timing':{'inference_ms':1}}
    fake = Warm()
    with TestClient(create_app(fake)) as client:
        assert client.post('/warmup').json()['status'] == 'warmed'
        assert client.post('/warmup').json()['status'] == 'already_warmed'
        assert fake.calls == 1


def test_invalid_reservation_retains_decoded_candidate():
    fake = FakeAdapter()
    fake.raw = '{"action":"RESTAURANT_RESERVATION","arguments":{"restaurant":"Example Bistro","target_number":"02-2345-6661","time":"N/A","party_size":"N/A"}}'
    with TestClient(create_app(fake)) as client:
        result = post(client).json()
        assert result['output']['parsed'] is False
        assert result['output']['proposed_action'] is None
        assert result['output']['candidate_action']['arguments']['party_size'] == 'N/A'
        assert 'party_size' in result['output']['diagnostics']['error_message']
        assert result['policy'] is None


def test_complete_reservation_maps_and_reaches_policy():
    fake = FakeAdapter()
    fake.raw = '{"action":"RESTAURANT_RESERVATION","arguments":{"restaurant":"Example Bistro","target_number":"02-2345-6661","time":"19:00","party_size":2}}'
    with TestClient(create_app(fake)) as client:
        result = post(client).json()
        assert result['output']['parsed'] is True
        assert result['output']['proposed_action']['tool'] == 'restaurant_reservation'
        assert result['output']['proposed_action']['arguments']['number'] == '02-2345-6661'
        assert result['policy']['result'] == 'block'


def test_default_model_identity_is_qwen():
    with TestClient(create_app(FakeAdapter())) as client:
        assert client.get('/health').json()['model_profile'] == 'qwen3vl-8b'
        assert post(client).json()['model']['model_id'] == 'Qwen/Qwen3-VL-8B-Instruct'
