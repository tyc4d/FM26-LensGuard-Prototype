import io

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


def test_gemma_runtime_constructs_and_loads_once(monkeypatch):
    from types import SimpleNamespace
    from prototype_demo_server import runtime as module
    constructed = []
    class Provider:
        model_revision = processor_revision = module.SPEC['revision']
        loads = 0
        def load(self): self.loads += 1
        def close(self): pass
    provider = Provider()
    def factory(alias, **kwargs):
        assert alias == 'gemma3-4b' and kwargs['max_new_tokens'] == 1024
        assert kwargs['revision'] == module.SPEC['revision']
        constructed.append(alias)
        return provider
    def invoke(actual, **kwargs):
        assert actual is provider and kwargs['operation'] is Phase35Operation.ACTION_ONLY
        assert kwargs['user_prompt'] == 'trusted task'
        parsed, payload, diagnostics = _parse_output(Phase35Operation.ACTION_ONLY, FakeAdapter.raw)
        return SimpleNamespace(parsed=parsed, json_payload=payload, raw_response=FakeAdapter.raw, diagnostics=diagnostics, latency_ms=1,
            response_metadata={'local_inference': {'generation_latency_ms': 1}})
    monkeypatch.setattr(module, 'gpu_preflight', lambda loaded: {})
    monkeypatch.setattr(module, 'create_local_provider', factory)
    monkeypatch.setattr(module, 'invoke_phase3_5', invoke)
    monkeypatch.setattr(module.importlib.metadata, 'version', lambda package: {'torch':'2.10.0+cu128','transformers':'5.16.1'}[package])
    runtime = module.GemmaRuntime()
    runtime.infer('fixture.png', 'trusted task')
    runtime.infer('fixture.png', 'trusted task')
    assert constructed == ['gemma3-4b'] and provider.loads == 1


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
