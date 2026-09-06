import subprocess

import pytest
from fastapi.testclient import TestClient

from prototype_demo_server.app import create_app
from prototype_demo_server.runtime import LocalRuntime


def test_health_refreshes_device_memory_without_loading_the_model(monkeypatch):
    runtime = LocalRuntime()
    runtime.gpu = {'used_mib': 15}  # Previous inference preflight is not current usage.
    readings = iter(['RTX 4090, 24564, 16000', 'RTX 4090, 24564, 17000'])

    def query(args, **kwargs):
        assert '--query-gpu=name,memory.total,memory.used' in args
        assert kwargs['timeout'] == 1
        return next(readings)

    monkeypatch.setattr('subprocess.check_output', query)
    monkeypatch.setattr('prototype_demo_server.runtime.gpu_preflight',
                        lambda *args: pytest.fail('Health must not run inference preflight'))
    with TestClient(create_app(runtime)) as client:
        first = client.get('/health').json()
        second = client.get('/health').json()
        assert first['model_id'] == 'Qwen/Qwen3-VL-8B-Instruct'
        assert first['gpu_memory'] == {'name': 'RTX 4090', 'total_mib': 24564, 'used_mib': 16000}
        assert second['gpu_memory']['used_mib'] == 17000
        assert second['gpu']['used_mib'] == 15
        assert second['status'] == 'unloaded'
        assert not second['model_loaded']
        assert runtime.provider is None


@pytest.mark.parametrize('failure', [
    FileNotFoundError('nvidia-smi'),
    subprocess.TimeoutExpired('nvidia-smi', 1),
    subprocess.CalledProcessError(1, 'nvidia-smi'),
])
def test_gpu_telemetry_failure_does_not_break_health(monkeypatch, failure):
    def query(*args, **kwargs):
        raise failure

    monkeypatch.setattr('subprocess.check_output', query)
    with TestClient(create_app(LocalRuntime('gemma3-4b'))) as client:
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json()['model_id'] == 'google/gemma-3-4b-it'
        assert response.json()['gpu_memory'] is None
        assert response.json()['status'] == 'unloaded'


@pytest.mark.parametrize('row', ['RTX 4090, [N/A], [N/A]', 'RTX 4090, 0, 0',
                                  'RTX 4090, 24564, -1', 'RTX 4090, 24564, 25000'])
def test_invalid_device_reading_is_unavailable(monkeypatch, row):
    monkeypatch.setattr('subprocess.check_output', lambda *args, **kwargs: row)
    assert LocalRuntime().gpu_memory() is None
