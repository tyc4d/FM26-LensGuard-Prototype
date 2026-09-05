from types import SimpleNamespace

import pytest

from prototype_demo_server.runtime import LocalRuntime


@pytest.mark.parametrize('fails', [False, True])
def test_request_releases_only_unused_cache_without_unloading_weights(fails):
    runtime = LocalRuntime()
    events = []
    provider = SimpleNamespace(_synchronize=lambda: events.append('sync'),
        _torch_module=lambda: SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: events.append('release'))))
    runtime.provider = provider
    runtime.loaded = True
    result = {'raw_text': 'preserved output'}
    def infer(*args):
        events.append('infer')
        if fails:
            raise RuntimeError('original failure')
        return result
    runtime.infer = infer
    if fails:
        with pytest.raises(RuntimeError, match='original failure'):
            runtime.infer_for_demo('image', 'request')
    else:
        assert runtime.infer_for_demo('image', 'request') is result
    assert events == ['infer', 'sync', 'release']
    assert runtime.provider is provider and runtime.loaded
