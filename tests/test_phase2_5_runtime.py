import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from system_info_phase2_5 import (
    collect_phase2_5_system_info,
    huggingface_cache_preflight,
    huggingface_cache_root,
    huggingface_model_cache_path,
    json_safe,
    resolve_cached_revision,
    resolve_component_revision,
    write_phase2_5_system_info,
)


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_name(index):
        assert index == 1
        return "Test GPU"

    @staticmethod
    def get_device_properties(index):
        assert index == 1
        return SimpleNamespace(total_memory=24 * 1024**3)

    @staticmethod
    def is_bf16_supported():
        return True


class _FakeNvml:
    NVML_TEMPERATURE_GPU = 0
    shutdown_called = False

    @classmethod
    def nvmlInit(cls):
        return None

    @classmethod
    def nvmlShutdown(cls):
        cls.shutdown_called = True

    @staticmethod
    def nvmlDeviceGetHandleByIndex(index):
        assert index == 1
        return "handle"

    @staticmethod
    def nvmlDeviceGetName(handle):
        return b"NVML Test GPU"

    @staticmethod
    def nvmlDeviceGetMemoryInfo(handle):
        return SimpleNamespace(total=25_000_000_000)

    @staticmethod
    def nvmlSystemGetDriverVersion():
        return b"610.00"

    @staticmethod
    def nvmlDeviceGetUtilizationRates(handle):
        return SimpleNamespace(gpu=42)

    @staticmethod
    def nvmlDeviceGetPowerUsage(handle):
        return 123_500

    @staticmethod
    def nvmlDeviceGetTemperature(handle, kind):
        return 57


def test_collect_system_info_uses_injected_lazy_modules_and_is_json_safe():
    fake_torch = SimpleNamespace(
        __version__="2.test",
        version=SimpleNamespace(cuda="12.test"),
        cuda=_FakeCuda(),
    )
    fake_transformers = SimpleNamespace(__version__="5.test")
    modules = {
        "torch": fake_torch,
        "transformers": fake_transformers,
        "pynvml": _FakeNvml,
    }

    def loader(name):
        return modules[name]

    def command_runner(command):  # pragma: no cover - must not be needed
        raise AssertionError(f"unexpected command: {command}")

    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="model-sha"))
    processor = SimpleNamespace(init_kwargs={"_commit_hash": "processor-sha"})
    info = collect_phase2_5_system_info(
        model_repository_id="org/model",
        model_revision="main",
        processor_revision="main",
        dtype="bf16",
        quantization="none",
        attention_backend="sdpa",
        device="cuda:1",
        model=model,
        processor=processor,
        module_loader=loader,
        command_runner=command_runner,
        captured_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
    )

    assert info["torch_version"] == "2.test"
    assert info["transformers_version"] == "5.test"
    assert info["cuda_runtime_visible_to_torch"] == "12.test"
    assert info["gpu_model"] == "Test GPU"
    assert info["vram_total_bytes"] == 24 * 1024**3
    assert info["nvidia_driver_version"] == "610.00"
    assert info["gpu_utilization_percent"] == 42.0
    assert info["gpu_power_draw_watts"] == 123.5
    assert info["gpu_temperature_celsius"] == 57.0
    assert info["model_revision"] == "model-sha"
    assert info["processor_revision"] == "processor-sha"
    assert _FakeNvml.shutdown_called is True
    json.dumps(info, allow_nan=False)


def test_collect_system_info_never_requires_optional_ml_or_nvidia_packages():
    def missing_loader(name):
        raise ImportError(f"missing {name}")

    def missing_command(command):
        raise FileNotFoundError(command[0])

    info = collect_phase2_5_system_info(
        model_repository_id="org/model",
        model_revision="pinned-model",
        processor_revision="pinned-processor",
        dtype="bf16",
        quantization="none",
        attention_backend="sdpa",
        module_loader=missing_loader,
        command_runner=missing_command,
    )

    assert info["torch_version"] is None
    assert info["transformers_version"] is None
    assert info["gpu_model"] is None
    assert info["nvml_available"] is False
    assert len(info["metadata_warnings"]) >= 3
    json.dumps(info, allow_nan=False)


def test_huggingface_cache_resolution_and_disk_preflight(tmp_path):
    explicit = tmp_path / "explicit-hub"
    assert huggingface_cache_root(
        environ={
            "HF_HUB_CACHE": str(explicit),
            "HUGGINGFACE_HUB_CACHE": str(tmp_path / "legacy"),
            "HF_HOME": str(tmp_path / "home"),
        }
    ) == explicit
    assert huggingface_cache_root(
        environ={"HF_HOME": str(tmp_path / "hf-home")}
    ) == tmp_path / "hf-home" / "hub"

    model_path = huggingface_model_cache_path("org/model", cache_root=explicit)
    assert model_path == explicit / "models--org--model"
    (model_path / "refs").mkdir(parents=True)
    (model_path / "snapshots" / "abc123").mkdir(parents=True)
    (model_path / "blobs").mkdir()
    (model_path / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    (model_path / "blobs" / "one").write_bytes(b"12345")

    usage = SimpleNamespace(total=1_000, used=250, free=750)
    result = huggingface_cache_preflight(
        "org/model",
        cache_root=explicit,
        estimated_download_bytes=500,
        reserve_bytes=100,
        disk_usage=lambda path: usage,
    )
    assert resolve_cached_revision("org/model", cache_root=explicit) == "abc123"
    assert result["cached_revision"] == "abc123"
    assert result["model_cache_exists"] is True
    assert result["model_cache_bytes"] == 5
    assert result["sufficient_free_space"] is True
    assert result["post_download_free_bytes"] == 150


@pytest.mark.parametrize("repo_id", ["model", "../model", "org/../model", "org/"])
def test_huggingface_model_cache_path_rejects_unsafe_ids(repo_id):
    with pytest.raises(ValueError, match="repository ID"):
        huggingface_model_cache_path(repo_id)


def test_json_safe_and_atomic_system_info_write(tmp_path):
    class Scalar:
        @staticmethod
        def item():
            return 7

    value = json_safe({"nan": float("nan"), "scalar": Scalar(), "path": tmp_path})
    assert value == {"nan": None, "scalar": 7, "path": str(tmp_path)}

    destination = tmp_path / "run" / "system_info.json"
    write_phase2_5_system_info(destination, value)
    assert json.loads(destination.read_text(encoding="utf-8")) == value
    assert not destination.with_suffix(".json.tmp").exists()


def test_resolve_component_revision_prefers_loaded_commit():
    component = SimpleNamespace(
        _commit_hash=None,
        config=SimpleNamespace(_commit_hash="resolved-sha"),
        init_kwargs={"_commit_hash": "secondary-sha"},
    )
    assert resolve_component_revision(component, "main") == "resolved-sha"
    assert resolve_component_revision(None, "fallback-sha") == "fallback-sha"
