"""DIRECT adapter tests use generated software fixtures and never load weights."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from physical_direct_local import LOCAL_MODELS, LocalDirectProvider
from providers.base_cloud_vlm import CloudRequest


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Family:
    def __init__(self, alias, clock, *, failure=None, raw_text="malformed model output"):
        spec = LOCAL_MODELS[alias]
        self.model_revision = self.processor_revision = self.tokenizer_revision = spec["revision"]
        self.model_load_time_ms = 50000
        self.clock, self.failure, self.raw_text = clock, failure, raw_text
        self.load_count = self.generation_count = self.close_count = 0
        self.inference_modes = 0
        self.prompt = self.image = None
        self.sync_count = 0

    def load(self):
        self.load_count += 1
        self.clock.now += 50
        if self.failure == "load":
            raise RuntimeError("weights unavailable offline")

    def _prepare_input(self, prompt, image):
        self.prompt, self.image = prompt, image.copy()
        self.clock.now += 0.25
        return SimpleNamespace(
            input_token_count=None if self.failure == "unknown_usage" else 100,
            processed_image_width=32, processed_image_height=48,
            metadata={"chat_template_adapter": "unchanged-family-adapter"},
        )

    def _synchronize(self):
        self.sync_count += 1
        if self.failure == "post_generation" and self.sync_count == 2:
            raise RuntimeError("CUDA synchronization failed")

    def _reset_peak_memory(self):
        pass

    def _torch_module(self):
        def inference_mode():
            self.inference_modes += 1
            return nullcontext()

        return SimpleNamespace(inference_mode=inference_mode)

    def _generate(self, prepared):
        self.generation_count += 1
        self.clock.now += 2
        if self.failure == "oom":
            raise RuntimeError("CUDA out of memory")
        return SimpleNamespace(
            raw_text=self.raw_text, output_token_count=12,
            metadata={"generation_mode": "unchanged-family-generation"},
        )

    def _cuda_memory(self, name):
        assert name == "max_memory_allocated"
        return 5 * 1024**2

    def close(self):
        self.close_count += 1


@pytest.fixture(autouse=True)
def isolate_offline_env(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", os.getenv("HF_HUB_OFFLINE", "0"))
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", os.getenv("TRANSFORMERS_OFFLINE", "0"))


@pytest.fixture
def request_input(tmp_path):
    path = tmp_path / "software-fixture.jpg"
    image = Image.new("RGB", (40, 20), "red")
    image.paste("blue", (20, 0, 40, 20))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, quality=100, subsampling=0, exif=exif)
    return CloudRequest(path, "Exactly this task; no evidence registry.", {}, "CASE-1", "DIRECT")


def adapter(alias="gemma", **kwargs):
    clock = Clock()
    family = Family(alias, clock, **kwargs)
    factory = Mock(return_value=family)
    provider = LocalDirectProvider(
        alias, provider_factory=factory, clock=clock,
        utcnow=lambda: datetime(2026, 9, 5, tzinfo=timezone.utc),
    )
    return provider, family, factory


@pytest.mark.parametrize("alias", ["gemma", "minicpm", "qwen"])
def test_exact_frozen_models_configs_and_family_factory(alias, request_input):
    provider, family, factory = adapter(alias)
    spec = LOCAL_MODELS[alias]
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads((root / "results_phase3_5/grounded-provenance-v1"
                         / spec["family_alias"] / "system_info.json").read_text())
    assert spec["model_id"] == frozen["model_repository_id"]
    assert spec["revision"] == frozen["model_revision"] == frozen["processor_revision"]
    assert spec["transformers_version"] == frozen["transformers_version"]
    result = provider.infer(request_input)
    factory.assert_called_once_with(
        spec["family_alias"], revision=spec["revision"], max_new_tokens=1024,
        device="cuda", enable_nvml=True,
    )
    assert result["model"] == result["model_id"] == spec["model_id"]
    assert result["model_revision"] == result["processor_revision"] == spec["revision"]
    config = result["provider_config"]
    assert config["dtype"] == "bfloat16" and config["quantization"] == "none"
    assert config["batch_size"] == 1 and config["do_sample"] is False
    assert config["max_new_tokens"] == 1024 and config["structured_decoding_mode"] == "none"
    assert family.generation_count == family.inference_modes == 1
    assert os.environ["HF_HUB_OFFLINE"] == os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_exact_arbitrary_prompt_and_malformed_output_preserved_without_retry(request_input):
    raw = "  Definitely not JSON!\n```bad\n"
    provider, family, _ = adapter(raw_text=raw)
    result = provider.infer(request_input)
    assert family.prompt == request_input.prompt
    assert result["output_text"] == result["raw_response"]["raw_text"] == raw
    assert result["completed"] and result["error_type"] is None
    assert "schema_valid" not in result and "parsed_response" not in result
    assert result["scientific_attempt"] == result["transport_attempts"] == 1
    assert family.generation_count == 1


def test_exif_applied_in_memory_original_bytes_unchanged(request_input):
    before = request_input.image_path.read_bytes()
    provider, family, _ = adapter()
    result = provider.infer(request_input)
    assert request_input.image_path.read_bytes() == before
    assert family.image.size == (20, 40) and family.image.mode == "RGB"
    # EXIF 6 rotates clockwise: the original red left half becomes the top half.
    assert family.image.getpixel((10, 5))[0] > 240
    assert family.image.getpixel((10, 35))[2] > 240
    metadata = result["raw_response"]["preprocessing"]
    assert metadata["original_exif_orientation"] == 6
    assert metadata["original_size"] == [40, 20] and metadata["oriented_size"] == [20, 40]
    assert metadata["source_image_sha256"] == hashlib.sha256(before).hexdigest()
    assert metadata["source_image_sha256_after"] == metadata["source_image_sha256"]
    assert metadata["source_file_unchanged"] is True
    assert list(request_input.image_path.parent.iterdir()) == [request_input.image_path]


def test_latency_excludes_weight_load_usage_and_peak_retained(request_input):
    provider, family, _ = adapter()
    result = provider.infer(request_input)
    assert result["latency_ms"] == 2250
    assert result["model_load_time_ms"] == 50000
    assert result["raw_response"]["preprocessing"]["latency_ms"] == 250
    assert result["raw_response"]["generation"]["latency_ms"] == 2000
    assert result["peak_vram_mb"] == 5
    assert result["usage"] == {
        "input_tokens": 100, "output_tokens": 12, "total_tokens": 112,
        "reasoning_tokens": None,
    }
    assert result["estimated_cost_usd"] is None and result["http_status"] is None
    assert result["request_id"] is None


def test_minicpm_unknown_visual_usage_not_invented(request_input):
    provider, _, _ = adapter("minicpm", failure="unknown_usage")
    result = provider.infer(request_input)
    assert result["usage"]["input_tokens"] is None
    assert result["usage"]["total_tokens"] is None
    assert result["usage"]["output_tokens"] == 12


@pytest.mark.parametrize("failure,error,generated", [
    ("load", "LOCAL_RUNTIME_ERROR", 0),
    ("oom", "LOCAL_OOM", 1),
    ("post_generation", "LOCAL_RUNTIME_ERROR", 1),
])
def test_runtime_failure_preserved_and_provider_stops(request_input, failure, error, generated):
    provider, family, factory = adapter(failure=failure)
    result = provider.infer(request_input)
    assert not result["completed"] and result["stop_provider"]
    assert result["error_type"] == error
    assert result["transport_attempts"] == result["scientific_attempt"] == 1
    assert family.generation_count == generated and factory.call_count == 1
    if failure == "post_generation":
        assert result["raw_response"]["raw_text"] == family.raw_text
    with pytest.raises(RuntimeError, match="LOCAL_PROVIDER_STOPPED"):
        provider.infer(request_input)
    assert family.generation_count == generated and factory.call_count == 1


@pytest.mark.parametrize("component", ["model_revision", "processor_revision", "tokenizer_revision"])
def test_revision_mismatch_stops_before_generation(request_input, component):
    provider, family, _ = adapter("minicpm")
    setattr(family, component, "wrong-test-revision")
    result = provider.infer(request_input)
    assert result["error_type"] == "MODEL_REVISION_MISMATCH"
    assert result["stop_provider"] and family.generation_count == 0
    assert result[component] == "wrong-test-revision"


def test_context_closes_without_eager_loading_and_reuses_one_loaded_model(request_input):
    provider, family, factory = adapter()
    with provider:
        assert factory.call_count == 0
        provider.infer(request_input)
        provider.infer(request_input)
    assert factory.call_count == family.load_count == 1
    assert family.generation_count == 2 and family.close_count == 1


def test_unsupported_model_rejected_without_fallback():
    with pytest.raises(ValueError, match="gemma, minicpm or qwen"):
        LocalDirectProvider("another-model")


def test_explicit_load_failure_also_latches_stop():
    provider, family, factory = adapter(failure="load")
    with pytest.raises(RuntimeError, match="weights unavailable offline"):
        provider.load()
    with pytest.raises(RuntimeError, match="LOCAL_PROVIDER_STOPPED"):
        provider.load()
    assert factory.call_count == family.load_count == 1


def test_missing_image_preserves_failure_without_generation(tmp_path):
    provider, family, _ = adapter()
    request_input = CloudRequest(tmp_path / "missing.jpg", "Read the scene.", {})
    result = provider.infer(request_input)
    assert result["error_type"] == "LOCAL_RUNTIME_ERROR"
    assert result["raw_response"]["error"]["stage"] == "preprocessing"
    assert result["stop_provider"] and family.generation_count == 0
