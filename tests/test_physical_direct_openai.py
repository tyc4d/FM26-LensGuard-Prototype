"""Original-byte OpenAI image-detail compatibility; all calls use an injected fake."""

from __future__ import annotations

import base64
import copy
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from cloud_baseline_store import digest
from physical_direct_contracts import DIRECT_SCHEMA, TASK_PROMPTS, build_prompt
from physical_direct_openai import PhysicalDirectOpenAIProvider
from providers.base_cloud_vlm import CloudRequest
from providers.openai_vlm import OpenAIProvider


class FakeResponse:
    output_text = "Malformed scientific output stays malformed."
    status = "completed"
    model = "gpt-5.6-sol"
    id = "mock-request"
    usage = {"input_tokens": 100, "output_tokens": 12, "total_tokens": 112,
             "input_tokens_details": {"cached_tokens": 0},
             "output_tokens_details": {"reasoning_tokens": 0}}

    def model_dump(self, **kwargs):
        return {name: getattr(self, name) for name in (
            "output_text", "status", "model", "id", "usage",
        )}


@pytest.fixture
def original_image(tmp_path):
    path = tmp_path / "original-software-fixture.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (128, 64), "red").save(path, exif=exif)
    return path


def make_providers(*responses):
    create = Mock(side_effect=responses or [FakeResponse()])
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    sleeps = []
    physical = PhysicalDirectOpenAIProvider(
        model="gpt-5.6-sol", client=client, sleep=sleeps.append,
    )
    frozen = OpenAIProvider(model="gpt-5.6-sol", client=client)
    return physical, frozen, create, sleeps


@pytest.mark.parametrize("scenario", tuple(TASK_PROMPTS))
def test_only_high_detail_added_all_tasks_keep_original_bytes_and_contract(original_image, scenario):
    before = original_image.read_bytes()
    request = CloudRequest(original_image, build_prompt(scenario), DIRECT_SCHEMA, "FIXTURE", "DIRECT")
    schema_before = copy.deepcopy(request.response_schema)
    physical, frozen, create, _ = make_providers()
    baseline = frozen.build_payload(request)
    payload = physical.build_payload(request)
    comparable = copy.deepcopy(payload)
    images = [item for item in comparable["input"][0]["content"] if item["type"] == "input_image"]
    assert len(images) == 1
    for item in images:
        assert item.pop("detail") == "high"
        data_url = item["image_url"]
        assert data_url.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(data_url.split(",", 1)[1]) == before
    assert comparable == baseline
    assert payload["model"] == "gpt-5.6-sol"
    assert payload["tools"] == [] and payload["store"] is False
    assert payload["text"]["format"]["schema"] == DIRECT_SCHEMA == schema_before
    assert request.response_schema == schema_before
    assert payload["input"][0]["content"][1]["text"] == request.prompt
    assert original_image.read_bytes() == before
    assert hashlib.sha256(original_image.read_bytes()).digest() == hashlib.sha256(before).digest()
    assert list(original_image.parent.iterdir()) == [original_image]
    create.assert_not_called()


def test_native_detail_is_bound_in_config_and_frozen_adapter_is_unchanged(original_image):
    physical, frozen, _, _ = make_providers()
    config = physical.provider_config
    assert config["image_detail"] == "high"
    assert config["image_bytes"] == "unchanged_original"
    assert config["generation_config"] == frozen.provider_config["generation_config"]
    assert config["sdk_internal_retries"] == 0
    assert "image_detail" not in frozen.provider_config
    changed = {**config, "image_detail": "original"}
    assert digest(config) != digest(changed)
    request = CloudRequest(original_image, build_prompt("CALL"), DIRECT_SCHEMA)
    assert "detail" not in frozen.build_payload(request)["input"][0]["content"][0]


def test_mock_inference_preserves_malformed_output_and_records_actual_detail(original_image):
    physical, _, create, _ = make_providers()
    request = CloudRequest(original_image, build_prompt("SAFETY"), DIRECT_SCHEMA)
    response = physical.infer(request)
    assert create.call_count == 1
    sent = create.call_args.kwargs
    assert sent["input"][0]["content"][0]["detail"] == "high"
    assert response.provider_config["image_detail"] == "high"
    assert response.output_text == response.raw_response["output_text"] == FakeResponse.output_text
    assert response.completed and response.scientific_attempt == response.transport_attempts == 1
    assert response.model == response.model_id == "gpt-5.6-sol"


def test_transport_retry_reuses_exact_high_detail_request_and_original_bytes(original_image):
    error = RuntimeError("Temporary transport condition")
    error.status_code = 429
    error.body = {"error": {"code": "rate_limit_exceeded"}}
    physical, _, create, sleeps = make_providers(error, FakeResponse())
    request = CloudRequest(original_image, build_prompt("NAVIGATION"), DIRECT_SCHEMA)
    before = original_image.read_bytes()
    response = physical.infer(request)
    assert response.completed and response.scientific_attempt == 1
    assert response.transport_attempts == 2 and sleeps == [1.0]
    assert create.call_args_list[0].kwargs == create.call_args_list[1].kwargs
    assert create.call_args.kwargs["input"][0]["content"][0]["detail"] == "high"
    assert original_image.read_bytes() == before
