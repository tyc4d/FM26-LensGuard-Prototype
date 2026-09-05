"""Cloud transport tests use synthetic credentials and make no network calls."""

from __future__ import annotations

import copy
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from providers.base import ProviderConfigurationError
from providers.base_cloud_vlm import CloudRequest, redact_secrets, require_secret_safety
from providers.gemini_vlm import GeminiProvider
from providers.openai_vlm import OpenAIProvider


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeResponse:
    def __init__(self, text='{"action":"NONE","arguments":{}}', **kwargs):
        self.output_text = text
        self.status = "completed"
        self.id = "response-test-id"
        self.model = "returned-model-snapshot"
        self.usage = {}
        self.__dict__.update(kwargs)

    def model_dump(self, **_):
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}


class ApiError(Exception):
    def __init__(self, status, body):
        super().__init__(json.dumps(body))
        self.status_code = status
        self.body = body
        self.request_id = "failed-request-test-id"


@pytest.fixture
def request_input(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color="white").save(path)
    return CloudRequest(
        image_path=path, prompt="Inspect the image and propose the requested action.",
        response_schema={
            "type": "object", "properties": {"action": {"type": "string"}},
            "required": ["action"], "additionalProperties": False,
        }, case_id="TEST-001", arm="ACTION_ONLY",
    )


def make_provider(kind, responses, clock=None, **kwargs):
    clock = clock or FakeClock()
    resource = SimpleNamespace(create=Mock(side_effect=responses))
    client = SimpleNamespace(**{"responses" if kind == "openai" else "interactions": resource})
    provider_class = OpenAIProvider if kind == "openai" else GeminiProvider
    if kind == "gemini":
        kwargs.setdefault("jitter", lambda: 0.0)
    provider = provider_class(
        client=client, sleep=clock.sleep, clock=clock,
        utcnow=lambda: datetime(2026, 9, 5, tzinfo=timezone.utc), **kwargs,
    )
    return provider, resource, clock


@pytest.mark.parametrize("kind,expected", [
    ("openai", "gpt-5.6-sol"), ("gemini", "gemini-3.1-flash-lite"),
])
def test_defaults_exact_model_native_payload_and_no_tools(kind, expected, request_input, monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    provider, resource, clock = make_provider(kind, [FakeResponse()])
    result = provider.infer(request_input)
    sent = resource.create.call_args.kwargs
    assert result.model == result.model_id == sent["model"] == expected
    assert result.returned_model == "returned-model-snapshot"
    assert result.completed and result.scientific_attempt == 1
    assert result.transport_attempts == 1 and result.http_status == 200
    assert result.estimated_cost_usd is None
    assert sent["tools"] == [] and sent["store"] is False
    assert "system_instruction" not in sent and "instructions" not in sent
    if kind == "openai":
        content = sent["input"][0]["content"]
        assert content[0]["image_url"].startswith("data:image/png;base64,")
        assert content[1]["text"] == request_input.prompt
        assert sent["text"]["format"]["schema"] == request_input.response_schema
        assert sent["temperature"] == 0 and sent["reasoning"] == {"effort": "none"}
    else:
        assert sent["input"][0]["mime_type"] == "image/png"
        assert sent["input"][1]["text"] == request_input.prompt
        assert sent["response_format"]["schema"] == request_input.response_schema
        assert sent["generation_config"]["seed"] == 0
        assert "temperature" not in sent["generation_config"]
    assert not clock.sleeps
    assert result.to_dict()["raw_response"]["output_text"] == result.output_text


@pytest.mark.parametrize("kind", ["openai", "gemini"])
def test_environment_model_override_is_preserved(kind, request_input, monkeypatch):
    monkeypatch.setenv(f"{kind.upper()}_MODEL", "explicit-requested-model")
    provider, resource, _ = make_provider(kind, [FakeResponse()])
    assert provider.infer(request_input).model == "explicit-requested-model"
    assert resource.create.call_args.kwargs["model"] == "explicit-requested-model"


@pytest.mark.parametrize("kind", ["openai", "gemini"])
@pytest.mark.parametrize("text", ["", "not json", '{"invented_evidence":"EV-NOT-REAL"}'])
def test_malformed_model_output_is_preserved_without_semantic_retry(kind, text, request_input):
    provider, resource, _ = make_provider(kind, [FakeResponse(text)])
    result = provider.infer(request_input)
    assert result.completed and result.output_text == text
    assert result.raw_response["output_text"] == text
    assert resource.create.call_count == result.transport_attempts == 1


@pytest.mark.parametrize("kind", ["openai", "gemini"])
def test_exact_request_transport_retry_and_native_usage(kind, request_input):
    usage = {
        "input_tokens": 100, "output_tokens": 15, "total_tokens": 118,
        "output_tokens_details": {"reasoning_tokens": 3},
    } if kind == "openai" else {
        "total_input_tokens": 100, "total_output_tokens": 15, "total_thought_tokens": 3,
        "total_tokens": 118,
    }
    provider, resource, clock = make_provider(kind, [
        ApiError(429, {"status": "RESOURCE_EXHAUSTED", "message": "per minute quota"}),
        FakeResponse(usage=usage),
    ])
    result = provider.infer(request_input)
    assert resource.create.call_args_list[0] == resource.create.call_args_list[1]
    assert result.scientific_attempt == 1 and result.transport_attempts == 2
    assert result.rate_limit_events == 1
    assert result.total_backoff_seconds == (1.0 if kind == "openai" else 30.0)
    assert result.usage["input_tokens"] == 100
    assert result.usage["output_tokens"] == 15
    assert result.usage["reasoning_tokens"] == 3
    assert result.usage["native"] == usage
    assert result.latency_ms == pytest.approx(sum(clock.sleeps) * 1000)
    assert result.transport_events[0]["raw_error_response"]["status"] == "RESOURCE_EXHAUSTED"


def test_gemini_exhausts_exactly_four_retries_with_full_audit(request_input):
    provider, resource, clock = make_provider("gemini", [
        ApiError(429, {"status": "RESOURCE_EXHAUSTED", "message": "temporary RPM quota"})
        for _ in range(5)
    ])
    result = provider.infer(request_input)
    assert not result.completed and result.error_type == "RATE_LIMIT_EXHAUSTED"
    assert result.transport_attempts == resource.create.call_count == 5
    assert result.rate_limit_events == 5 and result.total_backoff_seconds == 450
    assert not result.hard_quota and not result.stop_provider
    assert sum(clock.sleeps) == 450 and max(clock.sleeps) <= 60
    assert [event.get("backoff_seconds") for event in result.transport_events] == [30, 60, 120, 240, None]
    assert all(call == resource.create.call_args_list[0] for call in resource.create.call_args_list)


@pytest.mark.parametrize("message", [
    "daily request quota exhausted", "requests per day exceeded", "requestsPerDay quota",
    "insufficient_quota", "quota limit: 0",
])
def test_gemini_hard_quota_stops_without_hammering(message, request_input):
    provider, resource, clock = make_provider("gemini", [ApiError(429, {"message": message})])
    result = provider.infer(request_input)
    assert result.error_type == "RATE_LIMIT_EXHAUSTED"
    assert result.hard_quota and result.stop_provider
    assert result.transport_attempts == result.rate_limit_events == 1
    assert result.total_backoff_seconds == 0 and clock.sleeps == []
    with pytest.raises(ProviderConfigurationError, match="stopped"):
        provider.infer(request_input)
    assert resource.create.call_count == 1


@pytest.mark.parametrize("kind", ["openai", "gemini"])
@pytest.mark.parametrize("status,body,category", [
    (404, {"message": "model unavailable"}, "MODEL_UNAVAILABLE"),
    (400, {"code": "model_not_found"}, "MODEL_UNAVAILABLE"),
    (401, {"message": "invalid key"}, "AUTHENTICATION_FAILED"),
    (403, {"message": "billing restriction"}, "ACCESS_OR_BILLING_RESTRICTION"),
    (400, {"message": "schema not supported"}, "API_COMPATIBILITY_ERROR"),
])
def test_unavailable_auth_and_compatibility_stop_without_fallback(kind, status, body, category, request_input):
    provider, resource, clock = make_provider(kind, [ApiError(status, body)])
    result = provider.infer(request_input)
    assert result.error_type == category and result.stop_provider
    assert result.request_id == "failed-request-test-id"
    assert result.transport_attempts == resource.create.call_count == 1
    assert not clock.sleeps


@pytest.mark.parametrize("kind", ["openai", "gemini"])
def test_incomplete_returned_response_is_never_retried(kind, request_input):
    response = FakeResponse("partial", status="failed", errors=[{"code": "generation_failed"}])
    provider, resource, _ = make_provider(kind, [response])
    result = provider.infer(request_input)
    assert not result.completed and result.error_type == "MODEL_RESPONSE_INCOMPLETE"
    assert result.raw_response["output_text"] == "partial"
    assert resource.create.call_count == 1


def test_gemini_spaces_successes_excludes_pacing_from_latency(request_input, monkeypatch):
    monkeypatch.setenv("GEMINI_REQUEST_DELAY_SECONDS", "8")
    provider, resource, clock = make_provider("gemini", [FakeResponse(), FakeResponse()], jitter=lambda: 1.5)
    first = provider.infer(request_input)
    second = provider.infer(request_input)
    assert clock.sleeps == [9.5] and resource.create.call_count == 2
    assert first.latency_ms == second.latency_ms == 0.0
    assert provider.next_request_delay_seconds == 9.5


def test_openai_has_no_artificial_spacing(request_input):
    provider, _, clock = make_provider("openai", [FakeResponse(), FakeResponse()])
    provider.infer(request_input)
    provider.infer(request_input)
    assert clock.sleeps == []


@pytest.mark.parametrize("invalid", ["-1", "nan", "inf", "not-a-number"])
def test_invalid_pacing_configuration_is_rejected(invalid, monkeypatch):
    monkeypatch.setenv("GEMINI_REQUEST_DELAY_SECONDS", invalid)
    with pytest.raises(ProviderConfigurationError, match="finite nonnegative"):
        make_provider("gemini", [])


def test_gemini_sdk_retry_configuration_is_disabled():
    config = SimpleNamespace(retry_config=object())
    client = SimpleNamespace(interactions=SimpleNamespace(sdk_configuration=config))
    GeminiProvider(client=client)
    assert config.retry_config is None


def test_shared_schema_and_request_image_are_not_mutated(request_input):
    original = copy.deepcopy(request_input.response_schema)
    original_bytes = request_input.image_path.read_bytes()
    for kind in ("openai", "gemini"):
        provider, _, _ = make_provider(kind, [FakeResponse()])
        provider.infer(request_input)
    assert request_input.response_schema == original
    assert request_input.image_path.read_bytes() == original_bytes


def test_full_secret_value_is_redacted_from_success_error_and_metadata(request_input, monkeypatch):
    # Deliberately synthetic; never copy any real environment credential into tests.
    sentinel = "synthetic-sensitive-test-value"
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)
    provider, _, _ = make_provider("openai", [FakeResponse(sentinel, usage={"diagnostic": sentinel})])
    result = provider.infer(request_input)
    assert sentinel not in json.dumps(result.to_dict())
    assert result.raw_response_redacted and result.output_text == "[REDACTED]"
    provider, _, _ = make_provider("openai", [ApiError(400, {"message": sentinel})])
    result = provider.infer(request_input)
    assert sentinel not in json.dumps(result.to_dict()) and result.raw_response_redacted
    assert redact_secrets({sentinel: [sentinel]}) == {"[REDACTED]": ["[REDACTED]"]}


def test_masked_credential_echo_is_redacted_without_preserving_prefix(monkeypatch):
    sentinel = "synthetic-sensitive-test-value"
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)
    prefix = sentinel[:8]
    sanitized = redact_secrets({"message": f"Incorrect API key provided: {prefix}*****alue"})
    assert prefix not in json.dumps(sanitized)
    assert sanitized["message"] == "Incorrect API key provided: [REDACTED]"


@pytest.mark.parametrize("value", [0, "0"])
def test_json_encoded_zero_hard_quota_stops_immediately(value, request_input):
    provider, resource, clock = make_provider("gemini", [ApiError(429, json.dumps({
        "error": {"details": [{"quotaValue": value}]},
    }))])
    result = provider.infer(request_input)
    assert result.hard_quota and result.stop_provider
    assert resource.create.call_count == 1 and not clock.sleeps


def test_secret_safety_requires_ignored_untracked_dotenv(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / ".env").write_text("SYNTHETIC_SETTING=local\n")
    with pytest.raises(ProviderConfigurationError, match="SECRET_SAFETY_FAILED"):
        require_secret_safety(tmp_path)
    (tmp_path / ".gitignore").write_text(".env\n")
    require_secret_safety(tmp_path)
    subprocess.run(["git", "add", "-f", ".env"], cwd=tmp_path, check=True, capture_output=True)
    with pytest.raises(ProviderConfigurationError, match="SECRET_SAFETY_FAILED"):
        require_secret_safety(tmp_path)


@pytest.mark.parametrize("kind", ["openai", "gemini"])
def test_missing_credentials_is_normalized_before_any_network(kind, monkeypatch, request_input):
    monkeypatch.delenv(f"{kind.upper()}_API_KEY", raising=False)
    provider_class = OpenAIProvider if kind == "openai" else GeminiProvider
    provider = provider_class()
    result = provider.infer(request_input)
    assert result.error_type == "MISSING_CREDENTIALS" and result.stop_provider
    assert result.transport_attempts == 0


def test_installed_gemini_schema_serializes_native_configuration_without_hidden_fields(request_input):
    from google.genai._gaos.types.interactions.createmodelinteraction import CreateModelInteraction

    provider, _, _ = make_provider("gemini", [])
    payload = provider.build_payload(request_input)
    payload.pop("api_version")
    serialized = CreateModelInteraction.model_validate(payload).model_dump(mode="json", exclude_none=True)
    assert serialized["generation_config"] == payload["generation_config"]
    assert serialized["response_format"] == payload["response_format"]
    assert serialized["tools"] == []


@pytest.mark.parametrize("kind", ["openai", "gemini"])
def test_actual_sdk_http_transport_has_no_hidden_retries(kind, request_input, monkeypatch):
    import httpx

    requests = []

    def handler(request):
        requests.append(request.content)
        return httpx.Response(429, json={"error": {
            "code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Temporary RPM quota",
        }})

    clock = FakeClock()
    if kind == "openai":
        import openai

        constructor = openai.OpenAI
        monkeypatch.setenv("OPENAI_API_KEY", "synthetic-transport-test-value")
        monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: constructor(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ))
        provider = OpenAIProvider(sleep=clock.sleep, clock=clock)
        assert provider.client.max_retries == 0
    else:
        from google import genai

        constructor = genai.Client
        monkeypatch.setenv("GEMINI_API_KEY", "synthetic-transport-test-value")

        def client_factory(**kwargs):
            kwargs["http_options"].client_args = {"transport": httpx.MockTransport(handler)}
            return constructor(**kwargs)

        monkeypatch.setattr(genai, "Client", client_factory)
        provider = GeminiProvider(sleep=clock.sleep, clock=clock, jitter=lambda: 0)
        assert provider._interactions.sdk_configuration.retry_config is None
    try:
        result = provider.infer(request_input)
    finally:
        provider.close()
    assert len(requests) == result.transport_attempts == 5
    assert result.rate_limit_events == 5
    assert all(request == requests[0] for request in requests)
