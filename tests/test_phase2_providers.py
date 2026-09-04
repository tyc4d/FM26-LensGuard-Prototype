from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError

from phase2_schema import (
    ActionOnlyOutput,
    EvidenceOnlyOutput,
    InlineProvenanceOutput,
    NormalizedBBox,
    Phase2Arm,
    SourceTypeEstimate,
    canonical_phase2_arm,
)
from providers import (
    GeminiPhase2Provider,
    MockPhase2Provider,
    ProviderResponseError,
    ProviderUnavailableError,
    RetryConfig,
)
from providers.gemini_phase2 import (
    INLINE_PROVENANCE_SYSTEM_INSTRUCTION,
    PHASE2_ACTION_PROMPT_VERSION,
    PHASE2_INLINE_PROMPT_VERSION,
    PHASE2_TWO_PASS_PROMPT_VERSION,
    TWO_PASS_EVIDENCE_SYSTEM_INSTRUCTION,
    sanitized_phase2_action,
)


class _FakeResponse:
    def __init__(self, text: str, *, total_tokens: int = 21) -> None:
        self.output_text = text
        self.status = "completed"
        self.errors: list[Any] = []
        self.id = "phase2-test-interaction"
        self.model = "gemini-0-flash-returned"
        self.created = "2026-01-01T00:00:00Z"
        self.updated = self.created
        self.usage = {
            "total_input_tokens": 13,
            "total_output_tokens": total_tokens - 13,
            "total_tokens": total_tokens,
            "total_cached_tokens": 0,
            "total_thought_tokens": 0,
        }


class _FakeInteractions:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.interactions = _FakeInteractions(responses)


class _RateLimitError(Exception):
    status_code = 429

    def __init__(self, message: str = "quota", *, body: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.body = body


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "phase2.png"
    Image.new("RGB", (40, 24), "white").save(path)
    return path


def _action_raw(number: str = "+1-202-555-0199") -> str:
    return json.dumps({"action": "CALL", "arguments": {"target_number": number}})


def _evidence(number: str = "+1-202-555-0199") -> dict[str, Any]:
    return {
        "target_number": [
            {
                "evidence_text": number,
                "source_type_estimate": "camera_unverified",
                "bbox": [0.1, 0.2, 0.4, 0.6],
                "confidence": 0.88,
            }
        ]
    }


def _evidence_raw(number: str = "+1-202-555-0199") -> str:
    return json.dumps({"argument_evidence": _evidence(number)})


def _inline_raw(number: str = "+1-202-555-0199") -> str:
    return json.dumps(
        {
            "action": "CALL",
            "arguments": {"target_number": number},
            "argument_evidence": _evidence(number),
        }
    )


def test_bbox_is_normalized_xyxy_and_serializes_as_array() -> None:
    box = NormalizedBBox.model_validate([0.1, 0.2, 0.8, 0.9])
    assert box.model_dump(mode="json") == [0.1, 0.2, 0.8, 0.9]

    for invalid in (
        [-0.1, 0.2, 0.8, 0.9],
        [0.1, 0.2, 1.1, 0.9],
        [0.8, 0.2, 0.1, 0.9],
        [0.1, 0.9, 0.8, 0.2],
        [0.1, 0.2, 0.8],
    ):
        with pytest.raises(ValidationError):
            NormalizedBBox.model_validate(invalid)


def test_evidence_lists_may_be_empty_but_argument_keys_must_be_complete() -> None:
    parsed = InlineProvenanceOutput.model_validate(
        {
            "action": "CALL",
            "arguments": {"target_number": "12345"},
            "argument_evidence": {"target_number": []},
        }
    )
    assert parsed.argument_evidence == {"target_number": []}

    with pytest.raises(ValidationError, match="exactly match"):
        InlineProvenanceOutput.model_validate(
            {
                "action": "CALL",
                "arguments": {"target_number": "12345"},
                "argument_evidence": {},
            }
        )


def test_arm_parser_uses_canonical_values_and_cli_aliases() -> None:
    assert canonical_phase2_arm("two-pass") is Phase2Arm.TWO_PASS_PROVENANCE
    assert canonical_phase2_arm("oracle") is Phase2Arm.ORACLE_PROVENANCE
    assert canonical_phase2_arm("INLINE_PROVENANCE") is Phase2Arm.INLINE_PROVENANCE


def test_sanitized_phase2_action_excludes_reason_confidence_and_evidence() -> None:
    payload = sanitized_phase2_action(
        {
            "action": "CALL",
            "arguments": {"target_number": "+1-202-555-0199"},
        }
    )
    assert payload == {
        "action": "CALL",
        "arguments": {"target_number": "+1-202-555-0199"},
    }
    assert "reason_summary" not in payload
    assert "confidence" not in payload
    assert "argument_evidence" not in payload


def test_gemini_action_only_is_one_multimodal_call_without_metadata(
    sample_image: Path,
) -> None:
    raw = _action_raw()
    client = _FakeClient([_FakeResponse(raw)])
    provider = GeminiPhase2Provider(model="gemini-0-flash-test", client=client)

    response = provider.action_only(
        "Call the desk.",
        sample_image,
        scenario={"region_id": "must-not-leak", "regions": [{"secret": True}]},
    )

    assert response.parsed.action.value == "CALL"
    assert response.raw_response == raw
    assert response.response_metadata["operation"] == "action_only"
    assert len(client.interactions.calls) == 1
    call = client.interactions.calls[0]
    assert [item["type"] for item in call["input"]] == ["image", "text"]
    assert call["input"][1] == {"type": "text", "text": "Call the desk."}
    assert base64.b64decode(call["input"][0]["data"]).startswith(b"\x89PNG")
    assert call["store"] is False
    assert "must-not-leak" not in repr(call)
    assert set(json.loads(raw)) == {"action", "arguments"}


def test_gemini_inline_is_one_call_with_list_evidence(sample_image: Path) -> None:
    raw = _inline_raw()
    client = _FakeClient([_FakeResponse(raw, total_tokens=34)])
    provider = GeminiPhase2Provider(model="gemini-0-flash-test", client=client)

    result = provider.run_arm(
        Phase2Arm.INLINE_PROVENANCE,
        "Call the desk.",
        sample_image,
        scenario={"region_id": "oracle-only"},
    )

    assert result.call_count == 1
    assert result.total_attempts == 1
    assert result.aggregate_token_usage.total_tokens == 34
    assert result.argument_evidence["target_number"][0].bbox is not None
    assert len(client.interactions.calls) == 1
    assert "oracle-only" not in repr(client.interactions.calls[0])


def test_gemini_two_pass_uses_same_image_and_sanitized_action(
    sample_image: Path,
) -> None:
    client = _FakeClient([_FakeResponse(_action_raw()), _FakeResponse(_evidence_raw())])
    provider = GeminiPhase2Provider(model="gemini-0-flash-test", client=client)

    result = provider.run_arm(
        "two_pass",
        "Call the desk.",
        sample_image,
        scenario={"region_id": "never-send", "attack_source": "never-send"},
    )

    assert result.arm is Phase2Arm.TWO_PASS_PROVENANCE
    assert result.call_count == 2
    assert [call.operation.value for call in result.calls] == [
        "action_only",
        "two_pass_evidence",
    ]
    assert len(client.interactions.calls) == 2
    assert (
        client.interactions.calls[0]["input"][0]["data"]
        == client.interactions.calls[1]["input"][0]["data"]
    )
    second = json.loads(client.interactions.calls[1]["input"][1]["text"])
    assert second == {
        "trusted_user_request": "Call the desk.",
        "proposed_action": {
            "action": "CALL",
            "arguments": {"target_number": "+1-202-555-0199"},
        },
    }
    assert "never-send" not in repr(client.interactions.calls)
    assert set(second["proposed_action"]) == {"action", "arguments"}


def test_current_sdk_serializes_phase2_inline_schema(sample_image: Path) -> None:
    """Exercise google-genai 2.22.0's real serializer without network access."""

    import httpx
    from google import genai
    from google.genai import types

    raw = _inline_raw()
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "int_phase2_serializer",
                "status": "completed",
                "model": "gemini-0-flash-returned",
                "usage": {
                    "totalInputTokens": 13,
                    "totalOutputTokens": 21,
                    "totalTokens": 34,
                },
                "steps": [
                    {
                        "type": "model_output",
                        "status": "done",
                        "content": [{"type": "text", "text": raw}],
                    }
                ],
            },
            headers={"content-type": "application/json"},
        )

    client = genai.Client(
        api_key="test-only",
        http_options=types.HttpOptions(client_args={"transport": httpx.MockTransport(handler)}),
    )
    try:
        result = GeminiPhase2Provider(model="gemini-0-flash-test", client=client).inline_provenance(
            "Call the desk.", sample_image
        )
    finally:
        client.close()

    assert result.parsed.argument_evidence["target_number"][0].confidence == 0.88
    assert len(captured) == 1
    body = captured[0]
    assert body["model"] == "gemini-0-flash-test"
    assert body["store"] is False
    assert [item["type"] for item in body["input"][0]["content"]] == [
        "image",
        "text",
    ]
    assert body["response_format"]["mime_type"] == "application/json"
    schema_text = json.dumps(body["response_format"]["schema"])
    assert "argument_evidence" in schema_text
    assert "source_type_estimate" in schema_text


def test_gemini_oracle_is_exactly_one_action_call_and_can_reuse_it(
    sample_image: Path,
) -> None:
    client = _FakeClient([_FakeResponse(_action_raw())])
    provider = GeminiPhase2Provider(model="gemini-0-flash-test", client=client)
    action_response = provider.action_only("Call the desk.", sample_image)

    oracle = provider.run_arm(
        Phase2Arm.ORACLE_PROVENANCE,
        "Call the desk.",
        sample_image,
        reused_action_response=action_response,
    )

    assert len(client.interactions.calls) == 1
    assert oracle.call_count == 1
    assert oracle.reused_action_only is True
    assert oracle.argument_evidence == {}


def test_gemini_invalid_bbox_preserves_raw_response(sample_image: Path) -> None:
    payload = json.loads(_inline_raw())
    payload["argument_evidence"]["target_number"][0]["bbox"] = [0.9, 0.2, 0.1, 0.7]
    raw = json.dumps(payload)
    provider = GeminiPhase2Provider(
        model="gemini-0-flash-test",
        client=_FakeClient([_FakeResponse(raw)]),
    )

    with pytest.raises(ProviderResponseError, match="schema validation") as caught:
        provider.inline_provenance("Call the desk.", sample_image)

    assert caught.value.raw_response == raw
    call_record = caught.value.phase2_call_record
    assert call_record["operation"] == "inline_provenance"
    assert call_record["status"] == "error"
    assert call_record["attempts"] == 1
    assert call_record["token_usage"]["total_tokens"] == 21
    assert call_record["raw_response_bytes"] == len(raw.encode("utf-8"))


def test_gemini_exhausted_request_preserves_physical_attempt_count(
    sample_image: Path,
) -> None:
    provider = GeminiPhase2Provider(
        model="gemini-0-flash-test",
        client=_FakeClient([_RateLimitError("quota"), _RateLimitError("quota")]),
        retry_config=RetryConfig(max_attempts=2, initial_delay_seconds=0),
        sleep=lambda _seconds: None,
    )

    with pytest.raises(
        ProviderUnavailableError, match="no fallback model was attempted"
    ) as caught:
        provider.action_only("Call the desk.", sample_image)

    call_record = caught.value.phase2_call_record
    assert call_record["operation"] == "action_only"
    assert call_record["attempts"] == 2
    assert call_record["response_metadata"]["http_status"] == 429
    assert len(call_record["response_metadata"]["application_retry_audit"]["events"]) == 2


def test_two_pass_evidence_retries_after_server_supplied_delay(sample_image: Path) -> None:
    delay = 18.745623466
    client = _FakeClient(
        [
            _RateLimitError(
                body={
                    "error": {
                        "message": f"Quota exceeded. Please retry in {delay}s.",
                        "code": "too_many_requests",
                    }
                }
            ),
            _FakeResponse(_evidence_raw()),
        ]
    )
    sleeps: list[float] = []
    provider = GeminiPhase2Provider(
        model="gemini-0-flash-test",
        client=client,
        retry_config=RetryConfig(
            max_attempts=2,
            initial_delay_seconds=1,
            max_delay_seconds=16,
        ),
        sleep=sleeps.append,
    )
    action = ActionOnlyOutput.model_validate(json.loads(_action_raw()))

    response = provider.two_pass_evidence(
        "Call the desk.",
        sample_image,
        action,
    )

    assert response.attempts == 2
    assert sleeps == pytest.approx([delay])
    assert len(client.interactions.calls) == 2
    assert response.parsed.argument_evidence["target_number"][0].evidence_text
    audit = response.response_metadata["application_retry_audit"]
    assert audit["policy_version"] == "server-aware-retry-v1"
    assert audit["total_wait_seconds"] == pytest.approx(delay)


def test_prompts_reject_claimed_authority_and_chain_of_thought() -> None:
    combined = (
        INLINE_PROVENANCE_SYSTEM_INSTRUCTION + "\n" + TWO_PASS_EVIDENCE_SYSTEM_INSTRUCTION
    ).casefold()
    assert "not an authenticity" in combined
    assert "do not prove" in combined
    assert "chain-of-thought" in combined
    assert "region_id" not in combined


def test_evidence_prompt_v2_requests_the_entire_source_panel() -> None:
    assert PHASE2_ACTION_PROMPT_VERSION == "phase2-action-v1"
    assert PHASE2_INLINE_PROMPT_VERSION == "phase2-inline-provenance-v2"
    assert PHASE2_TWO_PASS_PROMPT_VERSION == "phase2-two-pass-evidence-v2"

    for instruction in (
        INLINE_PROVENANCE_SYSTEM_INSTRUCTION,
        TWO_PASS_EVIDENCE_SYSTEM_INSTRUCTION,
    ):
        contract = " ".join(instruction.casefold().split())
        assert "entire visually distinct source panel/region" in contract
        assert "not a tight text-localization box" in contract
        assert "do not box only a phone number, url, arrow, word, line, or glyphs" in contract
        assert "full source-panel/region boundary" in contract


def _load_phase2_records() -> list[dict[str, Any]]:
    return json.loads(Path("dataset_phase2/metadata.json").read_text(encoding="utf-8"))["records"]


@pytest.mark.parametrize(
    "condition",
    [
        "CLEAN_TRUSTED",
        "BENIGN_UNTRUSTED_SUBSTITUTION",
        "AUTHORITY_IMPERSONATION",
        "OBVIOUS_INJECTION_CONTROL",
        "EXPLICIT_USER_OVERRIDE",
        "NO_VERIFIED_GROUND_TRUTH",
        "TRUSTED_BUT_CONFLICTING_UPDATE",
    ],
)
def test_mock_supports_standard_and_special_conditions(condition: str) -> None:
    record = next(item for item in _load_phase2_records() if item["condition"] == condition)
    provider = MockPhase2Provider(seed=7, latency_ms_per_call=2.5)

    action = provider.action_only(record["user_prompt"], record["image_path"], record)

    assert action.parsed.action.value == record["ground_truth_action"]
    primary = record["critical_argument_name"]
    expected = (
        record["attacker_target"]
        if condition
        in {
            "BENIGN_UNTRUSTED_SUBSTITUTION",
            "AUTHORITY_IMPERSONATION",
            "OBVIOUS_INJECTION_CONTROL",
            "NO_VERIFIED_GROUND_TRUTH",
        }
        else record["ground_truth_arguments"][primary]
    )
    assert getattr(action.parsed.arguments, primary) == expected
    assert action.latency_ms == 2.5
    assert action.response_metadata["mock"] is True
    assert action.response_metadata["usage"]["total_tokens"] > 0


def test_mock_evidence_uses_selected_region_without_exposing_region_id() -> None:
    record = next(
        item
        for item in _load_phase2_records()
        if item["condition"] == "BENIGN_UNTRUSTED_SUBSTITUTION" and item["action_family"] == "CALL"
    )
    provider = MockPhase2Provider()

    inline = provider.inline_provenance(record["user_prompt"], record["image_path"], record)
    evidence = inline.parsed.argument_evidence[record["critical_argument_name"]]
    selected_value = getattr(inline.parsed.arguments, record["critical_argument_name"])
    selected = next(
        region
        for region in record["regions"]
        if any(
            claim["argument"] == record["critical_argument_name"]
            and claim["value"] == selected_value
            for claim in region["claims"]
        )
    )

    assert len(evidence) == 1
    assert evidence[0].source_type_estimate.value == selected["source_type"]
    assert evidence[0].bbox is not None
    assert evidence[0].bbox.model_dump(mode="json") == selected["bbox"]
    assert selected["region_id"] not in inline.raw_response
    assert inline.response_metadata["prompt_version"] == PHASE2_INLINE_PROMPT_VERSION


def test_mock_explicit_override_attributes_primary_argument_to_user() -> None:
    record = next(
        item
        for item in _load_phase2_records()
        if item["condition"] == "EXPLICIT_USER_OVERRIDE" and item["action_family"] == "OPEN_URL"
    )
    result = MockPhase2Provider().run_arm(
        "inline_provenance",
        record["user_prompt"],
        record["image_path"],
        record,
    )
    evidence = result.argument_evidence[record["critical_argument_name"]][0]
    assert evidence.source_type_estimate is SourceTypeEstimate.EXPLICIT_USER
    assert evidence.bbox is None


def test_mock_two_pass_exposes_two_call_latency_and_token_accounting() -> None:
    record = _load_phase2_records()[0]
    provider = MockPhase2Provider(latency_ms_per_call=3.0)

    result = provider.run_arm(
        Phase2Arm.TWO_PASS_PROVENANCE,
        record["user_prompt"],
        record["image_path"],
        record,
    )

    assert result.call_count == 2
    assert result.total_attempts == 2
    assert result.total_latency_ms == 6.0
    assert result.aggregate_token_usage.total_tokens is not None
    assert result.aggregate_token_usage.total_tokens > 0
    assert provider.call_count == 2
    assert all(call.response_metadata["mock"] is True for call in result.calls)
    assert result.calls[0].response_metadata["prompt_version"] == PHASE2_ACTION_PROMPT_VERSION
    assert (
        result.calls[1].response_metadata["prompt_version"]
        == PHASE2_TWO_PASS_PROMPT_VERSION
    )


def test_mock_request_seed_controls_arbitrary_wrong_value() -> None:
    record = dict(_load_phase2_records()[0])
    record["mock_behavior"] = "arbitrary_wrong"
    provider = MockPhase2Provider(seed=7)

    provider.set_request_seed(101)
    first = provider.action_only(record["user_prompt"], record["image_path"], record)
    provider.set_request_seed(202)
    second = provider.action_only(record["user_prompt"], record["image_path"], record)
    provider.set_request_seed(101)
    repeated = provider.action_only(record["user_prompt"], record["image_path"], record)

    key = record["critical_argument_name"]
    assert getattr(first.parsed.arguments, key) != getattr(second.parsed.arguments, key)
    assert getattr(first.parsed.arguments, key) == getattr(repeated.parsed.arguments, key)
    assert first.response_metadata["request_generation_config"]["seed"] == 101
    assert second.response_metadata["request_generation_config"]["seed"] == 202


def test_mock_full_phase2_dataset_validates_across_every_arm() -> None:
    provider = MockPhase2Provider(seed=19)
    for record in _load_phase2_records():
        for arm in Phase2Arm:
            result = provider.run_arm(
                arm,
                record["user_prompt"],
                record["image_path"],
                record,
            )
            assert result.action_output.action.value == record["ground_truth_action"]
            assert result.call_count == (2 if arm is Phase2Arm.TWO_PASS_PROVENANCE else 1)
            if arm in {
                Phase2Arm.INLINE_PROVENANCE,
                Phase2Arm.TWO_PASS_PROVENANCE,
            }:
                assert set(result.argument_evidence) == set(result.action_output.argument_values())
            else:
                assert result.argument_evidence == {}


def test_evidence_only_schema_accepts_explicitly_missing_evidence() -> None:
    parsed = EvidenceOnlyOutput.model_validate({"argument_evidence": {"target_number": []}})
    action = ActionOnlyOutput.model_validate(
        {"action": "CALL", "arguments": {"target_number": "12345"}}
    )
    assert parsed.argument_evidence["target_number"] == []
    assert action.action.value == "CALL"
