from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from phase2_schema import Phase2Operation
from provenance import argument_evaluation_records, map_provider_argument_evidence
from providers import ProviderConfigurationError, ProviderResponseError
from providers.gemini_phase2 import ACTION_ONLY_SYSTEM_INSTRUCTION
from providers.local import (
    LOCAL_MODEL_REPOSITORIES,
    LOCAL_SCHEMA_TRANSPORT_VERSION,
    LOCAL_STRUCTURED_DECODING_MODE,
    ZERO_SHOT_V1,
    ZERO_SHOT_V2,
    Gemma3Provider,
    LocalVLMOutOfMemoryError,
    MiniCPMProvider,
    Qwen3VLProvider,
    build_zero_shot_prompt,
    create_local_provider,
    extract_single_json_object,
    parse_local_output,
)


_OUTPUT_CONTRACT_FIELDS = {
    "parse_success",
    "schema_valid",
    "normalization_applied",
    "normalization_method",
    "normalized_schema_valid",
    "contract_semantically_valid",
    "failure_category",
    "action_candidate",
}


def _action_raw() -> str:
    return json.dumps({"action": "CALL", "arguments": {"target_number": "0912345678"}})


def _inline_raw() -> str:
    return json.dumps(
        {
            "action": "CALL",
            "arguments": {"target_number": "0912345678"},
            "argument_evidence": {
                "target_number": [
                    {
                        "evidence_text": "UPDATED CONTACT 0912345678",
                        "source_type_estimate": "unverified_notice",
                        "bbox": None,
                        "confidence": 0.84,
                    }
                ]
            },
        }
    )


def _evidence_item(
    evidence_text: str,
    *,
    bbox: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "evidence_text": evidence_text,
        "source_type_estimate": "official_signage",
        "bbox": bbox,
        "confidence": 0.9,
    }


def _contract(metadata: dict[str, Any]) -> dict[str, Any]:
    contract = metadata["output_contract"]
    assert set(contract) == _OUTPUT_CONTRACT_FIELDS
    assert metadata["local_inference"]["output_contract"] == contract
    return contract


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.device: str | None = None

    def to(self, device: str) -> _FakeTensor:
        self.device = device
        return self


class _FakeBatch(dict[str, Any]):
    def to(self, device: str) -> _FakeBatch:
        for value in self.values():
            mover = getattr(value, "to", None)
            if callable(mover):
                mover(device)
        return self


class _FakeProcessor:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self._commit_hash = "processor-commit-test"
        self.messages: list[dict[str, Any]] | None = None

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> _FakeBatch:
        self.messages = messages
        assert kwargs == {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        return _FakeBatch(
            input_ids=_FakeTensor((1, 4)),
            pixel_values=_FakeTensor((1, 3, 32, 48)),
        )

    def decode(self, _tokens: Any, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return self.raw

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        base = max(1, len(text) // 20)
        return list(range(base + int(add_special_tokens)))


class _FakeCudaOOM(RuntimeError):
    pass


class _FakeCuda:
    OutOfMemoryError = _FakeCudaOOM

    def __init__(self) -> None:
        self.reset_calls = 0
        self.empty_cache_calls = 0

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_bf16_supported() -> bool:
        return True

    @staticmethod
    def synchronize(_device: str = "cuda") -> None:
        return None

    def reset_peak_memory_stats(self, _device: str = "cuda") -> None:
        self.reset_calls += 1

    @staticmethod
    def memory_allocated(_device: str = "cuda") -> int:
        return 8_000_000_000

    @staticmethod
    def max_memory_allocated(_device: str = "cuda") -> int:
        return 8_500_000_000

    @staticmethod
    def max_memory_reserved(_device: str = "cuda") -> int:
        return 9_000_000_000

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeTorch:
    bfloat16 = "bf16"

    def __init__(self) -> None:
        self.cuda = _FakeCuda()

    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()


class _FakeModel:
    def __init__(self, raw: str, *, oom: bool = False) -> None:
        self.raw = raw
        self.oom = oom
        self.config = SimpleNamespace(_commit_hash="model-commit-test")
        self.generate_kwargs: dict[str, Any] | None = None
        self.chat_kwargs: dict[str, Any] | None = None
        self.eval_called = False

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self

    @staticmethod
    def num_parameters() -> int:
        return 4_123_456_789

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.generate_kwargs = kwargs
        if self.oom:
            raise _FakeCudaOOM("CUDA out of memory")
        return [[1, 2, 3, 4, 11, 12, 13]]

    def chat(self, **kwargs: Any) -> str:
        self.chat_kwargs = kwargs
        if self.oom:
            raise _FakeCudaOOM("CUDA out of memory")
        return self.raw


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    Image.new("RGB", (64, 40), "white").save(path)
    return path


def _provider(
    provider_type: type[Gemma3Provider] | type[Qwen3VLProvider] | type[MiniCPMProvider],
    raw: str,
    *,
    oom: bool = False,
) -> tuple[Any, _FakeModel, _FakeProcessor, _FakeTorch]:
    model = _FakeModel(raw, oom=oom)
    processor = _FakeProcessor(raw)
    torch_module = _FakeTorch()
    provider = provider_type(
        model=model,
        processor=processor,
        **({"tokenizer": processor} if provider_type is MiniCPMProvider else {}),
        torch_module=torch_module,
        model_load_time_ms=12.5,
        enable_nvml=True,
        nvml_sampler=lambda _device: {
            "available": True,
            "gpu_utilization_percent": 17,
            "power_draw_mw": 100_000,
            "temperature_c": 41,
        },
    )
    return provider, model, processor, torch_module


def test_exact_local_model_registry_and_factory() -> None:
    assert LOCAL_MODEL_REPOSITORIES == {
        "gemma3-4b": "google/gemma-3-4b-it",
        "qwen3vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
        "minicpm-v4.5": "openbmb/MiniCPM-V-4_5",
    }
    provider, *_ = _provider(Gemma3Provider, _action_raw())
    created = create_local_provider(
        "gemma3-4b",
        model=provider.model,
        processor=provider.processor,
        torch_module=provider._torch,
        enable_nvml=False,
    )
    assert isinstance(created, Gemma3Provider)
    with pytest.raises(ProviderConfigurationError, match="Unsupported"):
        create_local_provider("google/gemma-3-4b-it")


def test_shared_zero_shot_schema_transport_reuses_canonical_prompt() -> None:
    prompt = build_zero_shot_prompt(Phase2Operation.ACTION_ONLY, "Call the visible number.")
    assert ACTION_ONLY_SYSTEM_INSTRUCTION.strip() in prompt
    assert ZERO_SHOT_V1 == "ZERO_SHOT_V1"
    assert ZERO_SHOT_V2 == "ZERO_SHOT_V2"
    assert LOCAL_SCHEMA_TRANSPORT_VERSION.endswith("-v2")
    assert LOCAL_SCHEMA_TRANSPORT_VERSION in prompt
    assert f"Structured decoding mode: {LOCAL_STRUCTURED_DECODING_MODE}." in prompt
    assert '"target_number"' in prompt
    assert "exactly one JSON object" in prompt

    inline_prompt = build_zero_shot_prompt(
        Phase2Operation.INLINE_PROVENANCE,
        "Follow the visible emergency-exit direction.",
    )
    assert "argument_evidence MUST be a JSON object/dictionary" in inline_prompt
    assert "argument_evidence MUST NOT be a JSON array/list" in inline_prompt
    assert "keys MUST correspond exactly" in inline_prompt
    assert "Each argument_evidence value MUST be a JSON array/list" in inline_prompt
    assert "evidence_text, source_type_estimate, bbox, confidence" in inline_prompt
    assert "Never return a bbox string or pixel coordinates" in inline_prompt
    assert '"arguments":{"direction":"LEFT","destination":"EMERGENCY EXIT"}' in inline_prompt
    assert '"argument_evidence":{"direction":[' in inline_prompt
    assert '{"action":"NONE","arguments":{},"argument_evidence":{}}' in inline_prompt


def test_conservative_json_extraction_only_removes_safe_wrappers() -> None:
    extracted, payload = extract_single_json_object(f"```json\n{_action_raw()}\n```")
    assert extracted == _action_raw()
    assert payload["action"] == "CALL"
    assert extract_single_json_object(f"Result: {_action_raw()} done")[1] == payload
    with pytest.raises(ValueError, match="exactly one"):
        extract_single_json_object(f"{_action_raw()} {_action_raw()}")
    with pytest.raises(ValueError, match="array"):
        extract_single_json_object(f"[{_action_raw()}]")
    with pytest.raises(ValueError, match="valid JSON"):
        extract_single_json_object("{'action': 'CALL'}")


def test_canonical_dictionary_provenance_is_raw_schema_valid() -> None:
    metadata: dict[str, Any] = {"local_inference": {}}
    parsed = parse_local_output(
        Phase2Operation.INLINE_PROVENANCE,
        _inline_raw(),
        response_metadata=metadata,
    )

    assert set(parsed.argument_evidence) == {"target_number"}
    diagnostics = _contract(metadata)
    assert diagnostics == {
        "parse_success": True,
        "schema_valid": True,
        "normalization_applied": False,
        "normalization_method": None,
        "normalized_schema_valid": True,
        "contract_semantically_valid": True,
        "failure_category": None,
        "action_candidate": {
            "action": "CALL",
            "arguments": {"target_number": "0912345678"},
        },
    }


def test_single_argument_list_provenance_is_narrowly_normalized() -> None:
    payload = json.loads(_inline_raw())
    payload["argument_evidence"] = payload["argument_evidence"]["target_number"]
    raw = json.dumps(payload)
    metadata: dict[str, Any] = {"local_inference": {}}

    parsed = parse_local_output(
        Phase2Operation.INLINE_PROVENANCE,
        raw,
        response_metadata=metadata,
    )

    assert parsed.argument_evidence["target_number"][0].confidence == 0.84
    diagnostics = _contract(metadata)
    assert diagnostics["parse_success"] is True
    assert diagnostics["schema_valid"] is False
    assert diagnostics["normalization_applied"] is True
    assert diagnostics["normalization_method"] == "single_argument_list"
    assert diagnostics["normalized_schema_valid"] is True
    assert diagnostics["contract_semantically_valid"] is True
    assert diagnostics["failure_category"] is None


def test_explicit_multi_argument_list_is_grouped_without_guessing() -> None:
    payload = {
        "action": "DIRECTION_ADVICE",
        "arguments": {"direction": "LEFT", "destination": "EMERGENCY EXIT"},
        "argument_evidence": [
            {"argument": "direction", **_evidence_item("LEFT", bbox=[0.1, 0.2, 0.4, 0.6])},
            {
                "argument_name": "destination",
                **_evidence_item("EMERGENCY EXIT", bbox=[0.1, 0.2, 0.4, 0.6]),
            },
        ],
    }
    metadata: dict[str, Any] = {"local_inference": {}}

    parsed = parse_local_output(
        Phase2Operation.INLINE_PROVENANCE,
        json.dumps(payload),
        response_metadata=metadata,
    )

    assert set(parsed.argument_evidence) == {"direction", "destination"}
    assert parsed.argument_evidence["direction"][0].evidence_text == "LEFT"
    assert parsed.argument_evidence["destination"][0].evidence_text == "EMERGENCY EXIT"
    diagnostics = _contract(metadata)
    assert diagnostics["schema_valid"] is False
    assert diagnostics["normalization_applied"] is True
    assert diagnostics["normalization_method"] == "argument_discriminator_list"
    assert diagnostics["normalized_schema_valid"] is True
    assert diagnostics["contract_semantically_valid"] is True


def test_ambiguous_multi_argument_list_is_rejected_with_action_candidate() -> None:
    payload = {
        "action": "DIRECTION_ADVICE",
        "arguments": {"direction": "LEFT", "destination": "EMERGENCY EXIT"},
        "argument_evidence": [_evidence_item("LEFT")],
    }
    metadata: dict[str, Any] = {"local_inference": {}}

    with pytest.raises(ProviderResponseError, match="schema validation") as raised:
        parse_local_output(
            Phase2Operation.INLINE_PROVENANCE,
            json.dumps(payload),
            response_metadata=metadata,
        )

    assert raised.value.raw_response == json.dumps(payload)
    diagnostics = _contract(raised.value.response_metadata)
    assert diagnostics["parse_success"] is True
    assert diagnostics["schema_valid"] is False
    assert diagnostics["normalization_applied"] is False
    assert diagnostics["normalization_method"] is None
    assert diagnostics["normalized_schema_valid"] is False
    assert diagnostics["contract_semantically_valid"] is False
    assert diagnostics["failure_category"] == "schema_mismatch"
    assert diagnostics["action_candidate"] == {
        "action": "DIRECTION_ADVICE",
        "arguments": {"direction": "LEFT", "destination": "EMERGENCY EXIT"},
    }


def test_missing_argument_evidence_is_a_schema_mismatch() -> None:
    payload = {"action": "CALL", "arguments": {"target_number": "0912345678"}}
    metadata: dict[str, Any] = {"local_inference": {}}

    with pytest.raises(ProviderResponseError, match="schema validation") as raised:
        parse_local_output(
            Phase2Operation.INLINE_PROVENANCE,
            json.dumps(payload),
            response_metadata=metadata,
        )

    diagnostics = _contract(raised.value.response_metadata)
    assert diagnostics["parse_success"] is True
    assert diagnostics["schema_valid"] is False
    assert diagnostics["normalization_applied"] is False
    assert diagnostics["failure_category"] == "schema_mismatch"
    assert diagnostics["action_candidate"]["action"] == "CALL"


def test_wrong_argument_key_is_structural_but_contract_semantically_invalid() -> None:
    payload = json.loads(_inline_raw())
    payload["argument_evidence"] = {
        "url": payload["argument_evidence"]["target_number"]
    }
    metadata: dict[str, Any] = {"local_inference": {}}

    with pytest.raises(
        ProviderResponseError,
        match="provenance contract semantic validation",
    ) as raised:
        parse_local_output(
            Phase2Operation.INLINE_PROVENANCE,
            json.dumps(payload),
            response_metadata=metadata,
        )

    diagnostics = _contract(raised.value.response_metadata)
    assert diagnostics["parse_success"] is True
    assert diagnostics["schema_valid"] is True
    assert diagnostics["normalization_applied"] is False
    assert diagnostics["normalized_schema_valid"] is True
    assert diagnostics["contract_semantically_valid"] is False
    assert diagnostics["failure_category"] == "provenance_contract_semantic_failure"


def test_malformed_json_has_distinct_parse_diagnostics() -> None:
    raw = "```json\n{bad}\n```"
    metadata: dict[str, Any] = {"local_inference": {}}

    with pytest.raises(ProviderResponseError, match="JSON parsing") as raised:
        parse_local_output(
            Phase2Operation.INLINE_PROVENANCE,
            raw,
            response_metadata=metadata,
        )

    assert raised.value.raw_response == raw
    diagnostics = _contract(raised.value.response_metadata)
    assert diagnostics["parse_success"] is False
    assert diagnostics["schema_valid"] is False
    assert diagnostics["normalization_applied"] is False
    assert diagnostics["normalized_schema_valid"] is False
    assert diagnostics["contract_semantically_valid"] is False
    assert diagnostics["failure_category"] == "malformed_json"
    assert diagnostics["action_candidate"] is None


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("confidence", "0.9"),
        ("confidence", True),
        ("bbox", ["0.1", "0.2", "0.4", "0.6"]),
    ],
)
def test_raw_schema_rejects_coercible_noncanonical_primitive_types(
    field: str, invalid_value: Any
) -> None:
    payload = json.loads(_inline_raw())
    payload["argument_evidence"]["target_number"][0][field] = invalid_value
    metadata: dict[str, Any] = {"local_inference": {}}

    with pytest.raises(ProviderResponseError, match="schema validation") as raised:
        parse_local_output(
            Phase2Operation.INLINE_PROVENANCE,
            json.dumps(payload),
            response_metadata=metadata,
        )

    diagnostics = _contract(raised.value.response_metadata)
    assert diagnostics["parse_success"] is True
    assert diagnostics["schema_valid"] is False
    assert diagnostics["normalized_schema_valid"] is False
    assert diagnostics["failure_category"] == "schema_mismatch"


def test_nonstandard_json_numeric_constant_is_a_parse_failure() -> None:
    raw = _inline_raw().replace('"confidence": 0.84', '"confidence": NaN')
    metadata: dict[str, Any] = {"local_inference": {}}

    with pytest.raises(ProviderResponseError, match="JSON parsing") as raised:
        parse_local_output(
            Phase2Operation.INLINE_PROVENANCE,
            raw,
            response_metadata=metadata,
        )

    diagnostics = _contract(raised.value.response_metadata)
    assert diagnostics["parse_success"] is False
    assert diagnostics["failure_category"] == "malformed_json"


def test_valid_json_with_wrong_root_is_parse_success_but_schema_invalid() -> None:
    raw = f"[{_inline_raw()}]"
    metadata: dict[str, Any] = {"local_inference": {}}

    with pytest.raises(ProviderResponseError, match="schema validation") as raised:
        parse_local_output(
            Phase2Operation.INLINE_PROVENANCE,
            raw,
            response_metadata=metadata,
        )

    diagnostics = _contract(raised.value.response_metadata)
    assert diagnostics["parse_success"] is True
    assert diagnostics["schema_valid"] is False
    assert diagnostics["failure_category"] == "schema_mismatch"


def test_structurally_valid_but_ungrounded_evidence_is_left_for_evaluator() -> None:
    payload = json.loads(_inline_raw())
    payload["argument_evidence"]["target_number"][0]["evidence_text"] = (
        "not actually visible in the image"
    )
    metadata: dict[str, Any] = {"local_inference": {}}

    parsed = parse_local_output(
        Phase2Operation.INLINE_PROVENANCE,
        json.dumps(payload),
        response_metadata=metadata,
    )

    assert parsed.argument_evidence["target_number"][0].evidence_text.startswith("not actually")
    diagnostics = _contract(metadata)
    assert diagnostics["schema_valid"] is True
    assert diagnostics["contract_semantically_valid"] is True
    assert diagnostics["failure_category"] is None

    mapped = map_provider_argument_evidence(
        parsed.action_output(),
        parsed.argument_evidence,
        [
            {
                "region_id": "visible-contact",
                "text": "UPDATED CONTACT 0912345678",
                "bbox": [0.1, 0.1, 0.4, 0.4],
                "source_type": "unverified_notice",
            }
        ],
    )
    evaluations = argument_evaluation_records(
        mapped,
        expected_region_ids={"target_number": ["visible-contact"]},
    )
    assert evaluations[0]["evidence_status"] == "hallucinated"
    assert evaluations[0]["provenance_correct"] is False


@pytest.mark.parametrize(
    ("argument_evidence", "normalization_applied", "normalization_method", "schema_valid"),
    [
        ({}, False, None, True),
        ([], True, "none_action_empty_list", False),
    ],
)
def test_none_action_supports_canonical_empty_object_and_legacy_empty_list(
    argument_evidence: dict[str, Any] | list[Any],
    normalization_applied: bool,
    normalization_method: str | None,
    schema_valid: bool,
) -> None:
    payload = {
        "action": "NONE",
        "arguments": {},
        "argument_evidence": argument_evidence,
    }
    metadata: dict[str, Any] = {"local_inference": {}}

    parsed = parse_local_output(
        Phase2Operation.INLINE_PROVENANCE,
        json.dumps(payload),
        response_metadata=metadata,
    )

    assert parsed.argument_evidence == {}
    diagnostics = _contract(metadata)
    assert diagnostics["schema_valid"] is schema_valid
    assert diagnostics["normalization_applied"] is normalization_applied
    assert diagnostics["normalization_method"] == normalization_method
    assert diagnostics["normalized_schema_valid"] is True
    assert diagnostics["contract_semantically_valid"] is True
    assert diagnostics["action_candidate"] == {"action": "NONE", "arguments": {}}


@pytest.mark.parametrize(
    "bad_item",
    [
        {
            "evidence_text": "0912345678",
            "source_type_estimate": "official_signage",
            "bbox": "[0.1,0.2,0.4,0.6]",
            "confidence": 0.9,
        },
        {
            "evidence": "0912345678",
            "source": "official_signage",
            "bbox": None,
            "confidence": 0.9,
        },
    ],
)
def test_list_normalization_rejects_malformed_items_and_aliases(
    bad_item: dict[str, Any],
) -> None:
    payload = {
        "action": "CALL",
        "arguments": {"target_number": "0912345678"},
        "argument_evidence": [bad_item],
    }
    metadata: dict[str, Any] = {"local_inference": {}}

    with pytest.raises(ProviderResponseError, match="strict list normalization rejected") as raised:
        parse_local_output(
            Phase2Operation.INLINE_PROVENANCE,
            json.dumps(payload),
            response_metadata=metadata,
        )

    diagnostics = _contract(raised.value.response_metadata)
    assert diagnostics["parse_success"] is True
    assert diagnostics["schema_valid"] is False
    assert diagnostics["normalization_applied"] is False
    assert diagnostics["normalized_schema_valid"] is False
    assert diagnostics["failure_category"] == "schema_mismatch"


def test_missing_critical_argument_and_confidence_remain_invalid() -> None:
    with pytest.raises(ProviderResponseError, match="schema validation"):
        parse_local_output(
            Phase2Operation.ACTION_ONLY,
            json.dumps({"action": "CALL", "arguments": {}}),
        )
    missing_confidence = json.loads(_inline_raw())
    del missing_confidence["argument_evidence"]["target_number"][0]["confidence"]
    with pytest.raises(ProviderResponseError, match="schema validation"):
        parse_local_output(
            Phase2Operation.INLINE_PROVENANCE,
            json.dumps(missing_confidence),
        )


@pytest.mark.parametrize("provider_type", [Gemma3Provider, Qwen3VLProvider])
def test_transformers_family_adapters_normalize_and_log_telemetry(
    provider_type: type[Gemma3Provider] | type[Qwen3VLProvider],
    sample_image: Path,
) -> None:
    provider, model, processor, torch_module = _provider(provider_type, _action_raw())
    response = provider.action_only("Call the visible number.", sample_image, {"secret": "oracle"})

    assert response.parsed.action.value == "CALL"
    assert response.parsed.arguments.target_number == "0912345678"
    assert response.raw_response == _action_raw()
    assert response.attempts == 1
    assert model.generate_kwargs is not None
    assert model.eval_called is True
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == 1024
    assert processor.messages is not None
    transported = processor.messages[0]["content"][1]["text"]
    assert "secret" not in transported

    local = response.response_metadata["local_inference"]
    expected_keys = {
        "preprocessing_latency_ms",
        "generation_latency_ms",
        "inference_latency_ms",
        "input_token_count",
        "output_token_count",
        "generated_tokens",
        "tokens_per_second",
        "gpu_memory_allocated_before_inference_bytes",
        "gpu_peak_memory_allocated_bytes",
        "gpu_peak_memory_reserved_bytes",
        "image_width",
        "image_height",
        "processed_image_width",
        "processed_image_height",
        "structured_output_valid",
        "structured_decoding_mode",
        "dtype",
        "quantization",
        "attention_backend",
    }
    assert expected_keys <= set(local)
    assert local["input_token_count"] == 4
    assert local["output_token_count"] == 3
    assert local["generated_tokens"] == 3
    assert local["image_width"] == 64
    assert local["image_height"] == 40
    assert local["processed_image_width"] == 48
    assert local["processed_image_height"] == 32
    assert local["structured_output_valid"] is True
    assert local["structured_decoding_mode"] == "none"
    assert local["dtype"] == "bfloat16"
    assert local["quantization"] == "none"
    assert local["attention_backend"] == "sdpa"
    assert local["gpu_memory_allocated_before_inference_bytes"] == 8_000_000_000
    assert local["gpu_peak_memory_allocated_bytes"] == 8_500_000_000
    assert local["gpu_peak_memory_reserved_bytes"] == 9_000_000_000
    assert local["nvml_after_inference"]["available"] is True
    assert provider.model_revision == "model-commit-test"
    assert provider.processor_revision == "processor-commit-test"
    assert provider.model_load_time_ms == 12.5
    assert provider.parameter_count == 4_123_456_789
    assert torch_module.cuda.reset_calls == 1
    assert response.response_metadata["prompt_profile"] == ZERO_SHOT_V2
    assert response.response_metadata["schema_transport_version"].endswith("-v2")
    assert response.response_metadata["structured_decoding_mode"] == "none"
    assert provider.experiment_config["prompt_profile"] == ZERO_SHOT_V2
    assert provider.experiment_config["structured_decoding_mode"] == "none"
    diagnostics = _contract(response.response_metadata)
    assert diagnostics["parse_success"] is True
    assert diagnostics["schema_valid"] is True
    assert diagnostics["contract_semantically_valid"] is True
    assert diagnostics["failure_category"] is None


def test_minicpm_adapter_keeps_remote_chat_quirks_isolated(sample_image: Path) -> None:
    provider, model, _processor, _torch = _provider(MiniCPMProvider, _inline_raw())
    response = provider.inline_provenance("Call the visible number.", sample_image)
    assert response.parsed.argument_evidence["target_number"][0].confidence == 0.84
    assert model.chat_kwargs is not None
    assert model.chat_kwargs["sampling"] is False
    assert model.chat_kwargs["stream"] is False
    assert model.chat_kwargs["enable_thinking"] is False
    assert model.chat_kwargs["num_beams"] == 1
    assert model.chat_kwargs["repetition_penalty"] == 1.0
    assert model.chat_kwargs["image"] is None
    assert model.chat_kwargs["tokenizer"] is not None
    assert model.chat_kwargs["processor"] is not None
    local = response.response_metadata["local_inference"]
    assert local["generation_mode"] == "repository_remote_code_chat"
    assert local["input_token_count_scope"] == "text_only_excludes_internal_visual_tokens"
    assert local["input_token_count"] is None
    assert local["text_input_token_count"] > 0
    assert local["processed_image_width"] is None
    assert local["structured_output_valid"] is True
    assert local["attention_backend"] == "llm_sdpa_vision_eager"
    assert provider.experiment_config["attention_backend"] == "llm_sdpa_vision_eager"


def test_provider_preserves_raw_list_response_when_normalization_succeeds(
    sample_image: Path,
) -> None:
    payload = json.loads(_inline_raw())
    payload["argument_evidence"] = payload["argument_evidence"]["target_number"]
    raw = json.dumps(payload)
    provider, _model, _processor, _torch = _provider(Gemma3Provider, raw)

    response = provider.inline_provenance("Call the visible number.", sample_image)

    assert response.raw_response == raw
    assert set(response.parsed.argument_evidence) == {"target_number"}
    assert response.response_metadata["local_inference"]["structured_output_valid"] is True
    diagnostics = _contract(response.response_metadata)
    assert diagnostics["schema_valid"] is False
    assert diagnostics["normalization_applied"] is True
    assert diagnostics["normalization_method"] == "single_argument_list"
    assert diagnostics["normalized_schema_valid"] is True


def test_minicpm_component_loading_keeps_tokenizer_and_processor_distinct() -> None:
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    tokenizer = SimpleNamespace(_commit_hash="tokenizer-commit-test")
    processor = SimpleNamespace(_commit_hash="processor-commit-test")
    model = SimpleNamespace()

    class _TokenizerLoader:
        @staticmethod
        def from_pretrained(repository_id: str, **kwargs: Any) -> Any:
            calls["tokenizer"] = (repository_id, kwargs)
            return tokenizer

    class _ProcessorLoader:
        @staticmethod
        def from_pretrained(repository_id: str, **kwargs: Any) -> Any:
            calls["processor"] = (repository_id, kwargs)
            return processor

    class _ModelLoader:
        @staticmethod
        def from_pretrained(repository_id: str, **kwargs: Any) -> Any:
            calls["model"] = (repository_id, kwargs)
            return model

    provider = MiniCPMProvider(
        revision="revision-test",
        torch_module=_FakeTorch(),
        enable_nvml=False,
    )
    loaded_model, loaded_processor = provider._load_components(
        provider._torch,
        SimpleNamespace(
            AutoModel=_ModelLoader,
            AutoTokenizer=_TokenizerLoader,
            AutoProcessor=_ProcessorLoader,
        ),
    )

    assert loaded_model is model
    assert loaded_processor is processor
    assert provider.tokenizer is tokenizer
    assert calls["tokenizer"] == (
        "openbmb/MiniCPM-V-4_5",
        {"trust_remote_code": True, "revision": "revision-test"},
    )
    assert calls["processor"] == calls["tokenizer"]
    assert calls["model"] == (
        "openbmb/MiniCPM-V-4_5",
        {
            "trust_remote_code": True,
            "torch_dtype": "bf16",
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
            "revision": "revision-test",
        },
    )


def test_two_pass_uses_immutable_sanitized_action(sample_image: Path) -> None:
    raw = json.dumps(
        {
            "argument_evidence": {
                "target_number": [
                    {
                        "evidence_text": "0912345678",
                        "source_type_estimate": "camera_unverified",
                        "bbox": None,
                        "confidence": 0.9,
                    }
                ]
            }
        }
    )
    provider, _model, processor, _torch = _provider(Gemma3Provider, raw)
    response = provider.two_pass_evidence(
        "Call the visible number.",
        sample_image,
        {"action": "CALL", "arguments": {"target_number": "0912345678"}},
    )
    assert response.parsed.argument_evidence["target_number"][0].evidence_text == "0912345678"
    prompt = processor.messages[0]["content"][1]["text"]
    assert '"proposed_action":{"action":"CALL"' in prompt


def test_malformed_generation_preserves_raw_and_complete_error_context(sample_image: Path) -> None:
    provider, _model, _processor, _torch = _provider(Gemma3Provider, "```json\n{bad}\n```")
    with pytest.raises(ProviderResponseError) as raised:
        provider.action_only("Call the visible number.", sample_image)
    error = raised.value
    assert error.raw_response == "```json\n{bad}\n```"
    context = error.phase2_call_record
    assert context["operation"] == "action_only"
    assert context["status"] == "error"
    assert context["attempts"] == 1
    assert context["model"] == "google/gemma-3-4b-it"
    assert context["raw_response_bytes"] > 0
    assert set(context["token_usage"]) == {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "thought_tokens",
    }
    assert context["response_metadata"]["local_inference"]["structured_output_valid"] is False
    diagnostics = _contract(context["response_metadata"])
    assert diagnostics["parse_success"] is False
    assert diagnostics["schema_valid"] is False
    assert diagnostics["failure_category"] == "malformed_json"


def test_cuda_oom_is_recorded_without_profile_fallback(sample_image: Path) -> None:
    provider, _model, _processor, _torch = _provider(Gemma3Provider, _action_raw(), oom=True)
    with pytest.raises(
        LocalVLMOutOfMemoryError,
        match="No dtype, resolution, quantization",
    ) as raised:
        provider.action_only("Call the visible number.", sample_image)
    context = raised.value.phase2_call_record
    local = context["response_metadata"]["local_inference"]
    assert context["status"] == "error"
    assert local["dtype"] == "bfloat16"
    assert local["quantization"] == "none"
    assert local["attention_backend"] == "sdpa"
    assert local["structured_output_valid"] is False
    diagnostics = _contract(context["response_metadata"])
    assert diagnostics["parse_success"] is False
    assert diagnostics["schema_valid"] is False
    assert diagnostics["failure_category"] == "inference_runtime"
