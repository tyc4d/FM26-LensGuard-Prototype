"""One-attempt physical DIRECT transport over the frozen local VLM mechanics.

No prompt wrapper, evidence registry, output parser, evaluator or gate is used.
Original files remain untouched; EXIF orientation is applied only in memory.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import io
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from providers.base_cloud_vlm import CloudResponse, jsonable, redact_secrets


_COMMON_CONFIG = {
    "dtype": "bfloat16", "device": "cuda", "batch_size": 1,
    "quantization": "none", "do_sample": False, "max_new_tokens": 1024,
    "structured_decoding_mode": "none", "offline_cache_only": True,
    "semantic_attempts": 1, "runtime_retries": 0,
    "image_orientation": "Pillow.ImageOps.exif_transpose_in_memory_then_RGB",
    "image_resize": "frozen_family_processor_native_only",
    "prompt_transport": "exact_prompt_no_added_wrapper",
    "latency_scope": "image_preprocessing_generation_decode_excludes_model_load",
}

LOCAL_MODELS = {
    "gemma": {
        "model_id": "google/gemma-3-4b-it", "family_alias": "gemma3-4b",
        "revision": "093f9f388b31de276ce2de164bdc2081324b9767",
        "runtime_python": str(Path(os.environ.get("LENSGUARD_GEMMA_PYTHON",
            str(Path.home() / "venvs/lensguard-vlm/bin/python"))).expanduser()),
        "transformers_version": "5.16.1",
        "default_config": {**_COMMON_CONFIG, "attention_backend": "sdpa"},
    },
    "minicpm": {
        "model_id": "openbmb/MiniCPM-V-4_5", "family_alias": "minicpm-v4.5",
        "revision": "daef484c35ec93210ec93c5e901f8f3e9b78ee34",
        "runtime_python": str(Path(os.environ.get("LENSGUARD_MINICPM_PYTHON",
            str(Path.home() / "venvs/lensguard-minicpm/bin/python"))).expanduser()),
        "transformers_version": "4.51.0",
        "default_config": {
            **_COMMON_CONFIG, "attention_backend": "llm_sdpa_vision_eager",
            "trust_remote_code": True, "enable_thinking": False,
            "sampling": False, "stream": False, "num_beams": 1,
            "repetition_penalty": 1.0,
        },
    },
    "qwen": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct", "family_alias": "qwen3vl-8b",
        "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "runtime_python": str(Path(os.environ.get("LENSGUARD_QWEN_PYTHON",
            str(Path.home() / "venvs/lensguard-qwen/bin/python"))).expanduser()),
        "transformers_version": "5.16.1",
        "default_config": {**_COMMON_CONFIG, "attention_backend": "sdpa"},
    },
}


def _force_offline() -> None:
    """Also handle libraries imported before this adapter set the environment."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        constants.HF_HUB_OFFLINE = True
    transformers_hub = sys.modules.get("transformers.utils.hub")
    if transformers_hub is not None and hasattr(transformers_hub, "_is_offline_mode"):
        transformers_hub._is_offline_mode = True


class LocalDirectProvider:
    """Lazy local adapter; injected factories permit tests without model loading."""

    api_interface = "local_transformers_frozen_family_direct_v1"

    def __init__(
        self, alias: str, *, provider_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        utcnow: Callable[[], datetime] | None = None,
    ) -> None:
        if alias not in LOCAL_MODELS:
            raise ValueError("Local DIRECT provider must be gemma, minicpm or qwen")
        self.spec = copy.deepcopy(LOCAL_MODELS[alias])
        self.provider = alias
        self.model = self.model_id = self.spec["model_id"]
        self._factory = provider_factory
        self._clock = clock
        self._utcnow = utcnow or (lambda: datetime.now(timezone.utc))
        self._local: Any = None
        self._loaded = False
        self._stopped = False
        self._runtime: dict[str, str] = {}
        _force_offline()

    @property
    def provider_config(self) -> dict[str, Any]:
        return {
            **copy.deepcopy(self.spec["default_config"]),
            "model_id": self.model_id, "requested_revision": self.spec["revision"],
            "model_revision": getattr(self._local, "model_revision", None),
            "processor_revision": getattr(self._local, "processor_revision", None),
            "runtime_python": self.spec["runtime_python"],
            "required_torch_version": "2.10.0+cu128",
            "required_transformers_version": self.spec["transformers_version"],
            "observed_runtime": dict(self._runtime),
        }

    def load(self) -> LocalDirectProvider:
        if self._stopped:
            raise RuntimeError("LOCAL_PROVIDER_STOPPED: create no replacement semantic attempt")
        if self._loaded:
            return self
        try:
            return self._load_once()
        except Exception:
            self._stopped = True
            raise

    def _load_once(self) -> LocalDirectProvider:
        _force_offline()
        factory = self._factory
        if factory is None:
            self._runtime = {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "torchvision", "pillow")
            }
            self._runtime["python_executable"] = sys.executable
            if (self._runtime["torch"] != "2.10.0+cu128"
                    or self._runtime["transformers"] != self.spec["transformers_version"]):
                raise RuntimeError("FROZEN_RUNTIME_MISMATCH: no runtime replacement permitted")
            from providers.local import create_local_provider

            factory = create_local_provider
        self._local = factory(
            self.spec["family_alias"], revision=self.spec["revision"],
            max_new_tokens=1024, device="cuda", enable_nvml=True,
        )
        self._local.load()
        if (self._local.model_revision != self.spec["revision"]
                or self._local.processor_revision != self.spec["revision"]):
            raise RuntimeError("FROZEN_REVISION_MISMATCH: no model replacement permitted")
        if self.provider == "minicpm" and self._local.tokenizer_revision != self.spec["revision"]:
            raise RuntimeError("FROZEN_REVISION_MISMATCH: tokenizer differs from pinned revision")
        self._loaded = True
        return self

    def infer(self, request: Any) -> dict[str, Any]:
        """Return raw text even when it is malformed; never invoke a parser."""
        if self._stopped:
            raise RuntimeError("LOCAL_PROVIDER_STOPPED: no further inference allowed")
        timestamp = self._utcnow().astimezone(timezone.utc).isoformat()
        started = self._clock()
        raw: dict[str, Any] = {"raw_text": "", "preprocessing": {}, "generation": {}}
        usage: dict[str, Any] = {
            "input_tokens": None, "output_tokens": None, "total_tokens": None,
            "reasoning_tokens": None,
        }
        peak: float | None = None
        error_type = error_detail = None
        stage = "load"
        try:
            self.load()
            started = self._clock()
            stage = "preprocessing"
            path = Path(request.image_path)
            if not isinstance(request.prompt, str) or not request.prompt.strip():
                raise ValueError("Direct prompt must be a nonempty string")
            source_bytes = path.read_bytes()
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            with Image.open(io.BytesIO(source_bytes)) as original:
                original.load()
                raw["preprocessing"].update({
                    "source_image_sha256": source_hash,
                    "original_exif_orientation": original.getexif().get(274),
                    "original_size": list(original.size),
                })
                oriented = ImageOps.exif_transpose(original).convert("RGB")
            raw["preprocessing"]["oriented_size"] = list(oriented.size)
            prepared = self._local._prepare_input(request.prompt, oriented)
            raw["preprocessing"].update({
                "latency_ms": (self._clock() - started) * 1000,
                "processed_image_width": prepared.processed_image_width,
                "processed_image_height": prepared.processed_image_height,
                **jsonable(prepared.metadata),
            })
            usage["input_tokens"] = prepared.input_token_count
            self._local._synchronize()
            self._local._reset_peak_memory()
            generation_started = self._clock()
            stage = "generation"
            with self._local._torch_module().inference_mode():
                generated = self._local._generate(prepared)
                # Capture first, before synchronization or downstream metadata work.
                raw["raw_text"] = generated.raw_text
                raw["generation"] = jsonable(generated.metadata)
                usage["output_tokens"] = generated.output_token_count
            self._local._synchronize()
            raw["generation"]["latency_ms"] = (self._clock() - generation_started) * 1000
            memory = self._local._cuda_memory("max_memory_allocated")
            peak = memory / 1024**2 if memory is not None else None
            stage = "source_integrity"
            after_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            raw["preprocessing"].update({
                "source_image_sha256_after": after_hash,
                "source_file_unchanged": after_hash == source_hash,
            })
            if after_hash != source_hash:
                raise RuntimeError("SOURCE_IMAGE_CHANGED: input changed during inference")
        except Exception as error:
            detail = str(error)
            if "out of memory" in detail.casefold() or "outofmemory" in type(error).__name__.casefold():
                error_type = "LOCAL_OOM"
            elif "FROZEN_REVISION_MISMATCH" in detail:
                error_type = "MODEL_REVISION_MISMATCH"
            elif "FROZEN_RUNTIME_MISMATCH" in detail:
                error_type = "RUNTIME_CONFIG_MISMATCH"
            elif stage == "source_integrity":
                error_type = "SOURCE_IMAGE_CHANGED"
            else:
                error_type = "LOCAL_RUNTIME_ERROR"
            self._stopped = True
            # Sanitize the complete response once below, retaining an accurate
            # raw_response_redacted flag when an exception contained a secret.
            error_detail = detail
            raw["error"] = {"type": error_type, "stage": stage,
                            "exception_type": type(error).__name__, "detail": error_detail}
        if usage["input_tokens"] is not None and usage["output_tokens"] is not None:
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        response = CloudResponse(
            provider=self.provider, model=self.model, model_id=self.model_id,
            api_interface=self.api_interface, timestamp_utc=timestamp, raw_response=raw,
            output_text=raw["raw_text"], completed=error_type is None,
            error_type=error_type, error_detail=error_detail, http_status=None,
            api_status="COMPLETED" if error_type is None else "LOCAL_ERROR", request_id=None,
            latency_ms=(self._clock() - started) * 1000, usage=usage,
            provider_config=self.provider_config, transport_attempts=1,
            stop_provider=self._stopped, returned_model=self.model_id if self._loaded else None,
            cost_basis="Not applicable: local GPU inference; electricity cost not measured",
        ).to_dict()
        response.update({
            "peak_vram_mb": peak,
            "model_revision": getattr(self._local, "model_revision", None),
            "processor_revision": getattr(self._local, "processor_revision", None),
            "tokenizer_revision": getattr(self._local, "tokenizer_revision", None),
            "model_load_time_ms": getattr(self._local, "model_load_time_ms", None),
        })
        sanitized = redact_secrets(response)
        sanitized["raw_response_redacted"] = sanitized != response
        return sanitized

    def close(self) -> None:
        if self._local is not None:
            self._local.close()
        self._loaded = False

    def __enter__(self) -> LocalDirectProvider:
        # Loading remains inside infer so initialization failures can be preserved.
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = ["LOCAL_MODELS", "LocalDirectProvider"]
