"""Best-effort, JSON-safe runtime metadata for Phase 2.5 local VLM runs.

This module intentionally imports no ML or NVIDIA package at import time.  A
benchmark can therefore use the cache/disk helpers (and the unit tests can
exercise every branch) without requiring PyTorch, Transformers, or NVML.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import platform
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


SYSTEM_INFO_SCHEMA_VERSION = "phase2.5-system-info-v1"

ModuleLoader = Callable[[str], Any]
CommandRunner = Callable[[Sequence[str]], Any]


def json_safe(value: Any, *, _seen: set[int] | None = None) -> Any:
    """Return a value accepted by strict ``json.dumps(..., allow_nan=False)``.

    Tensor/NumPy scalar-like objects are reduced through ``item()``.  Unknown
    metadata objects are represented by their string form; scientific result
    validation remains the responsibility of the result-store module.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (Path, os.PathLike)):
        return os.fspath(value)
    if isinstance(value, Enum):
        return json_safe(value.value, _seen=_seen)
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value), _seen=_seen)

    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        return "<recursive>"

    if isinstance(value, Mapping):
        seen.add(identity)
        try:
            return {
                str(key): json_safe(item, _seen=seen)
                for key, item in value.items()
            }
        finally:
            seen.remove(identity)
    if isinstance(value, (list, tuple, set, frozenset)):
        seen.add(identity)
        try:
            return [json_safe(item, _seen=seen) for item in value]
        finally:
            seen.remove(identity)

    item = getattr(value, "item", None)
    if callable(item):
        try:
            reduced = item()
        except Exception:  # pragma: no cover - defensive third-party boundary
            pass
        else:
            if reduced is not value:
                return json_safe(reduced, _seen=seen)
    return str(value)


def resolve_component_revision(component: Any, fallback: str | None = None) -> str | None:
    """Extract a Hugging Face commit hash from a loaded model/processor.

    Transformers stores the resolved commit in slightly different locations
    across model, tokenizer, and processor classes.  Callers should still pass
    the provider's resolved revision as ``fallback`` when it is available.
    """

    candidates: list[Any] = []
    if component is not None:
        candidates.extend(
            (
                getattr(component, "_commit_hash", None),
                getattr(getattr(component, "config", None), "_commit_hash", None),
                getattr(getattr(component, "config", None), "commit_hash", None),
            )
        )
        init_kwargs = getattr(component, "init_kwargs", None)
        if isinstance(init_kwargs, Mapping):
            candidates.extend(
                (init_kwargs.get("_commit_hash"), init_kwargs.get("commit_hash"))
            )
        for child_name in ("tokenizer", "image_processor"):
            child = getattr(component, child_name, None)
            if child is not None:
                candidates.append(getattr(child, "_commit_hash", None))
                child_kwargs = getattr(child, "init_kwargs", None)
                if isinstance(child_kwargs, Mapping):
                    candidates.append(child_kwargs.get("_commit_hash"))
    candidates.append(fallback)
    for candidate in candidates:
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return None


def huggingface_cache_root(
    *,
    environ: Mapping[str, str] | None = None,
    home: str | Path | None = None,
) -> Path:
    """Resolve the Hugging Face Hub cache without importing huggingface_hub."""

    env = os.environ if environ is None else environ
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"]).expanduser()
    if env.get("HUGGINGFACE_HUB_CACHE"):
        return Path(env["HUGGINGFACE_HUB_CACHE"]).expanduser()
    if env.get("HF_HOME"):
        hf_home = Path(env["HF_HOME"]).expanduser()
    else:
        base = (
            Path(env["XDG_CACHE_HOME"]).expanduser()
            if env.get("XDG_CACHE_HOME")
            else Path(home).expanduser() / ".cache"
            if home is not None
            else Path.home() / ".cache"
        )
        hf_home = base / "huggingface"
    return hf_home / "hub"


def huggingface_model_cache_path(
    repository_id: str,
    *,
    cache_root: str | Path | None = None,
) -> Path:
    """Return the normal Hub cache directory for ``org/model`` without writes."""

    pieces = repository_id.split("/")
    if len(pieces) < 2 or any(piece in {"", ".", ".."} for piece in pieces):
        raise ValueError(f"Invalid Hugging Face repository ID: {repository_id!r}")
    root = Path(cache_root) if cache_root is not None else huggingface_cache_root()
    return root / ("models--" + "--".join(pieces))


def resolve_cached_revision(
    repository_id: str,
    revision: str = "main",
    *,
    cache_root: str | Path | None = None,
) -> str | None:
    """Resolve a cached branch/tag to its snapshot commit, never contacting HF."""

    model_path = huggingface_model_cache_path(repository_id, cache_root=cache_root)
    direct_snapshot = model_path / "snapshots" / revision
    if direct_snapshot.is_dir():
        return revision
    reference = model_path / "refs" / revision
    try:
        resolved = reference.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return None
    if not resolved or not (model_path / "snapshots" / resolved).is_dir():
        return None
    return resolved


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _cache_size_bytes(model_path: Path) -> int:
    blobs = model_path / "blobs"
    if not blobs.is_dir():
        return 0
    total = 0
    try:
        for entry in blobs.iterdir():
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
    except OSError:
        return 0
    return total


def huggingface_cache_preflight(
    repository_id: str,
    *,
    estimated_download_bytes: int | None = None,
    reserve_bytes: int = 0,
    revision: str = "main",
    cache_root: str | Path | None = None,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    """Describe cache location and disk headroom without creating/downloading files."""

    if estimated_download_bytes is not None and (
        not isinstance(estimated_download_bytes, int)
        or isinstance(estimated_download_bytes, bool)
        or estimated_download_bytes < 0
    ):
        raise ValueError("estimated_download_bytes must be a nonnegative integer or None")
    if not isinstance(reserve_bytes, int) or isinstance(reserve_bytes, bool) or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a nonnegative integer")

    root = Path(cache_root) if cache_root is not None else huggingface_cache_root()
    model_path = huggingface_model_cache_path(repository_id, cache_root=root)
    usage = disk_usage(_nearest_existing_path(root))
    free = int(usage.free)
    required = (
        estimated_download_bytes + reserve_bytes
        if estimated_download_bytes is not None
        else None
    )
    return {
        "model_repository_id": repository_id,
        "requested_revision": revision,
        "cached_revision": resolve_cached_revision(
            repository_id, revision, cache_root=root
        ),
        "huggingface_cache_root": str(root.expanduser().absolute()),
        "model_cache_path": str(model_path.expanduser().absolute()),
        "model_cache_exists": model_path.is_dir(),
        "model_cache_bytes": _cache_size_bytes(model_path),
        "disk_total_bytes": int(usage.total),
        "disk_used_bytes": int(usage.used),
        "disk_free_bytes": free,
        "estimated_download_bytes": estimated_download_bytes,
        "reserve_bytes": reserve_bytes,
        "required_free_bytes": required,
        "sufficient_free_space": free >= required if required is not None else None,
        "post_download_free_bytes": free - required if required is not None else None,
    }


def _load_optional(module_loader: ModuleLoader, name: str) -> tuple[Any | None, str | None]:
    try:
        return module_loader(name), None
    except Exception as error:  # optional runtime dependency boundary
        return None, f"{type(error).__name__}: {error}"


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    rendered = str(value).strip()
    return rendered or None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _device_index(device: str | int) -> int:
    if isinstance(device, int):
        return max(0, device)
    rendered = str(device)
    if ":" in rendered:
        try:
            return max(0, int(rendered.rsplit(":", 1)[1]))
        except ValueError:
            pass
    return 0


def _collect_nvml(nvml: Any, device_index: int) -> dict[str, Any]:
    values: dict[str, Any] = {
        "nvml_available": False,
        "nvml_error": None,
        "gpu_model": None,
        "vram_total_bytes": None,
        "nvidia_driver_version": None,
        "gpu_utilization_percent": None,
        "gpu_power_draw_watts": None,
        "gpu_temperature_celsius": None,
    }
    initialized = False
    try:
        nvml.nvmlInit()
        initialized = True
        handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
        values["gpu_model"] = _text(nvml.nvmlDeviceGetName(handle))
        memory = nvml.nvmlDeviceGetMemoryInfo(handle)
        values["vram_total_bytes"] = int(memory.total)
        values["nvidia_driver_version"] = _text(nvml.nvmlSystemGetDriverVersion())
        utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
        values["gpu_utilization_percent"] = _number(utilization.gpu)
        values["gpu_power_draw_watts"] = _number(nvml.nvmlDeviceGetPowerUsage(handle))
        if values["gpu_power_draw_watts"] is not None:
            values["gpu_power_draw_watts"] /= 1000.0
        temperature_kind = getattr(nvml, "NVML_TEMPERATURE_GPU", 0)
        values["gpu_temperature_celsius"] = _number(
            nvml.nvmlDeviceGetTemperature(handle, temperature_kind)
        )
        values["nvml_available"] = True
    except Exception as error:  # optional telemetry must not abort a run
        values["nvml_error"] = f"{type(error).__name__}: {error}"
    finally:
        if initialized:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass
    return values


def _default_command_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )


def _collect_nvidia_smi(
    command_runner: CommandRunner,
    device_index: int,
) -> tuple[dict[str, Any], str | None]:
    command = (
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = command_runner(command)
    except Exception as error:
        return {}, f"{type(error).__name__}: {error}"
    returncode = getattr(completed, "returncode", 1)
    if returncode != 0:
        error_text = _text(getattr(completed, "stderr", None))
        return {}, error_text or f"nvidia-smi exited with status {returncode}"
    first_line = _text(getattr(completed, "stdout", None))
    if not first_line:
        return {}, "nvidia-smi returned no GPU metadata"
    columns = [item.strip() for item in first_line.splitlines()[0].split(",")]
    if len(columns) != 3:
        return {}, f"unexpected nvidia-smi output: {first_line!r}"
    memory_mib = _number(columns[1])
    return {
        "gpu_model": columns[0] or None,
        "vram_total_bytes": (
            int(memory_mib * 1024 * 1024) if memory_mib is not None else None
        ),
        "nvidia_driver_version": columns[2] or None,
    }, None


def collect_phase2_5_system_info(
    *,
    model_repository_id: str,
    model_revision: str | None,
    processor_revision: str | None,
    dtype: str,
    quantization: str,
    attention_backend: str,
    device: str = "cuda",
    model: Any = None,
    processor: Any = None,
    include_nvml: bool = True,
    module_loader: ModuleLoader = importlib.import_module,
    command_runner: CommandRunner | None = None,
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    """Collect stable run metadata, degrading to nulls when optional APIs fail."""

    warnings: list[str] = []
    torch, torch_error = _load_optional(module_loader, "torch")
    transformers, transformers_error = _load_optional(module_loader, "transformers")
    if torch_error:
        warnings.append(f"torch metadata unavailable: {torch_error}")
    if transformers_error:
        warnings.append(f"transformers metadata unavailable: {transformers_error}")

    torch_version = _text(getattr(torch, "__version__", None)) if torch else None
    transformers_version = (
        _text(getattr(transformers, "__version__", None)) if transformers else None
    )
    cuda_runtime = (
        _text(getattr(getattr(torch, "version", None), "cuda", None)) if torch else None
    )
    cuda_available = False
    gpu_model = None
    vram_total_bytes = None
    bf16_supported = None
    index = _device_index(device)
    cuda = getattr(torch, "cuda", None) if torch else None
    if cuda is not None:
        try:
            cuda_available = bool(cuda.is_available())
        except Exception as error:
            warnings.append(
                "PyTorch CUDA availability unavailable: "
                f"{type(error).__name__}: {error}"
            )
        if cuda_available:
            try:
                gpu_model = _text(cuda.get_device_name(index))
                properties = cuda.get_device_properties(index)
                vram_total_bytes = int(properties.total_memory)
            except Exception as error:
                warnings.append(
                    f"PyTorch GPU metadata unavailable: {type(error).__name__}: {error}"
                )
            try:
                bf16_supported = bool(cuda.is_bf16_supported())
            except Exception as error:
                warnings.append(
                    "PyTorch BF16 capability unavailable: "
                    f"{type(error).__name__}: {error}"
                )

    nvml_values = {
        "nvml_available": False,
        "nvml_error": None,
        "gpu_model": None,
        "vram_total_bytes": None,
        "nvidia_driver_version": None,
        "gpu_utilization_percent": None,
        "gpu_power_draw_watts": None,
        "gpu_temperature_celsius": None,
    }
    if include_nvml:
        nvml, nvml_import_error = _load_optional(module_loader, "pynvml")
        if nvml is not None:
            nvml_values = _collect_nvml(nvml, index)
        else:
            nvml_values["nvml_error"] = nvml_import_error

    gpu_model = gpu_model or nvml_values["gpu_model"]
    vram_total_bytes = vram_total_bytes or nvml_values["vram_total_bytes"]
    driver_version = nvml_values["nvidia_driver_version"]
    if gpu_model is None or vram_total_bytes is None or driver_version is None:
        smi_values, smi_error = _collect_nvidia_smi(
            command_runner or _default_command_runner, index
        )
        gpu_model = gpu_model or smi_values.get("gpu_model")
        vram_total_bytes = vram_total_bytes or smi_values.get("vram_total_bytes")
        driver_version = driver_version or smi_values.get("nvidia_driver_version")
        if smi_error:
            warnings.append(f"nvidia-smi metadata unavailable: {smi_error}")
    if nvml_values["nvml_error"]:
        warnings.append(f"optional NVML telemetry unavailable: {nvml_values['nvml_error']}")

    timestamp = captured_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    info = {
        "schema_version": SYSTEM_INFO_SCHEMA_VERSION,
        "captured_at_utc": timestamp.astimezone(timezone.utc).isoformat(),
        "os": platform.platform(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "cuda_runtime_visible_to_torch": cuda_runtime,
        "cuda_available": cuda_available,
        "bf16_supported": bf16_supported,
        "gpu_model": gpu_model,
        "vram_total_bytes": vram_total_bytes,
        "nvidia_driver_version": driver_version,
        "gpu_utilization_percent": nvml_values["gpu_utilization_percent"],
        "gpu_power_draw_watts": nvml_values["gpu_power_draw_watts"],
        "gpu_temperature_celsius": nvml_values["gpu_temperature_celsius"],
        "nvml_available": nvml_values["nvml_available"],
        "model_repository_id": model_repository_id,
        "model_revision": resolve_component_revision(model, model_revision),
        "processor_revision": resolve_component_revision(
            processor, processor_revision
        ),
        "dtype": str(dtype),
        "quantization": str(quantization),
        "attention_backend": str(attention_backend),
        "device": str(device),
        "metadata_warnings": warnings,
    }
    return json_safe(info)


def write_phase2_5_system_info(path: str | Path, metadata: Mapping[str, Any]) -> None:
    """Atomically write strict JSON metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    rendered = json.dumps(
        json_safe(metadata),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


# Short aliases for callers that already namespace imports by module.
collect_system_info = collect_phase2_5_system_info
write_system_info = write_phase2_5_system_info
hf_cache_root = huggingface_cache_root
hf_model_cache_path = huggingface_model_cache_path
hf_cache_preflight = huggingface_cache_preflight
