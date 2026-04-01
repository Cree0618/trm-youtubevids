from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
from pathlib import Path

import torch

from .backend import TritonGemmBackend
from .schedules import default_schedule
from .types import GemmTaskSpec, T4


BACKEND_BUILD_ID = "triton-real-gemm-v1"


def collect_runtime_info() -> dict:
    backend = TritonGemmBackend()
    module = importlib.import_module("trm_gemm.backend")
    module_path = str(Path(module.__file__).resolve())
    editable_root = str(Path(__file__).resolve().parents[1])
    return {
        "build_id": BACKEND_BUILD_ID,
        "backend_mode": backend.mode,
        "cuda_available": torch.cuda.is_available(),
        "triton_available": backend.capabilities.triton_available,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "backend_module_path": module_path,
        "expected_project_root": editable_root,
        "imported_from_project": module_path.startswith(editable_root),
    }


def run_preflight(strict_t4: bool = True) -> dict:
    info = collect_runtime_info()
    backend = TritonGemmBackend()
    errors: list[str] = []
    warnings: list[str] = []

    if not info["cuda_available"]:
        errors.append("CUDA is not available.")
    if not info["triton_available"]:
        errors.append("Triton is not importable in the current runtime.")
    if strict_t4 and info["device_name"] and "T4" not in info["device_name"]:
        warnings.append(f"Expected T4 runtime, got {info['device_name']}.")
    if not info["imported_from_project"]:
        errors.append("trm_gemm is not being imported from the current project root. Reinstall with `pip install -e .` and restart runtime.")
    if info["backend_mode"] != "triton_cuda" and info["cuda_available"] and info["triton_available"]:
        errors.append("Backend did not enter triton_cuda mode even though CUDA and Triton are available.")

    sanity_feedback = None
    if not errors:
        task = GemmTaskSpec(256, 256, 256)
        sanity_feedback = backend.evaluate(task, default_schedule(), T4).to_dict()
        if not sanity_feedback["compiled"] or not sanity_feedback["correct"]:
            errors.append("Sanity GEMM kernel did not compile/correctly execute.")
        if sanity_feedback["runtime_us"] <= 0:
            errors.append("Sanity GEMM runtime_us is not positive.")
        if sanity_feedback["normalized_tflops"] <= 0:
            errors.append("Sanity GEMM normalized_tflops is not positive.")
        if sanity_feedback["normalized_tflops"] < 0.01:
            warnings.append(
                "Sanity GEMM throughput is suspiciously low; this often indicates a stale installed package or incorrect backend code path."
            )

    result = {
        "runtime": info,
        "sanity_feedback": sanity_feedback,
        "warnings": warnings,
        "errors": errors,
        "ok": len(errors) == 0,
    }
    return result


def print_preflight(strict_t4: bool = True) -> dict:
    result = run_preflight(strict_t4=strict_t4)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise RuntimeError("Preflight failed. See JSON output above.")
    return result
