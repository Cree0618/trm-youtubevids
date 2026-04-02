from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import torch


COMPILER_BUILD_ID = "compiler-colab-v1"


def collect_runtime_info() -> dict:
    module = importlib.import_module("trm_compiler")
    module_path = str(Path(module.__file__).resolve())
    editable_root = str(Path(__file__).resolve().parents[1])
    return {
        "build_id": COMPILER_BUILD_ID,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compiler_gym_available": bool(importlib.util.find_spec("compiler_gym")),
        "module_path": module_path,
        "expected_project_root": editable_root,
        "imported_from_project": module_path.startswith(editable_root),
    }


def run_preflight(require_compiler_gym: bool = False) -> dict:
    runtime = collect_runtime_info()
    errors: list[str] = []
    warnings: list[str] = []

    if not runtime["imported_from_project"]:
        errors.append("trm_compiler is not being imported from the current project root. Reinstall with `pip install -e .` and restart runtime.")
    if require_compiler_gym and not runtime["compiler_gym_available"]:
        errors.append("compiler_gym is not installed in this runtime.")
    if not require_compiler_gym and not runtime["compiler_gym_available"]:
        warnings.append("compiler_gym is not installed. The notebook will run in synthetic mode only.")

    result = {
        "runtime": runtime,
        "warnings": warnings,
        "errors": errors,
        "ok": len(errors) == 0,
    }
    return result


def print_preflight(require_compiler_gym: bool = False) -> dict:
    result = run_preflight(require_compiler_gym=require_compiler_gym)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise RuntimeError("Compiler preflight failed. See JSON output above.")
    return result
