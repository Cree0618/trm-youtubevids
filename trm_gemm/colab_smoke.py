from __future__ import annotations

import json

import torch

from .backend import TritonGemmBackend
from .baselines import greedy_search, random_search
from .data import TraceGenerationConfig, generate_trace_records
from .model import TinyRecursiveGemmRefiner, rollout_refiner
from .training import compute_losses
from .types import GemmTaskSpec, T4


def main() -> None:
    backend = TritonGemmBackend()
    print(json.dumps(
        {
            "backend_mode": backend.mode,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "triton_available": backend.capabilities.triton_available,
        },
        indent=2,
    ))

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. This smoke script is intended for Colab T4.")

    tasks = [
        GemmTaskSpec(512, 512, 512),
        GemmTaskSpec(1024, 512, 768),
        GemmTaskSpec(1536, 1024, 640),
    ]
    records = generate_trace_records(tasks, T4, backend, TraceGenerationConfig(seeds_per_task=2, max_steps_per_seed=2))
    print(f"generated_trace_records={len(records)}")

    model = TinyRecursiveGemmRefiner()
    batch = records[: min(4, len(records))]
    losses = compute_losses(model, batch)
    print("losses", {k: round(v.item(), 4) for k, v in losses.items()})

    task = tasks[0]
    random_schedule, random_feedback = random_search(backend, task, T4, budget=6, seed=0)
    greedy_schedule, greedy_feedback = greedy_search(backend, task, T4, budget=6)
    edits = rollout_refiner(model, task, T4, random_schedule, random_feedback, steps=3)

    print("random_runtime_us", round(random_feedback.runtime_us, 4))
    print("greedy_runtime_us", round(greedy_feedback.runtime_us, 4))
    print("rollout_edits", [edit.to_dict() for edit in edits])
    print("done")


if __name__ == "__main__":
    main()
