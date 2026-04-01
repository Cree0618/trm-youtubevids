from __future__ import annotations

from .backend import TritonGemmBackend
from .data import TraceGenerationConfig, generate_trace_records
from .model import TinyRecursiveGemmRefiner
from .training import compute_losses
from .types import GemmTaskSpec, RTX_1650


def main() -> None:
    backend = TritonGemmBackend()
    tasks = [
        GemmTaskSpec(512, 512, 512),
        GemmTaskSpec(1024, 512, 768),
        GemmTaskSpec(1536, 1024, 640),
    ]
    records = generate_trace_records(tasks, RTX_1650, backend, TraceGenerationConfig(seeds_per_task=2, max_steps_per_seed=2))
    model = TinyRecursiveGemmRefiner()
    losses = compute_losses(model, records[: min(4, len(records))])
    print(f"records={len(records)} total_loss={losses['total'].item():.4f}")


if __name__ == "__main__":
    main()
