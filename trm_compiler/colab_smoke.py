from __future__ import annotations

from .data import CompilerTraceDataset, generate_compiler_traces
from .env_check import print_preflight
from .model import TinyPassOrderingRefiner
from .training import train_one_epoch

import torch
from torch.utils.data import DataLoader


def main() -> None:
    print_preflight(require_compiler_gym=False)
    traces = generate_compiler_traces(
        benchmarks=["qsort", "adpcm"],
        episodes_per_benchmark=4,
        max_steps_per_episode=10,
        use_heuristic=True,
        seed=42,
        strategy="mixed",
    )
    print(f"generated_traces={len(traces)}")
    dataset = CompilerTraceDataset(traces)
    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True, num_workers=0, drop_last=False)
    model = TinyPassOrderingRefiner()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = train_one_epoch(model, loader, optimizer, "cpu")
    print({k: round(v, 4) for k, v in losses.items()})


if __name__ == "__main__":
    main()
