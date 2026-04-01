from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .backend import TritonGemmBackend
from .schedules import all_edits_for_schedule, apply_edit, default_schedule
from .types import GemmSchedule, GemmTaskSpec, GpuTarget, TraceRecord


@dataclass
class TraceGenerationConfig:
    seeds_per_task: int = 3
    max_steps_per_seed: int = 4
    invalid_samples_per_task: int = 2


def reward_from_feedback(prev_runtime_us: float, next_runtime_us: float) -> float:
    if not torch.isfinite(torch.tensor(prev_runtime_us)):
        prev_runtime_us = 1e9
    if not torch.isfinite(torch.tensor(next_runtime_us)):
        next_runtime_us = 1e9
    return prev_runtime_us - next_runtime_us


def generate_trace_records(
    tasks: list[GemmTaskSpec],
    gpu_target: GpuTarget,
    backend: TritonGemmBackend,
    config: TraceGenerationConfig | None = None,
    seed: int = 0,
) -> list[TraceRecord]:
    cfg = config or TraceGenerationConfig()
    rng = random.Random(seed)
    records: list[TraceRecord] = []

    for task in tasks:
        for _ in range(cfg.seeds_per_task):
            state = _random_legal_schedule(gpu_target, backend, task, rng)
            feedback = backend.evaluate(task, state, gpu_target)
            for _ in range(cfg.max_steps_per_seed):
                edits = all_edits_for_schedule(state, gpu_target)
                if not edits:
                    break
                scored = []
                for edit in edits:
                    nxt = apply_edit(state, edit)
                    nxt_feedback = backend.evaluate(task, nxt, gpu_target)
                    scored.append((reward_from_feedback(feedback.runtime_us, nxt_feedback.runtime_us), edit, nxt, nxt_feedback))
                scored.sort(key=lambda item: item[0], reverse=True)
                reward, edit, next_state, next_feedback = scored[0]
                records.append(
                    TraceRecord(
                        task=task,
                        gpu_target=gpu_target,
                        state_t=state,
                        feedback_t=feedback,
                        edit_t=edit,
                        state_t1=next_state,
                        feedback_t1=next_feedback,
                        reward=reward,
                    )
                )
                state, feedback = next_state, next_feedback

        for _ in range(cfg.invalid_samples_per_task):
            bad = default_schedule().with_update("NUM_STAGES", 4).with_update("BLOCK_M", 128).with_update("BLOCK_N", 128)
            feedback = backend.evaluate(task, bad, gpu_target)
            edit = all_edits_for_schedule(default_schedule(), gpu_target)[0]
            next_state = apply_edit(default_schedule(), edit)
            next_feedback = backend.evaluate(task, next_state, gpu_target)
            records.append(
                TraceRecord(
                    task=task,
                    gpu_target=gpu_target,
                    state_t=bad,
                    feedback_t=feedback,
                    edit_t=edit,
                    state_t1=next_state,
                    feedback_t1=next_feedback,
                    reward=reward_from_feedback(feedback.runtime_us, next_feedback.runtime_us),
                )
            )
    return records


def _random_legal_schedule(
    gpu_target: GpuTarget, backend: TritonGemmBackend, task: GemmTaskSpec, rng: random.Random
) -> GemmSchedule:
    schedule = default_schedule()
    edits = all_edits_for_schedule(schedule, gpu_target)
    rng.shuffle(edits)
    for edit in edits[: rng.randint(1, min(4, len(edits)))]:
        candidate = apply_edit(schedule, edit)
        if backend.evaluate(task, candidate, gpu_target).compiled:
            schedule = candidate
    return schedule


class TraceDataset(Dataset[TraceRecord]):
    def __init__(self, records: list[TraceRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> TraceRecord:
        return self.records[idx]

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record.to_dict()) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TraceDataset":
        path = Path(path)
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(TraceRecord.from_dict(json.loads(line)))
        return cls(records)
