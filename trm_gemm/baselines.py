from __future__ import annotations

import random

from .backend import TritonGemmBackend
from .schedules import all_edits_for_schedule, apply_edit, default_schedule
from .types import GemmSchedule, GemmTaskSpec, GpuTarget


def random_search(
    backend: TritonGemmBackend, task: GemmTaskSpec, gpu_target: GpuTarget, budget: int = 8, seed: int = 0
):
    rng = random.Random(seed)
    best_schedule = default_schedule()
    best_feedback = backend.evaluate(task, best_schedule, gpu_target)
    for _ in range(budget):
        edits = all_edits_for_schedule(best_schedule, gpu_target)
        if not edits:
            break
        edit = rng.choice(edits)
        candidate = apply_edit(best_schedule, edit)
        feedback = backend.evaluate(task, candidate, gpu_target)
        if feedback.compiled and feedback.runtime_us < best_feedback.runtime_us:
            best_schedule, best_feedback = candidate, feedback
    return best_schedule, best_feedback


def greedy_search(
    backend: TritonGemmBackend, task: GemmTaskSpec, gpu_target: GpuTarget, start: GemmSchedule | None = None, budget: int = 8
):
    schedule = start or default_schedule()
    feedback = backend.evaluate(task, schedule, gpu_target)
    for _ in range(budget):
        edits = all_edits_for_schedule(schedule, gpu_target)
        scored = []
        for edit in edits:
            candidate = apply_edit(schedule, edit)
            cand_feedback = backend.evaluate(task, candidate, gpu_target)
            scored.append((cand_feedback.runtime_us, candidate, cand_feedback))
        if not scored:
            break
        scored.sort(key=lambda item: item[0])
        next_runtime, next_schedule, next_feedback = scored[0]
        if next_runtime >= feedback.runtime_us:
            break
        schedule, feedback = next_schedule, next_feedback
    return schedule, feedback
