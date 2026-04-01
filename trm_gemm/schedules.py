from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .types import GemmSchedule, GpuTarget, ScheduleEdit


PORTABLE_SEARCH_SPACE: dict[str, tuple[int, ...]] = {
    "BLOCK_M": (32, 64, 128),
    "BLOCK_N": (32, 64, 128),
    "BLOCK_K": (16, 32, 64),
    "NUM_WARPS": (2, 4, 8),
    "NUM_STAGES": (1, 2, 3, 4),
    "GROUP_M": (1, 4, 8),
    "SPLIT_K": (1, 2, 4),
    "VEC_A": (1, 2, 4),
    "VEC_B": (1, 2, 4),
}

PORTABLE_FIELD_ORDER = tuple(PORTABLE_SEARCH_SPACE.keys())


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str | None = None


def default_schedule() -> GemmSchedule:
    return GemmSchedule(
        portable_core={
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "NUM_WARPS": 4,
            "NUM_STAGES": 2,
            "GROUP_M": 4,
            "SPLIT_K": 1,
            "VEC_A": 1,
            "VEC_B": 1,
        }
    )


def legal_values(field_name: str) -> tuple[int, ...]:
    return PORTABLE_SEARCH_SPACE[field_name]


def enumerate_portable_schedules(limit: int | None = None) -> list[GemmSchedule]:
    combos = product(*(PORTABLE_SEARCH_SPACE[field] for field in PORTABLE_FIELD_ORDER))
    results: list[GemmSchedule] = []
    for idx, combo in enumerate(combos):
        if limit is not None and idx >= limit:
            break
        core = {field: value for field, value in zip(PORTABLE_FIELD_ORDER, combo)}
        results.append(GemmSchedule(portable_core=core))
    return results


def validate_schedule(schedule: GemmSchedule, gpu_target: GpuTarget) -> ValidationResult:
    core = schedule.portable_core
    for field in PORTABLE_FIELD_ORDER:
        if field not in core:
            return ValidationResult(False, f"missing_{field}")
        if core[field] not in PORTABLE_SEARCH_SPACE[field]:
            return ValidationResult(False, f"invalid_{field}")

    if core["BLOCK_M"] % 32 != 0 or core["BLOCK_N"] % 32 != 0:
        return ValidationResult(False, "block_alignment")
    if core["BLOCK_K"] not in (16, 32, 64):
        return ValidationResult(False, "block_k")
    if core["VEC_A"] > core["BLOCK_K"] or core["VEC_B"] > core["BLOCK_K"]:
        return ValidationResult(False, "vector_width")
    if core["NUM_WARPS"] * 32 > 256:
        return ValidationResult(False, "too_many_threads")

    estimated_smem = estimate_shared_mem_bytes(schedule)
    if estimated_smem > gpu_target.max_shared_mem_bytes:
        return ValidationResult(False, "shared_mem")

    estimated_regs = estimate_registers_per_thread(schedule)
    if estimated_regs > gpu_target.max_registers_per_thread:
        return ValidationResult(False, "registers")

    return ValidationResult(True, None)


def estimate_shared_mem_bytes(schedule: GemmSchedule) -> int:
    core = schedule.portable_core
    bytes_per_stage = (core["BLOCK_M"] * core["BLOCK_K"] + core["BLOCK_N"] * core["BLOCK_K"]) * 4
    return bytes_per_stage * core["NUM_STAGES"]


def estimate_registers_per_thread(schedule: GemmSchedule) -> int:
    core = schedule.portable_core
    accum = (core["BLOCK_M"] * core["BLOCK_N"]) // max(core["NUM_WARPS"] * 32, 1)
    reg_estimate = 16 + accum // 16 + core["BLOCK_K"] // 8 + core["NUM_STAGES"] * 4
    reg_estimate += core["VEC_A"] + core["VEC_B"] + core["SPLIT_K"] * 2
    return reg_estimate


def all_edits_for_schedule(schedule: GemmSchedule, gpu_target: GpuTarget) -> list[ScheduleEdit]:
    edits: list[ScheduleEdit] = []
    for field in PORTABLE_FIELD_ORDER:
        current = schedule.portable_core[field]
        for value in PORTABLE_SEARCH_SPACE[field]:
            if value == current:
                continue
            candidate = schedule.with_update(field, value)
            if validate_schedule(candidate, gpu_target).valid:
                edits.append(ScheduleEdit(field_name=field, value=value))
    return edits


def apply_edit(schedule: GemmSchedule, edit: ScheduleEdit) -> GemmSchedule:
    return schedule.with_update(edit.field_name, edit.value)
