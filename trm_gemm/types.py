from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class FailureReason(str, Enum):
    INVALID_SCHEDULE = "invalid_schedule"
    COMPILE_ERROR = "compile_error"
    RESOURCE_EXCEEDED = "resource_exceeded"
    INCORRECT_OUTPUT = "incorrect_output"
    RUNTIME_ERROR = "runtime_error"
    VALID = "valid"


@dataclass(frozen=True)
class GpuTarget:
    arch_family: str
    compute_capability: str
    max_shared_mem_bytes: int
    max_registers_per_thread: int
    warp_size: int = 32
    tensor_cores: bool = False
    supported_dtypes: tuple[str, ...] = ("fp32", "fp16")
    instruction_families: tuple[str, ...] = ("simt",)
    arch_extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["supported_dtypes"] = list(self.supported_dtypes)
        data["instruction_families"] = list(self.instruction_families)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GpuTarget":
        payload = dict(data)
        payload["supported_dtypes"] = tuple(payload.get("supported_dtypes", ()))
        payload["instruction_families"] = tuple(payload.get("instruction_families", ()))
        return cls(**payload)


@dataclass(frozen=True)
class GemmTaskSpec:
    m: int
    n: int
    k: int
    dtype_a: str = "fp32"
    dtype_b: str = "fp32"
    dtype_acc: str = "fp32"
    dtype_out: str = "fp32"
    layout_a: str = "row_major"
    layout_b: str = "col_major"
    layout_c: str = "row_major"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GemmTaskSpec":
        return cls(**data)


@dataclass(frozen=True)
class GemmSchedule:
    portable_core: dict[str, int]
    arch_extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"portable_core": dict(self.portable_core), "arch_extensions": dict(self.arch_extensions)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GemmSchedule":
        return cls(
            portable_core=dict(data["portable_core"]),
            arch_extensions=dict(data.get("arch_extensions", {})),
        )

    def with_update(self, field_name: str, value: int) -> "GemmSchedule":
        core = dict(self.portable_core)
        core[field_name] = value
        return GemmSchedule(portable_core=core, arch_extensions=dict(self.arch_extensions))


@dataclass(frozen=True)
class KernelFeedback:
    compiled: bool
    correct: bool
    runtime_us: float
    normalized_tflops: float
    registers_per_thread: int
    shared_mem_bytes: int
    occupancy: float | None
    failure_reason: FailureReason

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_reason"] = self.failure_reason.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KernelFeedback":
        payload = dict(data)
        payload["failure_reason"] = FailureReason(payload["failure_reason"])
        return cls(**payload)


@dataclass(frozen=True)
class ScheduleEdit:
    field_name: str
    value: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduleEdit":
        return cls(**data)


@dataclass(frozen=True)
class TraceRecord:
    task: GemmTaskSpec
    gpu_target: GpuTarget
    state_t: GemmSchedule
    feedback_t: KernelFeedback
    edit_t: ScheduleEdit
    state_t1: GemmSchedule
    feedback_t1: KernelFeedback
    reward: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "gpu_target": self.gpu_target.to_dict(),
            "state_t": self.state_t.to_dict(),
            "feedback_t": self.feedback_t.to_dict(),
            "edit_t": self.edit_t.to_dict(),
            "state_t1": self.state_t1.to_dict(),
            "feedback_t1": self.feedback_t1.to_dict(),
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceRecord":
        return cls(
            task=GemmTaskSpec.from_dict(data["task"]),
            gpu_target=GpuTarget.from_dict(data["gpu_target"]),
            state_t=GemmSchedule.from_dict(data["state_t"]),
            feedback_t=KernelFeedback.from_dict(data["feedback_t"]),
            edit_t=ScheduleEdit.from_dict(data["edit_t"]),
            state_t1=GemmSchedule.from_dict(data["state_t1"]),
            feedback_t1=KernelFeedback.from_dict(data["feedback_t1"]),
            reward=float(data["reward"]),
        )


RTX_1650 = GpuTarget(
    arch_family="turing",
    compute_capability="7.5",
    max_shared_mem_bytes=48 * 1024,
    max_registers_per_thread=255,
    tensor_cores=False,
    supported_dtypes=("fp32", "fp16"),
    instruction_families=("simt",),
)

T4 = GpuTarget(
    arch_family="turing",
    compute_capability="7.5",
    max_shared_mem_bytes=64 * 1024,
    max_registers_per_thread=255,
    tensor_cores=True,
    supported_dtypes=("fp32", "fp16"),
    instruction_families=("simt", "tensorcore"),
)
