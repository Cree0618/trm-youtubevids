from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .schedules import estimate_registers_per_thread, estimate_shared_mem_bytes, validate_schedule
from .types import FailureReason, GemmSchedule, GemmTaskSpec, GpuTarget, KernelFeedback

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except Exception:  # pragma: no cover - exercised by absence in local env
    triton = None
    tl = None


if triton is not None and tl is not None:  # pragma: no branch

    @triton.jit
    def _matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
            b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


@dataclass
class BackendCapabilities:
    triton_available: bool
    cuda_available: bool


class TritonGemmBackend:
    """
    Conservative Turing-first backend.

    It exposes the intended Triton execution surface, but falls back to a
    deterministic heuristic evaluator when Triton or CUDA are not available.
    """

    def __init__(self):
        self.capabilities = BackendCapabilities(
            triton_available=triton is not None,
            cuda_available=torch.cuda.is_available(),
        )
        self._tensor_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @property
    def mode(self) -> str:
        if self.capabilities.triton_available and self.capabilities.cuda_available:
            return "triton_cuda"
        if self.capabilities.cuda_available:
            return "cuda_heuristic"
        return "cpu_heuristic"

    def evaluate(self, task: GemmTaskSpec, schedule: GemmSchedule, gpu_target: GpuTarget) -> KernelFeedback:
        validation = validate_schedule(schedule, gpu_target)
        if not validation.valid:
            return KernelFeedback(
                compiled=False,
                correct=False,
                runtime_us=float("inf"),
                normalized_tflops=0.0,
                registers_per_thread=estimate_registers_per_thread(schedule),
                shared_mem_bytes=estimate_shared_mem_bytes(schedule),
                occupancy=0.0,
                failure_reason=FailureReason.INVALID_SCHEDULE
                if validation.reason != "shared_mem" and validation.reason != "registers"
                else FailureReason.RESOURCE_EXCEEDED,
            )
        if self.mode == "triton_cuda" and self._supports_real_triton(task, gpu_target):
            return self._triton_feedback(task, schedule, gpu_target)
        return self._heuristic_feedback(task, schedule, gpu_target)

    def _supports_real_triton(self, task: GemmTaskSpec, gpu_target: GpuTarget) -> bool:
        return (
            task.dtype_a == "fp32"
            and task.dtype_b == "fp32"
            and task.dtype_out == "fp32"
            and task.layout_a == "row_major"
            and task.layout_b == "col_major"
            and task.layout_c == "row_major"
            and gpu_target.arch_family in {"turing", "ampere", "hopper", "blackwell"}
        )

    def _tensor_key(self, task: GemmTaskSpec, device: str) -> tuple:
        return (
            device,
            task.m,
            task.n,
            task.k,
            task.dtype_a,
            task.dtype_b,
            task.dtype_out,
            task.layout_a,
            task.layout_b,
            task.layout_c,
        )

    def _get_task_tensors(self, task: GemmTaskSpec, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = self._tensor_key(task, device)
        if key not in self._tensor_cache:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(0)
            a_cpu = torch.randn((task.m, task.k), generator=gen, dtype=torch.float32)
            b_cpu = torch.randn((task.k, task.n), generator=gen, dtype=torch.float32)
            a = a_cpu.to(device=device).contiguous()
            b = b_cpu.to(device=device).contiguous()
            ref = torch.matmul(a, b)
            self._tensor_cache[key] = (a, b, ref)
        return self._tensor_cache[key]

    def _launch_triton(self, a: torch.Tensor, b: torch.Tensor, schedule: GemmSchedule) -> torch.Tensor:
        core = schedule.portable_core
        c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
        grid = lambda META: (
            triton.cdiv(a.shape[0], META["BLOCK_M"]) * triton.cdiv(b.shape[1], META["BLOCK_N"]),
        )
        _matmul_kernel[grid](
            a,
            b,
            c,
            a.shape[0],
            b.shape[1],
            a.shape[1],
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=core["BLOCK_M"],
            BLOCK_N=core["BLOCK_N"],
            BLOCK_K=core["BLOCK_K"],
            GROUP_M=core["GROUP_M"],
            num_warps=core["NUM_WARPS"],
            num_stages=core["NUM_STAGES"],
        )
        return c

    def _benchmark_triton(
        self, a: torch.Tensor, b: torch.Tensor, schedule: GemmSchedule, warmup: int = 2, iters: int = 5
    ) -> float:
        for _ in range(warmup):
            _ = self._launch_triton(a, b, schedule)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = self._launch_triton(a, b, schedule)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / iters
        return elapsed_ms * 1000.0

    def _triton_feedback(self, task: GemmTaskSpec, schedule: GemmSchedule, gpu_target: GpuTarget) -> KernelFeedback:
        regs = estimate_registers_per_thread(schedule)
        smem = estimate_shared_mem_bytes(schedule)
        occupancy = min(
            1.0,
            max(
                0.05,
                1.2
                - regs / max(gpu_target.max_registers_per_thread, 1)
                - smem / max(gpu_target.max_shared_mem_bytes * 1.1, 1),
            ),
        )
        try:
            a, b, ref = self._get_task_tensors(task, "cuda")
            out = self._launch_triton(a, b, schedule)
            torch.cuda.synchronize()
            if not torch.allclose(out, ref, atol=1e-3, rtol=1e-3):
                return KernelFeedback(
                    compiled=True,
                    correct=False,
                    runtime_us=float("inf"),
                    normalized_tflops=0.0,
                    registers_per_thread=regs,
                    shared_mem_bytes=smem,
                    occupancy=occupancy,
                    failure_reason=FailureReason.INCORRECT_OUTPUT,
                )
            runtime_us = self._benchmark_triton(a, b, schedule)
            tflops = (2.0 * task.m * task.n * task.k) / (runtime_us * 1e-6) * 1e-12
            return KernelFeedback(
                compiled=True,
                correct=True,
                runtime_us=runtime_us,
                normalized_tflops=tflops,
                registers_per_thread=regs,
                shared_mem_bytes=smem,
                occupancy=occupancy,
                failure_reason=FailureReason.VALID,
            )
        except torch.cuda.OutOfMemoryError:
            return KernelFeedback(
                compiled=False,
                correct=False,
                runtime_us=float("inf"),
                normalized_tflops=0.0,
                registers_per_thread=regs,
                shared_mem_bytes=smem,
                occupancy=0.0,
                failure_reason=FailureReason.RESOURCE_EXCEEDED,
            )
        except Exception:
            return KernelFeedback(
                compiled=False,
                correct=False,
                runtime_us=float("inf"),
                normalized_tflops=0.0,
                registers_per_thread=regs,
                shared_mem_bytes=smem,
                occupancy=0.0,
                failure_reason=FailureReason.COMPILE_ERROR,
            )

    def _heuristic_feedback(
        self, task: GemmTaskSpec, schedule: GemmSchedule, gpu_target: GpuTarget
    ) -> KernelFeedback:
        core = schedule.portable_core
        regs = estimate_registers_per_thread(schedule)
        smem = estimate_shared_mem_bytes(schedule)
        occupancy = min(
            1.0,
            max(
                0.1,
                1.2
                - regs / max(gpu_target.max_registers_per_thread, 1)
                - smem / max(gpu_target.max_shared_mem_bytes * 1.1, 1),
            ),
        )
        problem_scale = (task.m * task.n * task.k) / 1e6
        block_fit = min(task.m / core["BLOCK_M"], 1.0) + min(task.n / core["BLOCK_N"], 1.0)
        reuse_bonus = min(core["BLOCK_K"] / 32.0, 2.0)
        split_penalty = 0.15 * (core["SPLIT_K"] - 1)
        vec_bonus = 0.03 * (core["VEC_A"] + core["VEC_B"] - 2)
        stage_bonus = 0.06 * min(core["NUM_STAGES"], 3)
        warp_penalty = 0.04 * max(core["NUM_WARPS"] - 4, 0)
        size_penalty = 0.08 * abs(math.log2(core["BLOCK_M"] / max(core["BLOCK_N"], 1)))

        score = 0.8 + 0.25 * block_fit + 0.15 * reuse_bonus + vec_bonus + stage_bonus
        score += occupancy - split_penalty - warp_penalty - size_penalty
        runtime_us = max(5.0, (problem_scale / max(score, 0.2)) * 3.2)
        tflops = (2.0 * task.m * task.n * task.k) / max(runtime_us, 1e-6) / 1e6

        return KernelFeedback(
            compiled=True,
            correct=True,
            runtime_us=runtime_us,
            normalized_tflops=tflops / 1000.0,
            registers_per_thread=regs,
            shared_mem_bytes=smem,
            occupancy=occupancy,
            failure_reason=FailureReason.VALID,
        )
