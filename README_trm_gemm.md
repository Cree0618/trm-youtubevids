# Portable TRM GEMM Refiner

This is a minimal research prototype for a `TRM-style recursive schedule refiner`
for `GEMM only`, designed around a `portable architecture schema` and a
`Turing-first` execution strategy.

## What is included

- Stable public types:
  - `GpuTarget`
  - `GemmTaskSpec`
  - `GemmSchedule`
  - `KernelFeedback`
  - `ScheduleEdit`
  - `TraceRecord`
- Conservative Turing-safe schedule space
- Optional Triton backend surface with a deterministic heuristic evaluator
- Offline trace generation
- Tiny recursive model with edit / feasibility / value / halt heads
- Tests for schema, legality, trace generation, and recursive rollout

## Why there is a heuristic backend

The current local environment does not have Triton installed. To keep the
prototype runnable and testable, the backend exposes the intended Triton-facing
interface but uses a deterministic heuristic evaluator when Triton/CUDA are not
available.

This keeps the architecture correct while leaving the real GPU execution path as
an additive backend improvement.

## Example

```bash
python -m trm_gemm.example
pytest -q
```

## Local vs Colab workflow

This project is designed for:

- local development on a non-NVIDIA machine such as an M3 MacBook Air
- GPU execution and future real benchmarking on Google Colab with a `T4`

Recommended split:

```bash
# local Mac development
python -m trm_gemm.example
pytest -q
```

```bash
# Colab / T4 setup
pip install -e .
pip install triton
python -m trm_gemm.colab_smoke
```

On local Apple Silicon, the backend will use the deterministic heuristic mode.
On a Colab T4, the intended next step is to wire in the real Triton execution
path behind the same backend interface.

Expected Colab behavior:

- `backend_mode` should print `triton_cuda` if both CUDA and Triton are visible
- if Triton is missing but CUDA is present, it will print `cuda_heuristic`
- the smoke script should still complete end to end and print generated traces,
  losses, baseline runtimes, and rollout edits

Minimal Colab cell:

```python
!pip install -e .
!pip install -r requirements-colab.txt
!python -m trm_gemm.colab_smoke
```

## Current scope

- `GEMM only`
- `offline traces`
- `Turing-first`
- `portable schema`

## Out of scope in this version

- fused pointwise epilogues
- fused reductions
- BF16
- tensor-core-specific search spaces
- raw CUDA generation
