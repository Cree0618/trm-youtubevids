from __future__ import annotations

from trm_gemm.backend import TritonGemmBackend
from trm_gemm.data import TraceDataset, TraceGenerationConfig, generate_trace_records
from trm_gemm.model import TinyRecursiveGemmRefiner, legal_edit_mask, rollout_refiner
from trm_gemm.schedules import apply_edit, default_schedule, validate_schedule
from trm_gemm.training import compute_losses
from trm_gemm.types import GemmSchedule, GemmTaskSpec, GpuTarget, RTX_1650, T4, TraceRecord


def test_schema_roundtrip():
    backend = TritonGemmBackend()
    task = GemmTaskSpec(256, 512, 384)
    schedule = default_schedule()
    feedback = backend.evaluate(task, schedule, RTX_1650)
    record = TraceRecord(
        task=task,
        gpu_target=RTX_1650,
        state_t=schedule,
        feedback_t=feedback,
        edit_t=rollout_refiner(TinyRecursiveGemmRefiner(), task, RTX_1650, schedule, feedback, steps=1)[0],
        state_t1=schedule,
        feedback_t1=feedback,
        reward=0.0,
    )
    payload = record.to_dict()
    restored = TraceRecord.from_dict(payload)
    assert restored.to_dict() == payload


def test_arch_extensions_portable():
    target = GpuTarget(
        arch_family="hopper",
        compute_capability="9.0",
        max_shared_mem_bytes=228 * 1024,
        max_registers_per_thread=255,
        tensor_cores=True,
        supported_dtypes=("fp32", "fp16", "bf16"),
        instruction_families=("simt", "tensorcore", "wgmma"),
        arch_extensions={"wgmma": True},
    )
    payload = target.to_dict()
    restored = GpuTarget.from_dict(payload)
    assert restored.arch_extensions["wgmma"] is True


def test_turing_safe_schedule_validation():
    schedule = default_schedule()
    assert validate_schedule(schedule, RTX_1650).valid
    bad = GemmSchedule(portable_core={**schedule.portable_core, "BLOCK_M": 96})
    assert not validate_schedule(bad, RTX_1650).valid


def test_action_mask_is_architecture_aware():
    schedule = default_schedule()
    mask_1650 = legal_edit_mask(schedule, RTX_1650)
    mask_t4 = legal_edit_mask(schedule, T4)
    assert mask_1650.shape == mask_t4.shape
    assert mask_1650.any()


def test_trace_generation_contains_valid_and_invalid_records(tmp_path):
    backend = TritonGemmBackend()
    tasks = [GemmTaskSpec(512, 512, 512), GemmTaskSpec(768, 512, 256)]
    records = generate_trace_records(tasks, RTX_1650, backend, TraceGenerationConfig(seeds_per_task=2, max_steps_per_seed=2))
    assert records
    assert any(r.feedback_t.compiled for r in records)
    assert any(not r.feedback_t.compiled for r in records)

    dataset = TraceDataset(records)
    path = tmp_path / "traces.jsonl"
    dataset.to_jsonl(path)
    restored = TraceDataset.from_jsonl(path)
    assert len(restored) == len(dataset)


def test_recursive_rollout_emits_legal_edits():
    backend = TritonGemmBackend()
    model = TinyRecursiveGemmRefiner()
    task = GemmTaskSpec(512, 512, 512)
    schedule = default_schedule()
    feedback = backend.evaluate(task, schedule, RTX_1650)
    edits = rollout_refiner(model, task, RTX_1650, schedule, feedback, steps=3)
    current = schedule
    for edit in edits:
        current = apply_edit(current, edit)
        assert validate_schedule(current, RTX_1650).valid


def test_training_losses_are_finite():
    backend = TritonGemmBackend()
    tasks = [GemmTaskSpec(256, 256, 256), GemmTaskSpec(512, 256, 384)]
    records = generate_trace_records(tasks, RTX_1650, backend, TraceGenerationConfig(seeds_per_task=1, max_steps_per_seed=2))
    model = TinyRecursiveGemmRefiner()
    losses = compute_losses(model, records[:4])
    assert losses["total"].item() >= 0.0
