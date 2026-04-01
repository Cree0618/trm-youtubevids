from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .schedules import PORTABLE_FIELD_ORDER, PORTABLE_SEARCH_SPACE, apply_edit, default_schedule, validate_schedule
from .types import GemmSchedule, GemmTaskSpec, GpuTarget, KernelFeedback, ScheduleEdit


def _arch_family_id(name: str) -> int:
    vocab = {"turing": 0, "ampere": 1, "hopper": 2, "blackwell": 3}
    return vocab.get(name, 4)


def encode_task(task: GemmTaskSpec, gpu: GpuTarget) -> torch.Tensor:
    values = [
        float(task.m),
        float(task.n),
        float(task.k),
        1.0 if task.dtype_a == "fp16" else 0.0,
        1.0 if task.dtype_b == "fp16" else 0.0,
        float(_arch_family_id(gpu.arch_family)),
        float(int(float(gpu.compute_capability))),
        float(gpu.max_shared_mem_bytes),
    ]
    return torch.tensor(values, dtype=torch.float32)


def encode_schedule(schedule: GemmSchedule) -> torch.Tensor:
    return torch.tensor([float(schedule.portable_core[field]) for field in PORTABLE_FIELD_ORDER], dtype=torch.float32)


def encode_feedback(feedback: KernelFeedback) -> torch.Tensor:
    return torch.tensor(
        [
            1.0 if feedback.compiled else 0.0,
            1.0 if feedback.correct else 0.0,
            float(feedback.runtime_us if torch.isfinite(torch.tensor(feedback.runtime_us)) else 1e9),
            float(feedback.normalized_tflops),
            float(feedback.registers_per_thread),
            float(feedback.shared_mem_bytes),
            float(feedback.occupancy or 0.0),
            float(list(type(feedback.failure_reason)).index(feedback.failure_reason)),
        ],
        dtype=torch.float32,
    )


def all_possible_edits() -> list[ScheduleEdit]:
    edits = []
    for field in PORTABLE_FIELD_ORDER:
        for value in PORTABLE_SEARCH_SPACE[field]:
            edits.append(ScheduleEdit(field_name=field, value=value))
    return edits


ALL_EDITS = all_possible_edits()


def edit_to_index(edit: ScheduleEdit) -> int:
    return ALL_EDITS.index(edit)


@dataclass
class RolloutState:
    schedule: GemmSchedule
    latent: torch.Tensor


class TinyRecursiveGemmRefiner(nn.Module):
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        input_dim = 8 + len(PORTABLE_FIELD_ORDER) + 8 + latent_dim
        self.latent_dim = latent_dim
        self.reason = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.edit_head = nn.Linear(latent_dim, len(ALL_EDITS))
        self.feasibility_head = nn.Linear(latent_dim, 1)
        self.value_head = nn.Linear(latent_dim, 1)
        self.halt_head = nn.Linear(latent_dim, 1)

    def forward(
        self,
        task_vec: torch.Tensor,
        schedule_vec: torch.Tensor,
        feedback_vec: torch.Tensor,
        latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = torch.cat([task_vec, schedule_vec, feedback_vec, latent], dim=-1)
        next_latent = self.reason(x)
        return {
            "latent": next_latent,
            "edit_logits": self.edit_head(next_latent),
            "feasibility_logit": self.feasibility_head(next_latent).squeeze(-1),
            "value": self.value_head(next_latent).squeeze(-1),
            "halt_logit": self.halt_head(next_latent).squeeze(-1),
        }

    def init_latent(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(batch_size, self.latent_dim, device=device)


def legal_edit_mask(schedule: GemmSchedule, gpu_target: GpuTarget) -> torch.Tensor:
    mask = torch.zeros(len(ALL_EDITS), dtype=torch.bool)
    for idx, edit in enumerate(ALL_EDITS):
        if schedule.portable_core[edit.field_name] == edit.value:
            continue
        candidate = apply_edit(schedule, edit)
        if validate_schedule(candidate, gpu_target).valid:
            mask[idx] = True
    return mask


def select_legal_edit(logits: torch.Tensor, schedule: GemmSchedule, gpu_target: GpuTarget) -> ScheduleEdit:
    mask = legal_edit_mask(schedule, gpu_target)
    masked_logits = logits.clone()
    masked_logits[~mask] = -1e9
    idx = int(masked_logits.argmax().item())
    return ALL_EDITS[idx]


def rollout_refiner(
    model: TinyRecursiveGemmRefiner,
    task: GemmTaskSpec,
    gpu_target: GpuTarget,
    schedule: GemmSchedule | None,
    feedback: KernelFeedback,
    steps: int = 4,
) -> list[ScheduleEdit]:
    current_schedule = schedule or default_schedule()
    latent = model.init_latent()
    edits: list[ScheduleEdit] = []
    for _ in range(steps):
        outputs = model(
            encode_task(task, gpu_target).unsqueeze(0),
            encode_schedule(current_schedule).unsqueeze(0),
            encode_feedback(feedback).unsqueeze(0),
            latent,
        )
        latent = outputs["latent"]
        edit = select_legal_edit(outputs["edit_logits"].squeeze(0), current_schedule, gpu_target)
        edits.append(edit)
        current_schedule = apply_edit(current_schedule, edit)
    return edits
