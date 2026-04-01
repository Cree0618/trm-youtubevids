from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import edit_to_index, encode_feedback, encode_schedule, encode_task
from .types import TraceRecord


def compute_losses(model, batch: list[TraceRecord]) -> dict[str, torch.Tensor]:
    task_vec = torch.stack([encode_task(record.task, record.gpu_target) for record in batch], dim=0)
    sched_vec = torch.stack([encode_schedule(record.state_t) for record in batch], dim=0)
    feedback_vec = torch.stack([encode_feedback(record.feedback_t) for record in batch], dim=0)
    latent = model.init_latent(batch_size=len(batch), device=task_vec.device)
    outputs = model(task_vec, sched_vec, feedback_vec, latent)

    edit_target = torch.tensor([edit_to_index(record.edit_t) for record in batch], dtype=torch.long)
    feasible_target = torch.tensor(
        [1.0 if record.feedback_t1.compiled and record.feedback_t1.correct else 0.0 for record in batch],
        dtype=torch.float32,
    )
    value_target = torch.tensor([record.reward for record in batch], dtype=torch.float32)
    halt_target = torch.tensor([1.0 if record.reward <= 0.0 else 0.0 for record in batch], dtype=torch.float32)

    losses = {
        "edit": F.cross_entropy(outputs["edit_logits"], edit_target),
        "feasibility": F.binary_cross_entropy_with_logits(outputs["feasibility_logit"], feasible_target),
        "value": F.mse_loss(outputs["value"], value_target),
        "halt": F.binary_cross_entropy_with_logits(outputs["halt_logit"], halt_target),
    }
    losses["total"] = losses["edit"] + losses["feasibility"] + 0.25 * losses["value"] + 0.25 * losses["halt"]
    return losses
