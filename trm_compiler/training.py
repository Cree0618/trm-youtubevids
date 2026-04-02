"""Training loop for compiler pass ordering."""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import TinyPassOrderingRefiner


def compute_compiler_losses(
    model: TinyPassOrderingRefiner,
    batch: dict[str, torch.Tensor],
    device: str = "cpu",
    entropy_coef: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Compute multi-head training loss for one batch.

    Losses:
    - Pass selection: cross-entropy on which pass to apply
    - Feasibility: BCE on whether pass compiled
    - Value: MSE on predicted reward
    - Halt: BCE on halt decision
    - Entropy: penalty for low-entropy (prevents logit collapse)

    Args:
        model: The TRM pass ordering model
        batch: Dict with observation, schedule, feedback, pass_id, reward
        device: Torch device
        entropy_coef: Coefficient for entropy regularization

    Returns:
        Dict of named losses and total loss
    """
    observation = batch["observation"].to(device)
    schedule = batch["schedule"].to(device)
    feedback = batch["feedback"].to(device)
    target_pass_id = batch["pass_id"].to(device)
    reward = batch["reward"].to(device)

    batch_size = observation.shape[0]

    # Initialize latents
    y, z = model.init_latent(batch_size, device)

    # Forward pass with inner recursion
    outputs = model(observation, schedule, feedback, y, z)

    # Pass selection loss (cross-entropy)
    pass_loss = F.cross_entropy(outputs["pass_logits"], target_pass_id)

    # Entropy regularization: penalize low-entropy outputs
    pass_probs = F.softmax(outputs["pass_logits"], dim=-1)
    log_probs = F.log_softmax(outputs["pass_logits"], dim=-1)
    entropy = -(pass_probs * log_probs).sum(dim=-1).mean()
    entropy_loss = -entropy_coef * entropy  # negative because we want to maximize entropy

    # Feasibility loss (BCE): compiled = reward > -2.0 (heuristic cutoff)
    compiled_target = (reward > -2.0).float()
    feasibility_loss = F.binary_cross_entropy(outputs["feasibility"], compiled_target)

    # Value loss (MSE): predict the reward
    value_loss = F.mse_loss(outputs["value"], reward)

    # Halt loss: learn to halt when reward is very negative
    halt_target = (reward < -1.0).float()
    halt_loss = F.binary_cross_entropy_with_logits(
        outputs["halt_logit"], halt_target
    )

    # Weighted combination
    total = pass_loss + entropy_loss + 0.5 * feasibility_loss + 0.3 * value_loss + 0.2 * halt_loss

    return {
        "pass_loss": pass_loss,
        "entropy_loss": entropy_loss,
        "feasibility_loss": feasibility_loss,
        "value_loss": value_loss,
        "halt_loss": halt_loss,
        "entropy": entropy,
        "total_loss": total,
    }


def train_one_epoch(
    model: TinyPassOrderingRefiner,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    totals = {
        "pass_loss": 0.0,
        "entropy_loss": 0.0,
        "feasibility_loss": 0.0,
        "value_loss": 0.0,
        "halt_loss": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
    }
    n_batches = 0

    for batch in dataloader:
        losses = compute_compiler_losses(model, batch, device)

        optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k, v in losses.items():
            totals[k] += v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_model(
    model: TinyPassOrderingRefiner,
    benchmarks: list[str],
    device: str = "cpu",
    use_heuristic: bool = True,
) -> dict[str, float]:
    """Evaluate model on benchmarks using heuristic or real CompilerGym."""
    from .env_wrapper import SyntheticCompilerEnv, CompilerGymWrapper, make_compiler_env
    from .model import rollout_pass_optimizer
    import numpy as np

    model.eval()
    rewards = []

    for bench_id in benchmarks:
        env = make_compiler_env(
            benchmark_id=bench_id,
            use_compilergym=not use_heuristic,
        )
        env.open() if hasattr(env, 'open') else None
        obs, _ = env.reset()

        trace = rollout_pass_optimizer(model, obs, max_steps=30, device=device)

        total_reward = sum(s.get("value", 0.0) for s in trace)
        num_passes = len([s for s in trace if s["pass_id"] >= 0])
        rewards.append(total_reward)

        if hasattr(env, 'close'):
            env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
    }
