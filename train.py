"""TRM Autoresearch — Single-file training script.

THIS IS THE ONLY FILE THAT AI AGENTS SHOULD MODIFY.

Contains the full TRM model, optimizer, and training loop for compiler
pass ordering. Agents can change architecture, hyperparameters, optimizer,
loss functions, etc. Everything is fair game.

The experiment runner (run_experiment.py) imports from this file, trains
for a fixed time budget, evaluates on held-out benchmarks, and reports
the val_reward metric.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────────────────────
# Import fixed utilities from prepare.py
# ──────────────────────────────────────────────────────────────

from prepare import (
    OBSERVATION_DIM, NUM_PASSES, FEEDBACK_DIM, SCHEDULE_DIM,
    MAX_STEPS, EVAL_BENCHMARKS, TraceDataset, evaluate,
)


# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE — agents can modify everything below
# ═══════════════════════════════════════════════════════════════

class TRMPassOrdering(nn.Module):
    """Tiny Recursive Model for compiler pass ordering.

    Dual-network architecture per TRM paper:
      net_z(x, y, z) → z   (inner recursion n times)
      net_y(x, z)    → y   (answer refinement)
    Multi-head output: pass logits, feasibility, value, halt.
    """

    # ── Hyperparameters (agents can change these) ──
    LATENT_DIM = 128
    HIDDEN_DIM = 128
    N_RECURSIONS = 6
    ACTIVATION = nn.SiLU

    def __init__(
        self,
        obs_dim: int = OBSERVATION_DIM,
        sched_dim: int = SCHEDULE_DIM,
        fb_dim: int = FEEDBACK_DIM,
        num_passes: int = NUM_PASSES,
    ):
        super().__init__()
        self.latent_dim = self.LATENT_DIM
        self.n_recursions = self.N_RECURSIONS

        x_dim = obs_dim + sched_dim + fb_dim

        # net_z: updates reasoning state z from (x, y, z)
        self.net_z = nn.Sequential(
            nn.Linear(x_dim + 2 * self.latent_dim, self.HIDDEN_DIM),
            self.ACTIVATION(),
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            self.ACTIVATION(),
            nn.Linear(self.HIDDEN_DIM, self.latent_dim),
        )

        # net_y: refines answer y from (x, z)
        self.net_y = nn.Sequential(
            nn.Linear(x_dim + self.latent_dim, self.HIDDEN_DIM),
            self.ACTIVATION(),
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            self.ACTIVATION(),
            nn.Linear(self.HIDDEN_DIM, self.latent_dim),
        )

        # Output heads
        self.head_pass = nn.Linear(self.latent_dim, num_passes)
        self.head_feas = nn.Linear(self.latent_dim, 1)
        self.head_val = nn.Linear(self.latent_dim, 1)
        self.head_halt = nn.Linear(self.latent_dim, 1)

    def init_latent(self, batch_size: int = 1, device: str = "cpu"):
        """Initialize y and z latents to zeros."""
        y = torch.zeros(batch_size, self.latent_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return y, z

    def forward(self, obs, sched, fb, y, z):
        """Full forward: inner recursion + output heads."""
        x = torch.cat([obs, sched, fb], dim=-1)

        # Inner recursion loop
        for _ in range(self.n_recursions):
            z = self.net_z(torch.cat([x, y, z], dim=-1))

        # Refine answer
        y = self.net_y(torch.cat([x, z], dim=-1))

        return {
            "y": y,
            "z": z,
            "pass_logits": self.head_pass(z),
            "feasibility": torch.sigmoid(self.head_feas(z).squeeze(-1)),
            "value": self.head_val(z).squeeze(-1),
            "halt_logit": self.head_halt(z).squeeze(-1),
        }


# ═══════════════════════════════════════════════════════════════
# LOSS FUNCTION — agents can modify
# ═══════════════════════════════════════════════════════════════

def compute_loss(model, batch, device, entropy_coef=0.05):
    """Compute multi-head training loss for one batch.

    Losses:
    - Pass selection: cross-entropy (reward-weighted)
    - Feasibility: BCE on whether pass compiled
    - Value: MSE on predicted reward
    - Halt: BCE on halt decision
    - Entropy: regularization to prevent collapse
    """
    obs = batch["observation"].to(device)
    sched = batch["schedule"].to(device)
    fb = batch["feedback"].to(device)
    target = batch["pass_id"].to(device)
    reward = batch["reward"].to(device)

    bs = obs.shape[0]
    y, z = model.init_latent(bs, device)
    out = model(obs, sched, fb, y, z)

    # Reward-weighted pass selection loss
    raw_pass_loss = F.cross_entropy(out["pass_logits"], target, reduction="none")
    weights = (reward - reward.min()) / (reward.max() - reward.min() + 1e-8)
    weights = 0.5 + 0.5 * weights
    pass_loss = (raw_pass_loss * weights).mean()

    # Entropy regularization
    probs = F.softmax(out["pass_logits"], dim=-1)
    log_probs = F.log_softmax(out["pass_logits"], dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    entropy_loss = -entropy_coef * entropy

    # Feasibility loss
    feas_target = (reward > -2.0).float()
    feas_loss = F.binary_cross_entropy(out["feasibility"], feas_target)

    # Value loss
    val_loss = F.mse_loss(out["value"], reward)

    # Halt loss
    halt_target = (reward <= 0.0).float()
    halt_loss = F.binary_cross_entropy_with_logits(out["halt_logit"], halt_target)

    # Weighted combination
    total = pass_loss + entropy_loss + 0.5 * feas_loss + 0.3 * val_loss + 0.2 * halt_loss

    return total, {
        "pass_loss": pass_loss.item(),
        "entropy": entropy.item(),
        "feas_loss": feas_loss.item(),
        "val_loss": val_loss.item(),
        "halt_loss": halt_loss.item(),
        "loss": total.item(),
    }


# ═══════════════════════════════════════════════════════════════
# OPTIMIZER & SCHEDULER — agents can modify
# ═══════════════════════════════════════════════════════════════

def create_optimizer(model):
    """Create optimizer and scheduler. Agents can change this."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    return optimizer


def create_scheduler(optimizer, total_steps):
    """Create learning rate scheduler. Agents can change this."""
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )
    return scheduler


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP — agents can modify
# ═══════════════════════════════════════════════════════════════

def train(
    model,
    dataloader,
    device,
    time_budget=300,
    epochs=None,
):
    """Train model for a fixed time budget.

    Args:
        model: TRMPassOrdering model
        dataloader: PyTorch DataLoader of training traces
        device: "cuda" or "cpu"
        time_budget: Maximum training time in seconds (default 5 min)
        epochs: If set, train for exactly this many epochs (ignore time_budget)

    Returns:
        dict with training history and final metrics
    """
    model = model.to(device)
    model.train()

    optimizer = create_optimizer(model)
    total_steps = epochs * len(dataloader) if epochs else len(dataloader) * 100
    scheduler = create_scheduler(optimizer, total_steps)

    history = {
        "loss": [],
        "pass_loss": [],
        "entropy": [],
        "feas_loss": [],
        "val_loss": [],
        "halt_loss": [],
        "lr": [],
        "step": [],
    }

    start_time = time.time()
    step_count = 0
    epoch_count = 0

    while True:
        epoch_count += 1
        epoch_losses = {k: 0.0 for k in history if k != "step" and k != "lr"}
        n_batches = 0

        for batch in dataloader:
            # Check time budget
            if epochs is None and (time.time() - start_time) >= time_budget:
                break

            optimizer.zero_grad()
            loss, info = compute_loss(model, batch, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            for k in epoch_losses:
                epoch_losses[k] += info[k]
            n_batches += 1
            step_count += 1

            history["step"].append(step_count)
            history["lr"].append(scheduler.get_last_lr()[0])

        if n_batches == 0:
            break

        avg = {k: v / n_batches for k, v in epoch_losses.items()}
        for k in avg:
            history[k].append(avg[k])

        elapsed = time.time() - start_time
        print(
            f"  Epoch {epoch_count:3d} | "
            f"Loss: {avg['loss']:.4f} | "
            f"Pass: {avg['pass_loss']:.4f} | "
            f"Entropy: {avg['entropy']:.4f} | "
            f"Value: {avg['val_loss']:.4f} | "
            f"Time: {elapsed:.0f}s"
        )

        # Stop if time budget exceeded
        if epochs is None and elapsed >= time_budget:
            break
        if epochs is not None and epoch_count >= epochs:
            break

    return {
        "history": history,
        "epochs": epoch_count,
        "steps": step_count,
        "total_time": time.time() - start_time,
        "final_loss": history["loss"][-1] if history["loss"] else None,
    }
