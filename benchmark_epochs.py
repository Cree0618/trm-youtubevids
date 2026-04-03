"""Fair benchmark: compare architectures with SAME number of epochs.

Usage:
    python benchmark_epochs.py

This script trains multiple configurations for the exact same number of
epochs, then evaluates each on the held-out benchmarks. This eliminates
the time-budget bias where slower architectures get fewer updates.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prepare import (
    OBSERVATION_DIM, NUM_PASSES, FEEDBACK_DIM, SCHEDULE_DIM,
    MAX_STEPS, EVAL_BENCHMARKS, TraceDataset, evaluate,
)
from train import TRMPassOrdering, create_optimizer, create_scheduler
from run_experiment import TRACES_FILE


# ── Configurations to compare ──

CONFIGS = {
    "baseline": {
        "LATENT_DIM": 128,
        "HIDDEN_DIM": 128,
        "N_RECURSIONS": 6,
        "N_SUPERVISION": 1,
        "ACTIVATION": nn.SiLU,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
    "deep_sup_2": {
        "LATENT_DIM": 128,
        "HIDDEN_DIM": 128,
        "N_RECURSIONS": 6,
        "N_SUPERVISION": 2,
        "ACTIVATION": nn.SiLU,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
    "deep_sup_4": {
        "LATENT_DIM": 128,
        "HIDDEN_DIM": 128,
        "N_RECURSIONS": 6,
        "N_SUPERVISION": 4,
        "ACTIVATION": nn.SiLU,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
}

EPOCHS = 100  # Same for all configs
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DeepSupModel(TRMPassOrdering):
    """TRM model with configurable deep supervision."""

    def __init__(self, n_supervision=1, **kwargs):
        super().__init__(**kwargs)
        self.n_supervision = n_supervision

    def forward(self, obs, sched, fb, y, z):
        """Support both single-step and deep supervision."""
        x = torch.cat([obs, sched, fb], dim=-1)

        if self.n_supervision == 1:
            # Single step (original behavior)
            for _ in range(self.n_recursions):
                z = self.net_z(torch.cat([x, y, z], dim=-1))
            y = self.net_y(torch.cat([x, z], dim=-1))
            return {
                "y": y, "z": z,
                "pass_logits": self.head_pass(z),
                "feasibility": torch.sigmoid(self.head_feas(z).squeeze(-1)),
                "value": self.head_val(z).squeeze(-1),
                "halt_logit": self.head_halt(z).squeeze(-1),
            }
        else:
            # Deep supervision: N_SUPERVISION steps with detach between them
            all_outputs = []
            for s in range(self.n_supervision):
                for _ in range(self.n_recursions):
                    z = self.net_z(torch.cat([x, y, z], dim=-1))
                y = self.net_y(torch.cat([x, z], dim=-1))
                out = {
                    "y": y, "z": z,
                    "pass_logits": self.head_pass(z),
                    "feasibility": torch.sigmoid(self.head_feas(z).squeeze(-1)),
                    "value": self.head_val(z).squeeze(-1),
                    "halt_logit": self.head_halt(z).squeeze(-1),
                }
                all_outputs.append(out)
                z = z.detach()
                y = y.detach()
            return all_outputs


def load_data():
    from prepare import CompilerTraceRecord
    with open(TRACES_FILE) as f:
        data = json.load(f)
    traces = [CompilerTraceRecord.from_dict(d) for d in data]
    print(f"  Loaded {len(traces)} trace records")
    dataset = TraceDataset(traces)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return dataloader


def compute_step_loss(out, target, reward, entropy_coef=0.05):
    """Compute loss for a single supervision step."""
    pass_loss = F.cross_entropy(out["pass_logits"], target)
    probs = F.softmax(out["pass_logits"], dim=-1)
    log_probs = F.log_softmax(out["pass_logits"], dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    entropy_loss = -entropy_coef * entropy
    feas_target = (reward > -2.0).float()
    feas_loss = F.binary_cross_entropy(out["feasibility"], feas_target)
    val_loss = F.mse_loss(out["value"], reward)
    halt_target = (reward <= 0.0).float()
    halt_loss = F.binary_cross_entropy_with_logits(out["halt_logit"], halt_target)
    total = pass_loss + entropy_loss + 0.5 * feas_loss + 0.3 * val_loss + 0.2 * halt_loss
    return total


def train_epoch(model, dataloader, optimizer, scheduler, device, n_supervision):
    """Train one epoch. Returns average losses."""
    model.train()
    epoch_losses = {"loss": 0.0, "pass_loss": 0.0, "entropy": 0.0, "val_loss": 0.0}
    n_batches = 0

    for batch in dataloader:
        obs = batch["observation"].to(device)
        sched = batch["schedule"].to(device)
        fb = batch["feedback"].to(device)
        target = batch["pass_id"].to(device)
        reward = batch["reward"].to(device)

        bs = obs.shape[0]
        y, z = model.init_latent(bs, device)
        result = model(obs, sched, fb, y, z)

        optimizer.zero_grad()

        if n_supervision == 1:
            # Single step: one backward
            loss = compute_step_loss(result, target, reward)
            loss.backward()
        else:
            # Deep supervision: sum losses across steps, single backward
            total_loss = torch.tensor(0.0, device=device)
            for out in result:
                total_loss = total_loss + compute_step_loss(out, target, reward)
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Log last step metrics
        last_out = result[-1] if n_supervision > 1 else result
        epoch_losses["pass_loss"] += F.cross_entropy(last_out["pass_logits"], target).item()
        probs = F.softmax(last_out["pass_logits"], dim=-1)
        log_probs = F.log_softmax(last_out["pass_logits"], dim=-1)
        epoch_losses["entropy"] += (-(probs * log_probs).sum(dim=-1).mean()).item()
        epoch_losses["val_loss"] += F.mse_loss(last_out["value"], reward).item()
        epoch_losses["loss"] += compute_step_loss(last_out, target, reward).item()
        n_batches += 1

    return {k: v / n_batches for k, v in epoch_losses.items()}


def run_config(name, config, dataloader, epochs):
    """Train and evaluate one configuration."""
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"  Epochs: {epochs}")
    print(f"  N_SUPERVISION: {config['N_SUPERVISION']}")
    print(f"{'='*60}")

    n_sup = config["N_SUPERVISION"]
    model = DeepSupModel(n_supervision=n_sup).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Scale scheduler steps: deep supervision does N_SUP optimizer steps per batch
    total_steps = epochs * len(dataloader) * n_sup
    optimizer = create_optimizer(model)
    optimizer.param_groups[0]["lr"] = config["lr"]
    optimizer.param_groups[0]["weight_decay"] = config["weight_decay"]
    scheduler = create_scheduler(optimizer, total_steps)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        avg = train_epoch(model, dataloader, optimizer, scheduler, DEVICE, n_sup)
        if ep % 20 == 0 or ep == 1:
            elapsed = time.time() - t0
            print(
                f"  Epoch {ep:4d}/{epochs} | "
                f"Loss: {avg['loss']:.4f} | "
                f"Pass: {avg['pass_loss']:.4f} | "
                f"Entropy: {avg['entropy']:.4f} | "
                f"Value: {avg['val_loss']:.4f} | "
                f"Time: {elapsed:.0f}s"
            )

    train_time = time.time() - t0
    print(f"\n  Training complete in {train_time:.0f}s")

    # Evaluate: use single-step forward (last supervision step during eval)
    model.eval()
    # Temporarily switch to single-step for evaluation
    orig_n_sup = model.n_supervision
    model.n_supervision = 1
    eval_result = evaluate(model, device=DEVICE)
    model.n_supervision = orig_n_sup

    print(f"  val_reward: {eval_result['val_reward']:+.4f}")
    print(f"  val_reward_std: {eval_result['val_reward_std']:.4f}")

    return {
        "name": name,
        "config": config,
        "params": n_params,
        "train_time": train_time,
        "val_reward": eval_result["val_reward"],
        "val_reward_std": eval_result["val_reward_std"],
    }


def main():
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")

    print("\nLoading training data...")
    dataloader = load_data()

    results = []
    for name, config in CONFIGS.items():
        result = run_config(name, config, dataloader, EPOCHS)
        results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<15} {'Params':>8} {'Time':>6} {'val_reward':>12} {'std':>8}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['name']:<15} {r['params']:>8,} {r['train_time']:>5.0f}s "
            f"{r['val_reward']:>+11.4f} {r['val_reward_std']:>7.4f}"
        )

    best = max(results, key=lambda r: r["val_reward"])
    print(f"\nBest: {best['name']} (val_reward: {best['val_reward']:+.4f})")

    # Save results
    out_path = Path("experiments/benchmark_epochs.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_results = []
    for r in results:
        sr = dict(r)
        cfg = dict(sr["config"])
        cfg["ACTIVATION"] = cfg["ACTIVATION"].__name__
        sr["config"] = cfg
        serializable_results.append(sr)
    with open(out_path, "w") as f:
        json.dump({
            "epochs": EPOCHS,
            "device": DEVICE,
            "results": serializable_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
