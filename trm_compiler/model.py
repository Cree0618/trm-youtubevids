"""TRM model adapted for compiler pass ordering."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env_wrapper import NUM_PASSES, pass_id_to_name
from .types import CompilerFeedback


class TinyPassOrderingRefiner(nn.Module):
    """TRM-style tiny network for compiler pass ordering.

    Architecture mirrors the TRM paper:
    - Single 2-layer MLP for reasoning (net)
    - Multiple heads: pass selection, feasibility, value, halt
    - Inner recursion loop for deep reasoning
    - Separate answer (y) and reasoning (z) latents per paper
    """

    def __init__(
        self,
        observation_dim: int = 56,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_passes: int = NUM_PASSES,
        n_recursions: int = 6,
        feedback_dim: int = 4,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.num_passes = num_passes
        self.n_recursions = n_recursions

        # Input dimensions:
        # observation (Autophase) + schedule encoding + feedback + answer (y)
        schedule_dim = 4  # step_norm, num_passes_norm, last_pass_onehot_approx
        # feedback_dim: 4 (simple) or 20 (rich) — configurable
        answer_dim = latent_dim  # y is same size as z

        # The paper uses a single net that processes (x, y, z) -> z
        # For pass ordering: input = concat(obs, schedule, feedback, y)
        input_dim = observation_dim + schedule_dim + feedback_dim + answer_dim

        # Core reasoning networks — 2 layers per TRM paper
        # x = cat(obs, schedule, feedback) = 56+4+20 = 80 dims
        # net_z: updates reasoning z from (x, y, z) — 80+64+64=208 -> hidden -> 64
        z_input_dim = observation_dim + schedule_dim + feedback_dim + answer_dim + latent_dim
        self.net_z = nn.Sequential(
            nn.Linear(z_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # net_y: refines answer y from (x, z) — 80+64=144 -> hidden -> 64
        y_input_dim = observation_dim + schedule_dim + feedback_dim + latent_dim
        self.net_y = nn.Sequential(
            nn.Linear(y_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Output heads (shared across recursion)
        self.pass_head = nn.Linear(latent_dim, num_passes)
        self.feasibility_head = nn.Linear(latent_dim, 1)
        self.value_head = nn.Linear(latent_dim, 1)
        self.halt_head = nn.Linear(latent_dim, 1)

    def reason(self, observation: torch.Tensor, schedule: torch.Tensor,
               feedback: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
               n: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Inner recursion loop: z = net_z(x, y, z) repeated n times, then refine y.

        Matches TRM paper's latent_recursion():
          for i in range(n): z = net_z(cat(x, y, z))
          y = net_y(cat(x, z))
        """
        if n is None:
            n = self.n_recursions

        x = torch.cat([observation, schedule, feedback], dim=-1)

        # Inner recursion: update z (reasoning state)
        for _ in range(n):
            net_input = torch.cat([x, y, z], dim=-1)
            z = self.net_z(net_input)

        # Refine answer y from reasoning z
        refine_input = torch.cat([x, z], dim=-1)
        y = self.net_y(refine_input)

        return y, z

    def forward(self, observation: torch.Tensor, schedule: torch.Tensor,
                feedback: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                n: Optional[int] = None) -> dict[str, torch.Tensor]:
        """Full forward: inner recursion + output heads."""
        y_new, z_new = self.reason(observation, schedule, feedback, y, z, n)

        pass_logits = self.pass_head(z_new)
        feasibility = torch.sigmoid(self.feasibility_head(z_new).squeeze(-1))
        value = self.value_head(z_new).squeeze(-1)
        halt_logit = self.halt_head(z_new).squeeze(-1)

        return {
            "y": y_new,
            "z": z_new,
            "pass_logits": pass_logits,
            "feasibility": feasibility,
            "value": value,
            "halt_logit": halt_logit,
        }

    def init_latent(self, batch_size: int = 1, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize y and z latents."""
        y = torch.zeros(batch_size, self.latent_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return y, z


def encode_schedule(step: int, max_steps: int, applied_passes: list[int]) -> list[float]:
    """Encode current schedule state for the model."""
    step_norm = step / max(max_steps, 1)
    num_passes_norm = len(applied_passes) / max(max_steps, 1)

    # Approximate last pass as a normalized value
    if applied_passes:
        last_pass_norm = applied_passes[-1] / NUM_PASSES
        last_pass_approx = sum(applied_passes[-min(3, len(applied_passes)):]) / (3 * NUM_PASSES)
    else:
        last_pass_norm = 0.0
        last_pass_approx = 0.0

    return [step_norm, num_passes_norm, last_pass_norm, last_pass_approx]


@torch.no_grad()
def rollout_pass_optimizer(
    model: TinyPassOrderingRefiner,
    observation: np.ndarray,
    max_steps: int = 30,
    temperature: float = 1.0,
    device: str = "cpu",
) -> list[dict]:
    """Run TRM to select passes sequentially.

    Args:
        model: The TRM pass ordering model
        observation: Initial Autophase observation (56-dim numpy)
        max_steps: Maximum number of passes to apply
        temperature: Sampling temperature (lower = more greedy)
        device: Torch device

    Returns:
        List of step dicts with pass_id, halt_prob, value, etc.
    """
    batch_size = 1
    y, z = model.init_latent(batch_size, device)

    obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    # Use 4-dim simple feedback (works best with synthetic data)
    feedback_tensor = torch.zeros(batch_size, 4, device=device)

    trace = []
    applied_passes = []

    for step in range(max_steps):
        schedule_vec = encode_schedule(step, max_steps, applied_passes)
        schedule_tensor = torch.tensor(
            schedule_vec, dtype=torch.float32, device=device
        ).unsqueeze(0)

        outputs = model(obs_tensor, schedule_tensor, feedback_tensor, y, z)

        y = outputs["y"]
        z = outputs["z"]

        # Check halt signal — only allow halt after minimum exploration
        halt_prob = torch.sigmoid(outputs["halt_logit"]).item()
        min_steps = 5  # minimum passes before halt can trigger

        if halt_prob > 0.5 and step >= min_steps:
            trace.append({
                "step": step,
                "pass_id": -1,  # halt
                "halt_prob": halt_prob,
                "value": outputs["value"].item(),
                "pass_logits": outputs["pass_logits"].squeeze(0).cpu().tolist(),
            })
            break

        # Select pass with legal mask (don't repeat immediately)
        pass_logits = outputs["pass_logits"].squeeze(0) / temperature

        # Mask out recently applied passes (soft penalty)
        if len(applied_passes) > 0:
            for prev_pass in applied_passes[-3:]:
                pass_logits[prev_pass] -= 2.0

        pass_probs = F.softmax(pass_logits, dim=-1)

        # Clamp to avoid multinomial errors with extreme logits
        pass_probs = torch.clamp(pass_probs, min=1e-6)
        pass_probs = pass_probs / pass_probs.sum(dim=-1, keepdim=True)

        pass_id = torch.multinomial(pass_probs, 1).item()

        # Update feedback for next step
        applied_passes.append(pass_id)
        schedule_vec = encode_schedule(step + 1, max_steps, applied_passes)

        trace.append({
            "step": step,
            "pass_id": pass_id,
            "halt_prob": halt_prob,
            "value": outputs["value"].item(),
            "pass_logits": outputs["pass_logits"].squeeze(0).cpu().tolist(),
        })

    return trace


# Alias for clarity in benchmarks
rollout_blind = rollout_pass_optimizer


@torch.no_grad()
def rollout_closed_loop(
    model: TinyPassOrderingRefiner,
    env,
    max_steps: int = 30,
    temperature: float = 1.0,
    device: str = "cpu",
) -> list[dict]:
    """Closed-loop TRM: run passes in real environment, feed back results.

    Key difference from blind rollout:
    - After each pass, the environment runs it and returns real results
    - The new observation and feedback are injected into the model
    - This gives TRM real compiler state at each reasoning step

    Args:
        model: The TRM pass ordering model
        env: Compiler environment (SyntheticCompilerEnv or CompilerGymWrapper)
        max_steps: Maximum number of passes to apply
        temperature: Sampling temperature
        device: Torch device

    Returns:
        List of step dicts with pass_id, halt_prob, value, real_reward, etc.
    """
    batch_size = 1
    y, z = model.init_latent(batch_size, device)

    # Get initial state
    obs, initial_inst = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    # Start with zero feedback (no pass applied yet)
    feedback_vec = CompilerFeedback.zero().simple_encode()
    feedback_tensor = torch.tensor(
        feedback_vec, dtype=torch.float32, device=device
    ).unsqueeze(0)

    trace = []
    applied_passes = []
    total_real_reward = 0.0

    for step in range(max_steps):
        schedule_vec = encode_schedule(step, max_steps, applied_passes)
        schedule_tensor = torch.tensor(
            schedule_vec, dtype=torch.float32, device=device
        ).unsqueeze(0)

        outputs = model(obs_tensor, schedule_tensor, feedback_tensor, y, z)

        y = outputs["y"]
        z = outputs["z"]

        # Check halt signal
        halt_prob = torch.sigmoid(outputs["halt_logit"]).item()
        min_steps = 5

        if halt_prob > 0.5 and step >= min_steps:
            trace.append({
                "step": step,
                "pass_id": -1,
                "halt_prob": halt_prob,
                "value": outputs["value"].item(),
                "real_reward": 0.0,
                "total_real_reward": total_real_reward,
                "inst_count": env.current_inst_count,
                "pass_logits": outputs["pass_logits"].squeeze(0).cpu().tolist(),
            })
            break

        # Select pass
        pass_logits = outputs["pass_logits"].squeeze(0) / temperature
        if len(applied_passes) > 0:
            for prev_pass in applied_passes[-3:]:
                pass_logits[prev_pass] -= 2.0

        pass_probs = F.softmax(pass_logits, dim=-1)
        pass_probs = torch.clamp(pass_probs, min=1e-6)
        pass_probs = pass_probs / pass_probs.sum(dim=-1, keepdim=True)
        pass_id = torch.multinomial(pass_probs, 1).item()

        # RUN THE ACTUAL PASS IN THE ENVIRONMENT
        next_obs, compiler_feedback, done, info = env.step(pass_id)

        # Get real feedback values
        real_reward = compiler_feedback.reward
        total_real_reward += real_reward

        trace.append({
            "step": step,
            "pass_id": pass_id,
            "halt_prob": halt_prob,
            "value": outputs["value"].item(),
            "real_reward": real_reward,
            "total_real_reward": total_real_reward,
            "inst_count": env.current_inst_count,
            "pass_logits": outputs["pass_logits"].squeeze(0).cpu().tolist(),
            "pass_name": info.get("pass_name", ""),
        })

        applied_passes.append(pass_id)

        # INJECT REAL COMPILER STATE INTO NEXT STEP
        obs_tensor = torch.tensor(
            next_obs, dtype=torch.float32, device=device
        ).unsqueeze(0)
        feedback_tensor = torch.tensor(
            compiler_feedback.simple_encode(), dtype=torch.float32, device=device
        ).unsqueeze(0)

        if done:
            break

    return trace
