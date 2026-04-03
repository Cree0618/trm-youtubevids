"""TRM Autoresearch — Prepare data and provide runtime utilities.

This file is FIXED and should NOT be modified by agents.
It handles one-time data preparation and provides shared utilities
for training and evaluation.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

EVAL_BENCHMARKS = ["qsort", "adpcm", "blowfish"]
ALL_BENCHMARKS = [
    "adpcm", "blowfish", "bzip2", "crc32", "dijkstra", "gsm",
    "ispell", "jpeg-c", "lame", "patricia", "qsort", "rijndael",
    "sha", "stringsearch", "susan", "tiff2bw", "tiff2rgba",
    "tiffdither", "tiffmedian",
]
OBSERVATION_DIM = 56
NUM_PASSES = 37
FEEDBACK_DIM = 4  # simple encoding
SCHEDULE_DIM = 4
MAX_STEPS = 30
SEED = 42
TRAIN_EPISODES_PER_BENCH = 20
EVAL_EPISODES_PER_BENCH = 5

# ──────────────────────────────────────────────────────────────
# Import project modules
# ──────────────────────────────────────────────────────────────

from trm_compiler.env_wrapper import SyntheticCompilerEnv, NUM_PASSES as _NP
from trm_compiler.env_wrapper import pass_id_to_name
from trm_compiler.types import BenchmarkSpec, CompilerFeedback, CompilerTraceRecord, PassEdit


# ──────────────────────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────────────────────

def generate_traces(
    benchmarks: Optional[list[str]] = None,
    episodes_per_benchmark: int = TRAIN_EPISODES_PER_BENCH,
    max_steps: int = MAX_STEPS,
    output_path: Optional[str] = None,
    seed: int = SEED,
) -> list[CompilerTraceRecord]:
    """Generate training traces using mixed random/greedy strategy."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    if benchmarks is None:
        benchmarks = ALL_BENCHMARKS

    all_traces: list[CompilerTraceRecord] = []

    for bench_id in benchmarks:
        for episode in range(episodes_per_benchmark):
            use_greedy = (episode % 2 == 1)
            traces = _generate_episode(
                bench_id, max_steps, rng, np_rng, use_greedy=use_greedy,
            )
            all_traces.extend(traces)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = [t.to_dict() for t in all_traces]
        with open(output_path, "w") as f:
            json.dump(data, f)

    return all_traces


def _generate_episode(
    bench_id: str,
    max_steps: int,
    rng: random.Random,
    np_rng: np.random.RandomState,
    use_greedy: bool = False,
) -> list[CompilerTraceRecord]:
    """Generate one episode trace."""
    env = SyntheticCompilerEnv(benchmark_id=bench_id, seed=np_rng.randint(0, 2**31))
    benchmark = BenchmarkSpec(benchmark_id=bench_id)
    obs, initial_inst = env.reset()

    traces = []
    total_reward = 0.0
    applied_passes = []

    for step in range(max_steps):
        if use_greedy:
            best_pass_id = rng.randint(0, NUM_PASSES - 1)
            best_reward = float("-inf")
            num_candidates = min(10, NUM_PASSES)

            for _ in range(num_candidates):
                candidate = rng.randint(0, NUM_PASSES - 1)
                test_env = SyntheticCompilerEnv(
                    benchmark_id=bench_id, seed=np_rng.randint(0, 2**31)
                )
                test_env.reset()
                done_inner = False
                for p in applied_passes:
                    _, _, done_inner, _ = test_env.step(p)
                    if done_inner:
                        break
                if done_inner:
                    continue
                _, fb, _, _ = test_env.step(candidate)
                if fb.reward > best_reward:
                    best_reward = fb.reward
                    best_pass_id = candidate

            if rng.random() < 0.2:
                pass_id = rng.randint(0, NUM_PASSES - 1)
            else:
                pass_id = best_pass_id
        else:
            pass_id = rng.randint(0, NUM_PASSES - 1)

        prev_inst = env.current_inst_count
        next_obs, feedback, done, info = env.step(pass_id)

        traces.append(CompilerTraceRecord(
            benchmark=benchmark,
            observation=obs.tolist(),
            prev_pass_sequence=list(applied_passes),
            edit=PassEdit(pass_id=pass_id, pass_name=pass_id_to_name(pass_id)),
            feedback=feedback,
            next_observation=next_obs.tolist(),
            reward=feedback.reward,
            step=step,
            score=total_reward + feedback.reward,
        ))

        total_reward += feedback.reward
        applied_passes.append(pass_id)
        obs = next_obs

        if done:
            break

    return traces


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class TraceDataset(Dataset):
    """PyTorch dataset for compiler traces."""

    def __init__(self, traces: list[CompilerTraceRecord]):
        self.traces = traces

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.traces[idx]

        observation = torch.tensor(
            record.observation[:OBSERVATION_DIM], dtype=torch.float32
        )
        if len(record.observation) < OBSERVATION_DIM:
            pad = torch.zeros(OBSERVATION_DIM - len(record.observation))
            observation = torch.cat([observation, pad])

        schedule_vec = _encode_schedule(
            record.step, MAX_STEPS * 10, record.prev_pass_sequence
        )
        schedule = torch.tensor(schedule_vec, dtype=torch.float32)

        feedback = torch.tensor(
            record.feedback.simple_encode(), dtype=torch.float32
        )

        pass_id = torch.tensor(record.edit.pass_id, dtype=torch.long)
        reward = torch.tensor(record.reward, dtype=torch.float32)

        return {
            "observation": observation,
            "schedule": schedule,
            "feedback": feedback,
            "pass_id": pass_id,
            "reward": reward,
        }


def _encode_schedule(step: int, max_steps: int, applied_passes: list[int]) -> list[float]:
    """Encode current schedule state."""
    step_norm = step / max(max_steps, 1)
    num_passes_norm = len(applied_passes) / max(max_steps, 1)

    if applied_passes:
        last_pass_norm = applied_passes[-1] / NUM_PASSES
        last_pass_approx = sum(applied_passes[-min(3, len(applied_passes)):]) / (3 * NUM_PASSES)
    else:
        last_pass_norm = 0.0
        last_pass_approx = 0.0

    return [step_norm, num_passes_norm, last_pass_norm, last_pass_approx]


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    benchmarks: Optional[list[str]] = None,
    max_steps: int = MAX_STEPS,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate model on held-out benchmarks. Returns mean cumulative reward."""
    if benchmarks is None:
        benchmarks = EVAL_BENCHMARKS

    model.eval()
    all_rewards = []

    for bench_id in benchmarks:
        env = SyntheticCompilerEnv(benchmark_id=bench_id, seed=SEED)
        obs, _ = env.reset()

        y, z = model.init_latent(1, device)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        feedback_tensor = torch.zeros(1, FEEDBACK_DIM, device=device)

        total_reward = 0.0
        applied_passes = []

        for step in range(max_steps):
            schedule_vec = _encode_schedule(step, max_steps, applied_passes)
            schedule_tensor = torch.tensor(
                schedule_vec, dtype=torch.float32, device=device
            ).unsqueeze(0)

            all_outputs = model(obs_tensor, schedule_tensor, feedback_tensor, y, z)
            # Handle both single-step (dict) and deep supervision (list) outputs
            if isinstance(all_outputs, list):
                outputs = all_outputs[-1]
            else:
                outputs = all_outputs
            y, z = outputs["y"], outputs["z"]

            halt_prob = torch.sigmoid(outputs["halt_logit"]).item()
            if halt_prob > 0.5 and step >= 5:
                break

            pass_logits = outputs["pass_logits"].squeeze(0)
            if applied_passes:
                for prev_pass in applied_passes[-3:]:
                    pass_logits[prev_pass] -= 2.0

            pass_probs = torch.softmax(pass_logits, dim=-1)
            pass_probs = torch.clamp(pass_probs, min=1e-6)
            pass_probs = pass_probs / pass_probs.sum(dim=-1, keepdim=True)
            pass_id = torch.multinomial(pass_probs, 1).item()

            next_obs, fb, done, _ = env.step(pass_id)
            total_reward += fb.reward
            applied_passes.append(pass_id)

            obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            feedback_tensor = torch.tensor(
                fb.simple_encode(), dtype=torch.float32, device=device
            ).unsqueeze(0)

            if done:
                break

        all_rewards.append(total_reward)

    return {
        "val_reward": float(np.mean(all_rewards)),
        "val_reward_std": float(np.std(all_rewards)),
        "val_reward_min": float(np.min(all_rewards)),
        "val_reward_max": float(np.max(all_rewards)),
    }


# ──────────────────────────────────────────────────────────────
# Main: one-time data preparation
# ──────────────────────────────────────────────────────────────

def main():
    """Generate training data. Run once before experiments."""
    traces_path = Path("data/traces.json")

    if traces_path.exists():
        print(f"Traces already exist at {traces_path}")
        with open(traces_path) as f:
            data = json.load(f)
        print(f"  Loaded {len(data)} trace records")
        return

    print("Generating training traces...")
    traces = generate_traces(output_path=str(traces_path))
    print(f"Generated {len(traces)} trace records")
    print(f"Saved to {traces_path}")


if __name__ == "__main__":
    main()
