"""Training data generation for compiler pass ordering."""
from __future__ import annotations
import json
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .types import BenchmarkSpec, CompilerFeedback, CompilerTraceRecord, PassEdit
from .env_wrapper import (
    NUM_PASSES, SyntheticCompilerEnv, CompilerGymWrapper,
    make_compiler_env, pass_id_to_name, _has_compilergym,
)


def _default_cbench_benchmarks() -> list[str]:
    """Return default list of cbench benchmarks."""
    return [
        "adpcm",
        "blowfish",
        "bzip2",
        "crc32",
        "dijkstra",
        "gsm",
        "ispell",
        "jpeg-c",
        "lame",
        "patricia",
        "qsort",
        "rijndael",
        "sha",
        "stringsearch",
        "susan",
        "tiff2bw",
        "tiff2rgba",
        "tiffdither",
        "tiffmedian",
    ]


def generate_compiler_traces(
    benchmarks: Optional[list[str]] = None,
    episodes_per_benchmark: int = 10,
    max_steps_per_episode: int = 30,
    use_heuristic: bool = True,
    output_path: Optional[str] = None,
    seed: int = 42,
    strategy: str = "mixed",
) -> list[CompilerTraceRecord]:
    """Generate training traces by running search on benchmarks.

    Args:
        benchmarks: List of benchmark IDs (default: cbench subset)
        episodes_per_benchmark: How many rollouts per benchmark
        max_steps_per_episode: Max passes per rollout
        use_heuristic: Use synthetic env instead of real CompilerGym
        output_path: If set, save traces to this JSON file
        seed: Random seed
        strategy: "random" (random passes), "greedy" (best at each step),
                  "mixed" (50% random, 50% greedy with noise)

    Returns:
        List of CompilerTraceRecord
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    if benchmarks is None:
        benchmarks = _default_cbench_benchmarks()

    all_traces: list[CompilerTraceRecord] = []

    for bench_id in benchmarks:
        for episode in range(episodes_per_benchmark):
            if strategy == "mixed":
                # Alternate between random and greedy+noise
                use_greedy = (episode % 2 == 1)
            elif strategy == "greedy":
                use_greedy = True
            else:
                use_greedy = False

            if use_heuristic:
                trace = _generate_mixed_trace(
                    bench_id, max_steps_per_episode, rng, np_rng,
                    use_greedy=use_greedy,
                )
            else:
                trace = _generate_compilergym_trace(
                    bench_id, max_steps_per_episode, rng, np_rng
                )
            all_traces.extend(trace)

    if output_path is not None:
        save_traces(all_traces, output_path)

    return all_traces


def _generate_mixed_trace(
    bench_id: str,
    max_steps: int,
    rng: random.Random,
    np_rng: np.random.RandomState,
    use_greedy: bool = False,
) -> list[CompilerTraceRecord]:
    """Generate one episode trace with random or greedy+noise pass selection."""
    env = SyntheticCompilerEnv(benchmark_id=bench_id, seed=np_rng.randint(0, 2**31))
    benchmark = BenchmarkSpec(benchmark_id=bench_id)
    obs, initial_inst = env.reset()

    traces = []
    total_reward = 0.0
    applied_passes = []

    for step in range(max_steps):
        if use_greedy:
            # Greedy with noise: try a few random passes, pick the best,
            # then add noise (sometimes pick random instead)
            best_pass_id = rng.randint(0, NUM_PASSES - 1)
            best_reward = float("-inf")
            num_candidates = min(10, NUM_PASSES)

            for _ in range(num_candidates):
                candidate = rng.randint(0, NUM_PASSES - 1)
                test_env = SyntheticCompilerEnv(benchmark_id=bench_id, seed=np_rng.randint(0, 2**31))
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

            # 20% chance to pick random instead (exploration)
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


def _generate_heuristic_trace(
    bench_id: str,
    max_steps: int,
    rng: random.Random,
    np_rng: np.random.RandomState,
) -> list[CompilerTraceRecord]:
    """Generate one episode trace using the synthetic environment."""
    env = SyntheticCompilerEnv(benchmark_id=bench_id, seed=np_rng.randint(0, 2**31))
    benchmark = BenchmarkSpec(benchmark_id=bench_id)
    obs, initial_inst = env.reset()

    traces = []
    total_reward = 0.0

    for step in range(max_steps):
        pass_id = rng.randint(0, NUM_PASSES - 1)

        prev_inst = env.current_inst_count
        next_obs, feedback, done, info = env.step(pass_id)

        traces.append(CompilerTraceRecord(
            benchmark=benchmark,
            observation=obs.tolist(),
            prev_pass_sequence=list(range(pass_id)) if step == 0 else [pass_id - 1],
            edit=PassEdit(pass_id=pass_id, pass_name=pass_id_to_name(pass_id)),
            feedback=feedback,
            next_observation=next_obs.tolist(),
            reward=feedback.reward,
            step=step,
            score=total_reward + feedback.reward,
        ))

        total_reward += feedback.reward
        obs = next_obs

        if done:
            break

    return traces


def _generate_compilergym_trace(
    bench_id: str,
    max_steps: int,
    rng: random.Random,
    np_rng: np.random.RandomState,
) -> list[CompilerTraceRecord]:
    """Generate one episode trace using real CompilerGym."""
    full_id = f"cbench-v1/{bench_id}"
    wrapper = CompilerGymWrapper(benchmark_id=full_id)

    traces = []
    try:
        wrapper.open()
        obs, initial_inst = wrapper.reset()
        applied_passes = []
        total_reward = 0.0

        for step in range(max_steps):
            pass_id = rng.randint(0, NUM_PASSES - 1)

            next_obs, feedback, done, info = wrapper.step(pass_id)

            traces.append(CompilerTraceRecord(
                benchmark=BenchmarkSpec(benchmark_id=full_id, dataset="cbench-v1"),
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

    finally:
        wrapper.close()

    return traces


def save_traces(traces: list[CompilerTraceRecord], path: str):
    """Save traces to JSON file."""
    data = [t.to_dict() for t in traces]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def load_traces(path: str) -> list[CompilerTraceRecord]:
    """Load traces from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [CompilerTraceRecord.from_dict(d) for d in data]


class CompilerTraceDataset(Dataset):
    """PyTorch dataset wrapping compiler training traces."""

    def __init__(
        self,
        traces: list[CompilerTraceRecord],
        observation_dim: int = 56,
        context_length: int = 4,
    ):
        self.traces = traces
        self.observation_dim = observation_dim
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.traces[idx]

        # Observation
        observation = torch.tensor(record.observation[:self.observation_dim], dtype=torch.float32)

        # Pad if needed
        if len(record.observation) < self.observation_dim:
            pad = torch.zeros(self.observation_dim - len(record.observation))
            observation = torch.cat([observation, pad])

        # Schedule encoding
        from .model import encode_schedule
        schedule_vec = encode_schedule(
            record.step, self.context_length * 10, record.prev_pass_sequence
        )
        schedule = torch.tensor(schedule_vec, dtype=torch.float32)

        # Feedback encoding
        feedback = torch.tensor(record.feedback.encode(), dtype=torch.float32)

        # Target pass ID
        pass_id = torch.tensor(record.edit.pass_id, dtype=torch.long)

        # Reward (used for loss weighting)
        reward = torch.tensor(record.reward, dtype=torch.float32)

        return {
            "observation": observation,
            "schedule": schedule,
            "feedback": feedback,
            "pass_id": pass_id,
            "reward": reward,
            "step": record.step,
        }
