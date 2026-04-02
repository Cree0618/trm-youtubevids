"""Baselines for benchmarking TRM compiler pass ordering.

Provides comparison against:
1. LLVM's built-in optimization levels (-O0 through -Oz)
2. Random search (various budgets)
3. Greedy search
4. Exhaustive search (small action space)
5. Reinforcement learning baselines (DQN, PPO reference points)
"""
from __future__ import annotations
import random
import time
from typing import Optional

import numpy as np

from .env_wrapper import NUM_PASSES, SyntheticCompilerEnv, pass_id_to_name


# ══════════════════════════════════════════════════════════════
# 1. LLVM Built-in Optimization Levels
# ══════════════════════════════════════════════════════════════
#
# These are the "gold standard" for pass ordering — hand-tuned by
# compiler engineers over decades. If TRM can't beat -Oz, it's not useful.
#
# With real CompilerGym, you can measure these directly:
#   env = compiler_gym.make("llvm-v0", benchmark="cbench-v1/qsort")
#   env.reset()
#   oz_inst_count = env.observation["IrInstructionCountOz"]

# In synthetic mode, we model the effectiveness of fixed pipelines:
_OPTIMIZATION_LEVELS = {
    # (pass_sequence, expected_reduction_fraction)
    "O0": (0.00, "No optimization — baseline"),
    "O1": (0.08, "Minimal optimization — simplifycfg, instcombine, early-cse"),
    "O2": (0.18, "Standard optimization — adds gvn, licm, loop passes, inline"),
    "O3": (0.22, "Aggressive optimization — adds loop-unroll, slp-vectorize"),
    "Os": (0.20, "Size optimization — similar to O2 but avoids unrolling"),
    "Oz": (0.19, "Minimal size — most aggressive size reduction"),
}

# LLVM -Oz default pass pipeline (approximate, ~40 passes)
_OZ_PIPELINE = [
    "mem2reg", "simplifycfg", "early-cse", "instcombine", "reassociate",
    "gvn", "sccp", "dce", "adce", "simplifycfg-2", "instcombine-2",
    "inline", "deadargelim", "globaldce", "ipsccp", "prune-eh",
    "loop-rotate", "licm", "indvars", "loop-deletion", "loop-idiom",
    "sink", "sroa", "correlated-propagation", "jump-threading",
    "strip-dead-prototypes", "constmerge",
]

_O3_PIPELINE = _OZ_PIPELINE + [
    "loop-unroll", "loop-vectorize", "slp-vectorize",
    "speculative-execution", "loop-unswitch",
]


def run_optimization_level(
    benchmark_id: str,
    level: str = "Oz",
    seed: int = 42,
) -> dict:
    """Run a fixed LLVM optimization pipeline (simulated).

    Args:
        benchmark_id: Benchmark name
        level: One of O0, O1, O2, O3, Os, Oz
        seed: Random seed

    Returns:
        Dict with total_reward, final_inst, reduction_pct, pipeline
    """
    env = SyntheticCompilerEnv(benchmark_id=benchmark_id, seed=seed)
    obs, initial_inst = env.reset()

    if level == "O0":
        return {
            "level": level,
            "total_reward": 0.0,
            "initial_inst": initial_inst,
            "final_inst": initial_inst,
            "reduction_pct": 0.0,
            "pipeline": [],
            "num_steps": 0,
        }

    pipeline = _OZ_PIPELINE if level in ("Oz", "Os") else _O3_PIPELINE
    if level == "O1":
        pipeline = pipeline[:8]  # truncate to ~8 passes
    elif level == "O2":
        pipeline = pipeline[:20]  # ~20 passes

    total_reward = 0.0
    for pass_name in pipeline:
        if pass_name in {pass_id_to_name(i) for i in range(NUM_PASSES)}:
            from .env_wrapper import pass_name_to_id
            pass_id = pass_name_to_id(pass_name)
            obs, feedback, done, info = env.step(pass_id)
            total_reward += feedback.reward
            if done:
                break

    return {
        "level": level,
        "total_reward": total_reward,
        "initial_inst": initial_inst,
        "final_inst": env.current_inst_count,
        "reduction_pct": 1.0 - env.current_inst_count / max(initial_inst, 1),
        "pipeline": pipeline,
        "num_steps": len(pipeline),
    }


# ══════════════════════════════════════════════════════════════
# 2. Random Search
# ══════════════════════════════════════════════════════════════

def random_search(
    benchmark_id: str,
    max_steps: int = 30,
    num_trials: int = 100,
    seed: int = 42,
) -> dict:
    """Random search: try random pass sequences.

    Returns:
        Dict with best_reward, best_sequence, mean_reward, std_reward
    """
    rng = random.Random(seed)

    best_reward = float("-inf")
    best_sequence = []
    all_rewards = []
    all_sequences = []

    for trial in range(num_trials):
        env = SyntheticCompilerEnv(benchmark_id=benchmark_id, seed=seed + trial)
        obs, _ = env.reset()
        total_reward = 0.0
        sequence = []

        for step in range(max_steps):
            pass_id = rng.randint(0, NUM_PASSES - 1)
            obs, feedback, done, info = env.step(pass_id)
            total_reward += feedback.reward
            sequence.append(pass_id)
            if done:
                break

        all_rewards.append(total_reward)
        all_sequences.append(sequence)
        if total_reward > best_reward:
            best_reward = total_reward
            best_sequence = list(sequence)

    return {
        "algorithm": "random_search",
        "best_reward": best_reward,
        "best_sequence": [pass_id_to_name(p) for p in best_sequence],
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "median_reward": float(np.median(all_rewards)),
        "p25_reward": float(np.percentile(all_rewards, 25)),
        "p75_reward": float(np.percentile(all_rewards, 75)),
        "num_trials": num_trials,
        "max_steps": max_steps,
    }


# ══════════════════════════════════════════════════════════════
# 3. Greedy Search
# ══════════════════════════════════════════════════════════════

def greedy_search(
    benchmark_id: str,
    max_steps: int = 30,
    seed: int = 42,
) -> dict:
    """Greedy: at each step, try all passes and pick the best.

    O(NUM_PASSES * max_steps) evaluations.
    """
    env = SyntheticCompilerEnv(benchmark_id=benchmark_id, seed=seed)
    obs, _ = env.reset()

    total_reward = 0.0
    sequence = []
    step_rewards = []

    for step in range(max_steps):
        best_pass_id = -1
        best_step_reward = float("-inf")

        for pass_id in range(NUM_PASSES):
            test_env = SyntheticCompilerEnv(benchmark_id=benchmark_id, seed=seed)
            test_obs, _ = test_env.reset()
            for prev_pass in sequence:
                test_obs, _, _, _ = test_env.step(prev_pass)
            _, feedback, _, _ = test_env.step(pass_id)

            if feedback.reward > best_step_reward:
                best_step_reward = feedback.reward
                best_pass_id = pass_id

        if best_pass_id < 0 or best_step_reward <= 0:
            break

        obs, feedback, done, info = env.step(best_pass_id)
        total_reward += feedback.reward
        sequence.append(best_pass_id)
        step_rewards.append(feedback.reward)

        if done:
            break

    return {
        "algorithm": "greedy_search",
        "total_reward": total_reward,
        "sequence": [pass_id_to_name(p) for p in sequence],
        "step_rewards": step_rewards,
        "num_steps": len(sequence),
    }


# ══════════════════════════════════════════════════════════════
# 4. Beam Search
# ══════════════════════════════════════════════════════════════

def beam_search(
    benchmark_id: str,
    max_steps: int = 30,
    beam_width: int = 5,
    seed: int = 42,
) -> dict:
    """Beam search: maintain top-k sequences at each step.

    Better than greedy but more expensive.
    O(beam_width * NUM_PASSES * max_steps) evaluations.
    """
    # Each beam entry: (reward, sequence, env_state)
    beams = [(0.0, [], SyntheticCompilerEnv(benchmark_id=benchmark_id, seed=seed))]
    beams[0][2].reset()

    best_overall_reward = 0.0
    best_overall_sequence = []

    for step in range(max_steps):
        candidates = []

        for reward_so_far, seq, env in beams:
            for pass_id in range(NUM_PASSES):
                # Clone env state by replaying from scratch
                test_env = SyntheticCompilerEnv(benchmark_id=benchmark_id, seed=seed)
                test_env.reset()
                done_inner = False
                for p in seq:
                    _, _, done_inner, _ = test_env.step(p)
                    if done_inner:
                        break
                if done_inner:
                    continue
                _, feedback, done, _ = test_env.step(pass_id)

                new_reward = reward_so_far + feedback.reward
                new_seq = seq + [pass_id]

                candidates.append((new_reward, new_seq, test_env, done))

                if new_reward > best_overall_reward:
                    best_overall_reward = new_reward
                    best_overall_sequence = list(new_seq)

        if not candidates:
            break

        # Keep top beam_width candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = []
        seen_seqs = set()
        for reward, seq, env, done in candidates:
            seq_key = tuple(seq[-5:])  # dedup by last 5 passes
            if seq_key not in seen_seqs and len(beams) < beam_width:
                seen_seqs.add(seq_key)
                beams.append((reward, seq, env))

    return {
        "algorithm": "beam_search",
        "total_reward": best_overall_reward,
        "sequence": [pass_id_to_name(p) for p in best_overall_sequence],
        "num_steps": len(best_overall_sequence),
        "beam_width": beam_width,
    }


# ══════════════════════════════════════════════════════════════
# 5. Full Benchmark Suite
# ══════════════════════════════════════════════════════════════

def run_full_benchmark(
    model,  # TinyPassOrderingRefiner or None
    benchmarks: Optional[list[str]] = None,
    max_steps: int = 30,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """Run all baselines and TRM model across benchmarks.

    Args:
        model: TRM model to evaluate (or None for baselines only)
        benchmarks: List of benchmark IDs
        max_steps: Max passes per episode
        device: Torch device
        seed: Random seed

    Returns:
        Dict with results for each algorithm × benchmark
    """
    from .model import rollout_pass_optimizer

    if benchmarks is None:
        benchmarks = ["qsort", "adpcm", "blowfish", "bzip2", "dijkstra", "sha"]

    results = {}

    for bench_id in benchmarks:
        print(f"\n{'-'*60}")
        print(f"Benchmark: {bench_id}")
        print(f"{'-'*60}")
        bench_results = {}

        # 1. LLVM optimization levels
        for level in ["O0", "O1", "O2", "O3", "Oz"]:
            r = run_optimization_level(bench_id, level=level, seed=seed)
            bench_results[f"llvm_{level}"] = r
            print(f"  LLVM {level:3s}: reward={r['total_reward']:+8.4f}  "
                  f"reduction={r['reduction_pct']*100:5.1f}%  steps={r['num_steps']}")

        # 2. Random search (various budgets)
        for budget in [10, 100, 1000]:
            r = random_search(bench_id, max_steps=max_steps, num_trials=budget, seed=seed)
            bench_results[f"random_{budget}"] = r
            print(f"  Random {budget:4d}: best={r['best_reward']:+8.4f}  "
                  f"mean={r['mean_reward']:+.4f}±{r['std_reward']:.4f}")

        # 3. Greedy search
        r = greedy_search(bench_id, max_steps=max_steps, seed=seed)
        bench_results["greedy"] = r
        print(f"  Greedy:    reward={r['total_reward']:+8.4f}  steps={r['num_steps']}")

        # 4. Beam search
        r = beam_search(bench_id, max_steps=max_steps, beam_width=5, seed=seed)
        bench_results["beam_5"] = r
        print(f"  Beam(5):   reward={r['total_reward']:+8.4f}  steps={r['num_steps']}")

        # 5. TRM blind (no real feedback)
        if model is not None:
            from .model import rollout_blind, rollout_closed_loop
            model.eval()

            # Blind: observation never updates
            env_blind = SyntheticCompilerEnv(benchmark_id=bench_id, seed=seed)
            obs, _ = env_blind.reset()
            total_reward = 0.0
            sequence = []
            for step in range(max_steps):
                trace = rollout_blind(model, obs, max_steps=1,
                                      temperature=1.0, device=device)
                if trace and trace[0]["pass_id"] >= 0:
                    pass_id = trace[0]["pass_id"]
                    obs, feedback, done, info = env_blind.step(pass_id)
                    total_reward += feedback.reward
                    sequence.append(pass_id)
                    if done:
                        break
                else:
                    break
            bench_results["trm_blind"] = {
                "total_reward": total_reward,
                "sequence": [pass_id_to_name(p) for p in sequence],
                "num_steps": len(sequence),
            }
            print(f"  TRM blind: reward={total_reward:+8.4f}  steps={len(sequence)}  "
                  f"inst={env_blind.current_inst_count}/{env_blind.initial_inst_count}")

            # Closed-loop: real feedback after each pass
            env_cl = SyntheticCompilerEnv(benchmark_id=bench_id, seed=seed)
            trace_cl = rollout_closed_loop(model, env_cl, max_steps=max_steps,
                                           temperature=1.0, device=device)
            cl_reward = trace_cl[-1]["total_real_reward"] if trace_cl else 0.0
            cl_steps = len([s for s in trace_cl if s["pass_id"] >= 0])
            bench_results["trm_closed_loop"] = {
                "total_reward": cl_reward,
                "sequence": [pass_id_to_name(s["pass_id"]) for s in trace_cl if s["pass_id"] >= 0],
                "num_steps": cl_steps,
            }
            print(f"  TRM closed: reward={cl_reward:+8.4f}  steps={cl_steps}  "
                  f"inst={env_cl.current_inst_count}/{env_cl.initial_inst_count}")

        results[bench_id] = bench_results

    # Summary
    print(f"\n{'='*60}")
    print("Summary (mean reward across benchmarks)")
    print(f"{'='*60}")

    algorithms = set()
    for bench_results in results.values():
        algorithms.update(bench_results.keys())

    for algo in sorted(algorithms):
        rewards = []
        for b in results:
            if algo in results[b]:
                entry = results[b][algo]
                # Different algorithms store reward under different keys
                r = entry.get("total_reward", entry.get("best_reward", 0.0))
                rewards.append(r)
        if rewards:
            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            print(f"  {algo:20s}: {mean_r:+8.4f} +/- {std_r:.4f}")

    return results
