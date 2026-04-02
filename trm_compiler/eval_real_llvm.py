"""Real LLVM evaluation via CompilerGym.

Compares TRM model against LLVM -Oz, -O3, and random search
using actual LLVM compiler passes on cbench benchmarks.
"""
from __future__ import annotations
import random as py_random
import math
import torch
import numpy as np

from .env_wrapper import (
    CompilerGymWrapper, pass_id_to_name, NUM_PASSES, _has_compilergym,
)
from .model import TinyPassOrderingRefiner, rollout_pass_optimizer
from .baselines import _OZ_PIPELINE, _O3_PIPELINE


def eval_on_compilergym(benchmark_id, pass_selector, max_steps=30):
    """Run a pass selection strategy on real LLVM.

    Args:
        benchmark_id: e.g. 'qsort' (will be prefixed with 'cbench-v1/')
        pass_selector: callable(step, obs, history) -> pass_id or None to stop
        max_steps: maximum passes to apply

    Returns:
        dict with results, or None on error
    """
    full_id = f'cbench-v1/{benchmark_id}'
    wrapper = CompilerGymWrapper(benchmark_id=full_id)

    try:
        wrapper.open()
        obs, initial_inst = wrapper.reset()

        steps = []
        total_reward = 0.0

        for step in range(max_steps):
            pass_id = pass_selector(step, obs, list(steps))
            if pass_id is None or pass_id < 0:
                break

            prev_inst = wrapper._current_inst_count
            obs, feedback, done, info = wrapper.step(pass_id)
            new_inst = wrapper._current_inst_count

            steps.append({
                'step': step,
                'pass_id': pass_id,
                'pass_name': pass_id_to_name(pass_id),
                'prev_inst': prev_inst,
                'new_inst': new_inst,
                'reward': feedback.reward,
            })
            total_reward += feedback.reward

            if done:
                break

        final_inst = wrapper._current_inst_count
        reduction = 1.0 - final_inst / max(initial_inst, 1)

    except Exception as e:
        print(f'  Error on {benchmark_id}: {e}')
        return None
    finally:
        wrapper.close()

    return {
        'benchmark': benchmark_id,
        'initial_inst': initial_inst,
        'final_inst': final_inst,
        'reduction_pct': reduction,
        'total_reward': total_reward,
        'num_steps': len(steps),
        'steps': steps,
    }


def random_selector(seed=42):
    """Random pass selector."""
    rng = py_random.Random(seed)

    def select(step, obs, history):
        return rng.randint(0, NUM_PASSES - 1)

    return select


def trm_selector(model, device='cpu', temperature=0.8):
    """TRM model pass selector."""

    def select(step, obs, history):
        trace = rollout_pass_optimizer(
            model, obs, max_steps=1,
            temperature=temperature, device=device,
        )
        if trace and trace[0]['pass_id'] >= 0:
            return trace[0]['pass_id']
        return None

    return select


def run_pipeline_on_compilergym(benchmark_id, pipeline, max_steps=30):
    """Run a fixed list of pass names on real CompilerGym."""
    from .env_wrapper import pass_name_to_id

    full_id = f'cbench-v1/{benchmark_id}'
    wrapper = CompilerGymWrapper(benchmark_id=full_id)

    try:
        wrapper.open()
        obs, initial_inst = wrapper.reset()

        steps = []
        total_reward = 0.0

        for pass_name in pipeline[:max_steps]:
            try:
                pass_id = pass_name_to_id(pass_name)
            except KeyError:
                continue

            prev_inst = wrapper._current_inst_count
            try:
                obs, feedback, done, info = wrapper.step(pass_id)
            except Exception:
                continue

            steps.append({
                'step': len(steps),
                'pass_id': pass_id,
                'pass_name': pass_name,
                'prev_inst': prev_inst,
                'new_inst': wrapper._current_inst_count,
                'reward': feedback.reward,
            })
            total_reward += feedback.reward

            if done:
                break

        final_inst = wrapper._current_inst_count
        reduction = 1.0 - final_inst / max(initial_inst, 1)

    except Exception as e:
        print(f'  Error: {e}')
        return None
    finally:
        wrapper.close()

    return {
        'benchmark': benchmark_id,
        'initial_inst': initial_inst,
        'final_inst': final_inst,
        'reduction_pct': reduction,
        'total_reward': total_reward,
        'num_steps': len(steps),
        'steps': steps,
    }


def evaluate_model_on_real_llvm(
    model, benchmarks, max_steps=30, device='cpu',
    num_random_trials=100, seed=42,
):
    """Full evaluation: LLVM -Oz, -O3, Random(100), TRM on real CompilerGym.

    Args:
        model: trained TinyPassOrderingRefiner
        benchmarks: list of benchmark IDs (e.g. ['qsort', 'adpcm'])
        max_steps: max passes per episode
        device: torch device
        num_random_trials: random search budget
        seed: random seed

    Returns:
        dict: {benchmark: {algorithm: result_dict}}
    """
    if not _has_compilergym():
        raise ImportError(
            'compiler_gym not installed. Install with: pip install compiler_gym'
        )

    model.eval()
    all_results = {}

    for bench_id in benchmarks:
        print(f'\n{"="*55}')
        print(f'  {bench_id}')
        print(f'{"="*55}')
        all_results[bench_id] = {}

        # LLVM -Oz
        print('  [1/4] LLVM -Oz ...', end=' ', flush=True)
        r = run_pipeline_on_compilergym(bench_id, _OZ_PIPELINE, max_steps)
        if r:
            all_results[bench_id]['LLVM-Oz'] = r
            print(f'reward={r["total_reward"]:+.4f}  reduction={r["reduction_pct"]*100:.1f}%  steps={r["num_steps"]}')

        # LLVM -O3
        print('  [2/4] LLVM -O3 ...', end=' ', flush=True)
        r = run_pipeline_on_compilergym(bench_id, _O3_PIPELINE, max_steps)
        if r:
            all_results[bench_id]['LLVM-O3'] = r
            print(f'reward={r["total_reward"]:+.4f}  reduction={r["reduction_pct"]*100:.1f}%  steps={r["num_steps"]}')

        # Random search (best of N trials)
        print(f'  [3/4] Random({num_random_trials}) ...', end=' ', flush=True)
        best_random = None
        for trial in range(num_random_trials):
            sel = random_selector(seed=seed + trial)
            r = eval_on_compilergym(bench_id, sel, max_steps)
            if r and (best_random is None or r['total_reward'] > best_random['total_reward']):
                best_random = r
        if best_random:
            all_results[bench_id]['Random'] = best_random
            print(f'best_reward={best_random["total_reward"]:+.4f}  reduction={best_random["reduction_pct"]*100:.1f}%')

        # TRM
        print('  [4/4] TRM ......', end=' ', flush=True)
        sel = trm_selector(model, device=device)
        r = eval_on_compilergym(bench_id, sel, max_steps)
        if r:
            all_results[bench_id]['TRM'] = r
            print(f'reward={r["total_reward"]:+.4f}  reduction={r["reduction_pct"]*100:.1f}%  steps={r["num_steps"]}')

    return all_results


def print_results_table(all_results, benchmarks):
    """Print results as a formatted table."""
    algorithms = set()
    for bench_r in all_results.values():
        algorithms.update(bench_r.keys())
    algorithms = sorted(algorithms)

    # Reward table
    print(f'\n{"Algorithm":<12s}', end='')
    for b in benchmarks:
        print(f' {b:>10s}', end='')
    print(f' {"Mean":>10s}')
    print('-' * (12 + 11 * (len(benchmarks) + 1)))

    for algo in algorithms:
        print(f'{algo:<12s}', end='')
        vals = []
        for b in benchmarks:
            if b in all_results and algo in all_results[b]:
                v = all_results[b][algo]['total_reward']
                vals.append(v)
                print(f' {v:+10.4f}', end='')
            else:
                print(f' {"—":>10s}', end='')
        print(f' {np.mean(vals):+10.4f}' if vals else '')

    # Reduction table
    print(f'\n{"Algorithm":<12s}', end='')
    for b in benchmarks:
        print(f' {b:>10s}', end='')
    print(f' {"Mean":>10s}')
    print('-' * (12 + 11 * (len(benchmarks) + 1)))

    for algo in algorithms:
        print(f'{algo:<12s}', end='')
        vals = []
        for b in benchmarks:
            if b in all_results and algo in all_results[b]:
                v = all_results[b][algo]['reduction_pct'] * 100
                vals.append(v)
                print(f' {v:9.1f}%', end='')
            else:
                print(f' {"—":>10s}', end='')
        print(f' {np.mean(vals):9.1f}%' if vals else '')
