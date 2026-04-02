# TRM Architecture — Application Domains

**Date:** 2026-04-01
**Based on:** "Less is More: Recursive Reasoning with Tiny Networks" (Jolicoeur-Martineau, 2025)

---

## Application 1: Compiler Pass Ordering — IMPLEMENTED

**Status:** Fully implemented in `trm_compiler/`. End-to-end tested on Windows with synthetic environment.

### Implementation Summary

| Component | File | Status |
|---|---|---|
| TRM Model | `trm_compiler/model.py` | Done — inner recursion, dual networks |
| Environment | `trm_compiler/env_wrapper.py` | Done — synthetic (Windows) + CompilerGym (Linux) |
| Trace Generation | `trm_compiler/data.py` | Done — 18 benchmarks, configurable episodes |
| Training Loop | `trm_compiler/training.py` | Done — multi-head loss, cosine LR schedule |
| Baselines | `trm_compiler/baselines.py` | Done — random + greedy search |
| Entry Point | `trm_compiler/example.py` | Done — CLI with `--compilergym` flag |

### Usage

```bash
# Synthetic environment (any platform)
python -m trm_compiler.example --epochs 20

# Real CompilerGym (Linux only)
pip install compiler_gym
python -m trm_compiler.example --compilergym --epochs 20

# Evaluate only (skip training)
python -m trm_compiler.example --eval

# Full pipeline test
python trm_compiler/test_pipeline.py
```

### Architecture

```python
class TinyPassOrderingRefiner(nn.Module):
    # Input: observation(56) + schedule(4) + feedback(4) = 64 dims
    # Dual networks per TRM paper:
    net_z: Linear(192, 128) → SiLU → Linear(128, 64)  # reasoning
    net_y: Linear(128, 128) → SiLU → Linear(128, 64)  # answer refinement

    # Inner recursion: n=6 steps of z-update before y refinement
    for _ in range(n):
        z = net_z(concat(x, y, z))
    y = net_y(concat(x, z))

    # Multi-head output
    pass_head(64, 37)        # pass selection
    feasibility_head(64, 1)  # compile likelihood
    value_head(64, 1)        # reward prediction
    halt_head(64, 1)         # stop signal
```

### Key Design Decisions

1. **Log-ratio reward**: `log(prev_inst / next_inst)` — scale-invariant, symmetric
2. **Soft pass penalty**: Recently applied passes get -2.0 logit penalty
3. **Minimum exploration**: 5 passes before halt can trigger
4. **Benchmark profiles**: 18 programs with different optimization potential
5. **Synthetic synergies**: Models real compiler behavior (instcombine→gvn, unroll→vectorize)

---

## Application 2: LLM Training Hyperparameter Optimization — FUTURE

### The Problem

Training LLMs requires tuning 15-30+ hyperparameters (lr, batch size, warmup, dropout, weight decay, gradient clipping, schedule type). Currently trial-and-error.

### Why TRM Fits

- Sequential refinement of hyperparameter configs
- Extremely expensive evaluation (hours/days per config)
- Patterns exist (lr interacts with batch size)
- Small data (only hundreds of public training runs)

### Two Modes

**Mode A: Pre-training Config Search** — Search for optimal config before training starts using proxy training (500 steps).

**Mode B: Online Adaptation** — Adjust hyperparameters during training based on loss curve.

### TRM Mapping

| TRM Concept | LLM HPO Equivalent |
|---|---|
| Input `x` | Model spec (params, data size, hardware) |
| Answer `y` | Current hyperparameter configuration |
| Reasoning `z` | Understanding of param interactions |
| Edit | Modify one hyperparameter |
| Reward | -final_loss or composite metric |

### Action Space

```python
ALL_EDITS = [
    ("learning_rate", "multiply_2"),
    ("learning_rate", "divide_2"),
    ("warmup_steps", "increase_2x"),
    ("batch_size", "multiply_2"),
    ("dropout", "increase_0.05"),
    ("grad_clip", "decrease_0.5"),
    ("schedule", "cosine"),
    # ... ~20 edits total
]
```

### Implementation Status

Not started. Would require:
1. GPU cluster access
2. Historical training run data
3. Proxy training infrastructure
4. More complex reward design

---

## Comparison: Both Applications

| Dimension | Compiler Pass Ordering | LLM HPO |
|---|---|---|
| **Action space size** | 37 (LLVM passes) | ~20 (config edits) |
| **Evaluation cost** | Seconds (instruction count) | Hours-days (training) |
| **State dimensionality** | 56 (Autophase) | 50-200 (model + training stats) |
| **Training data** | Thousands of programs | Hundreds of training runs |
| **Reward signal** | log(speedup) — clean | Composite — noisy |
| **Infrastructure** | Single machine | GPU cluster |
| **Status** | **Implemented** | Future |

### Shared TRM Core

Both applications use the same architecture:
- Dual networks (net_z, net_y)
- Inner recursion loop (n=6)
- Multi-head output (action + feasibility + value + halt)
- Log-ratio reward

The model can be ported between domains by changing:
- Observation encoder
- Action space
- Reward function
- Environment backend
