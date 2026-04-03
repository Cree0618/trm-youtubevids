# TRM Experiments — Recursive Reasoning for Compiler & Kernel Optimization

Tiny Recursive Model (TRM) implementations for finding fast kernels and optimal compiler passes.

**Paper:** "Less is More: Recursive Reasoning with Tiny Networks" — [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)

This repository does not require the legacy `gym` package. If you need a
maintained Gym-style environment API for extensions or wrappers, use
`gymnasium`, which is the actively maintained successor to OpenAI Gym.

---

## Autoresearch — Autonomous AI-Driven Experimentation

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch), this project supports autonomous overnight experimentation. An AI agent modifies `train.py`, runs experiments with a fixed time budget, evaluates results, and iterates — all without human intervention.

### How It Works

1. **`prepare.py`** — Fixed data preparation (generates training traces). Not modified by agents.
2. **`train.py`** — The single file agents modify. Contains model, optimizer, loss, and training loop.
3. **`program.md`** — Agent instructions. Humans edit this to guide research direction.
4. **`run_experiment.py`** — Fixed experiment runner. Trains for 5 minutes, evaluates, logs results.

### Quick Start for Autoresearch

```bash
# 1. Generate training data (one-time)
python prepare.py

# 2. Run a single experiment (~5 min training + eval)
python run_experiment.py

# 3. View experiment history
python run_experiment.py --history
```

### Running Autonomous Research

Give an AI agent (Claude, Codex, etc.) access to this repo and prompt:

```
Read program.md and run a new experiment!
```

**Important:** Instruct the agent **not to ask any questions** — the research loop runs autonomously (e.g., overnight while you sleep). Asking questions stops the loop. If unsure, the agent should make a reasonable assumption and continue.

The agent will:
1. Read `program.md` for research context
2. Modify `train.py` (architecture, hyperparameters, etc.)
3. Run `python run_experiment.py`
4. Check if `val_reward` improved
5. Keep or revert changes
6. Repeat with new ideas

Each experiment logs to `experiments/results.jsonl`. The best model is saved to `experiments/best_model/`.

---

## Projects

### `trm_compiler/` — Compiler Pass Ordering (ACTIVE)

TRM model that learns to order LLVM optimization passes for maximal instruction count reduction.

- **60,328 parameters** with TRM-aligned architecture (dual networks, inner recursion)
- **37 LLVM passes** with realistic synergy modeling
- **18 benchmark programs** with different optimization profiles
- Works on **all platforms** (synthetic env) or **Linux** (real CompilerGym)

```bash
python -m trm_compiler.example --epochs 20
```

### `trm_gemm/` — GEMM Kernel Schedule Optimization

TRM model for searching optimal Triton GEMM kernel configurations (block sizes, warps, stages).

- Original prototype
- Turing-first schedule space
- Heuristic backend for portability

```bash
python -m trm_gemm.example
```

---

## Quick Start

```bash
# Install
pip install torch numpy

# Run compiler pass ordering
python -m trm_compiler.example --epochs 10

# Run GEMM schedule optimization
python -m trm_gemm.example

# Run tests
python trm_compiler/test_pipeline.py
pytest -q
```

### With CompilerGym (Linux)

```bash
pip install compiler_gym
python -m trm_compiler.example --compilergym --epochs 20
```

### Gym-style API compatibility

```bash
pip install "trm-experiments[env]"
```

Use this only if you want to build Gym-style wrappers around the project
environments. The maintained dependency is `gymnasium`, not `gym`.

---

## TRM Paper Alignment

| Paper Concept | Implementation |
|---|---|
| Dual latents (answer `y` + reasoning `z`) | Separate `net_z` and `net_y` |
| Inner recursion loop (`n=6`) | `for _ in range(n): z = net_z(x, y, z)` |
| 2-layer networks | 2-layer MLPs (per paper) |
| Multi-head output | pass + feasibility + value + halt |
| Log-ratio reward | `log(prev / next)` for scale invariance |

**Not yet implemented:** Deep recursion (T-1 no-grad), Deep supervision, EMA.

---

## Documentation

| File | Contents |
|---|---|
| [TRM_REVIEW.md](TRM_REVIEW.md) | Paper review, alignment analysis, reward design |
| [TRM_NEW_APPLICATIONS.md](TRM_NEW_APPLICATIONS.md) | Compiler pass ordering + LLM HPO design |
| [TRM_COMPILER_TOOLS_DATASETS.md](TRM_COMPILER_TOOLS_DATASETS.md) | Tools, benchmarks, implementation details |
| [README_trm_gemm.md](README_trm_gemm.md) | GEMM-specific documentation |

---

## Project Structure

```
trm-youtubevids/
├── train.py                 # Model + training loop (MODIFIED BY AGENT)
├── prepare.py               # Data preparation (FIXED)
├── run_experiment.py        # Experiment runner (FIXED)
├── program.md               # Agent instructions (MODIFIED BY HUMAN)
├── trm_compiler/            # Compiler pass ordering (active)
│   ├── model.py             # TinyPassOrderingRefiner
│   ├── env_wrapper.py       # SyntheticCompilerEnv + CompilerGym
│   ├── data.py              # Trace generation
│   ├── training.py          # Training loop
│   ├── baselines.py         # Random/greedy baselines
│   └── example.py           # CLI entry point
├── trm_gemm/                # GEMM kernel optimization
│   ├── model.py             # TinyRecursiveGemmRefiner
│   ├── backend.py           # Triton + heuristic backend
│   └── ...
├── experiments/             # Autoresearch results (auto-created)
│   ├── results.jsonl        # Experiment log
│   └── best_model/          # Best model + train.py
├── data/                    # Training traces (auto-created)
│   └── traces.json
├── TRM_REVIEW.md            # Paper analysis
├── TRM_NEW_APPLICATIONS.md  # Application designs
├── TRM_COMPILER_TOOLS_DATASETS.md  # Tools & benchmarks
└── pyproject.toml
```

## License

Research prototype — see individual files for details.
