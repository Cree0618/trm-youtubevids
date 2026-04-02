# Compiler Pass Ordering — Tools, Datasets & Implementation

**Date:** 2026-04-01
**Status:** Implementation complete. See `trm_compiler/` for code.

---

## 1. Environment Options

### CompilerGym (Meta/Facebook Research)

**GitHub:** https://github.com/facebookresearch/CompilerGym
**Stars:** 1k | **License:** MIT | **Last release:** v0.2.5 (Nov 2022)

**Platform support:** Linux (primary), macOS (partial). **No Windows wheels.**

```bash
pip install compiler_gym  # Linux only
```

**Leaderboard (llvm-ic-v0 on cbench-v1):**
| Algorithm | Speedup | Walltime |
|---|---|---|
| PPO + Guided Search | **1.070x** | 69.8s |
| Random search (3hr) | 1.062x | 10,512s |
| Greedy search | 1.055x | 169s |
| DQN | 1.029x | 91s |

### SyntheticCompilerEnv (Custom — All Platforms)

**Location:** `trm_compiler/env_wrapper.py`

Realistic synthetic environment that models real LLVM pass ordering behavior without requiring CompilerGym. Works on Windows/macOS/Linux.

**Features modeled:**
- 37 LLVM passes with realistic effectiveness ranges
- 16 pass synergy pairs (instcombine→gvn, loop-unroll→vectorize)
- 4 anti-patterns (loop-vectorize→loop-rotate)
- Diminishing returns on repeated passes
- 18 benchmark profiles (different optimization potential)

**Usage:**
```python
from trm_compiler import SyntheticCompilerEnv

env = SyntheticCompilerEnv("qsort", seed=42)
obs, initial_inst = env.reset()
obs, feedback, done, info = env.step(pass_id=0)
```

### MLIR-RL (NYU Abu Dhabi)

**GitHub:** https://github.com/Modern-Compilers-Lab/MLIR-RL
**Paper:** arXiv:2409.11068

Targets MLIR's Linalg dialect. Requires building LLVM 19.x from source. More relevant for GPU GEMM optimization than LLVM pass ordering.

---

## 2. Benchmark Datasets Used

| Dataset | Programs | Used In |
|---|---|---|
| CBench subset | 18 | `SyntheticCompilerEnv` profiles |
| PolyBench/C | 30 | Available via CompilerGym |
| CHStone | 12 | CompilerGym bundled |

**Default benchmarks in `trm_compiler`:**
```
qsort, adpcm, blowfish, bzip2, dijkstra, gsm, ispell, jpeg-c,
lame, patricia, rijndael, sha, stringsearch, susan, tiff2bw,
tiff2rgba, tiffdither, tiffmedian
```

---

## 3. LLVM Passes (37 total)

| Pass | Base Effectiveness | Synergies |
|---|---|---|
| `mem2reg` | 2-15% | instcombine (1.3x), sroa (1.3x), gvn (1.2x) |
| `simplifycfg` | 1-10% | instcombine (1.3x) |
| `instcombine` | 5-25% | gvn (1.4x), simplifycfg (1.2x) |
| `gvn` | 3-20% | dce (1.3x), licm (1.2x) |
| `loop-unroll` | 2-15% | loop-vectorize (1.5x), slp-vectorize (1.3x) |
| `loop-vectorize` | 3-20% | — |
| `slp-vectorize` | 2-12% | — |
| `inline` | 2-18% | instcombine (1.4x), gvn (1.3x) |
| `sroa` | 2-12% | — |
| `licm` | 2-12% | — |
| ... | ... | ... |

Full list: `trm_compiler/env_wrapper.py` → `USEFUL_PASSES` and `_PASS_EFFECTIVENESS`

---

## 4. Implementation Architecture

```
trm_compiler/
├── __init__.py          # Package exports
├── types.py             # BenchmarkSpec, CompilerFeedback, PassEdit, CompilerTraceRecord
├── env_wrapper.py       # SyntheticCompilerEnv + CompilerGymWrapper + NUM_PASSES + synergies
├── model.py             # TinyPassOrderingRefiner (TRM: net_z + net_y, inner recursion)
├── data.py              # Trace generation, CompilerTraceDataset
├── training.py          # Multi-head loss, train_one_epoch, evaluate_model
├── baselines.py         # random_pass_search, greedy_pass_search
├── example.py           # CLI entry point
└── test_pipeline.py     # End-to-end test
```

### Model

- **60,328 parameters** (tiny)
- **Dual networks**: `net_z` (reasoning: 192→128→64), `net_y` (refinement: 128→128→64)
- **Inner recursion**: `n=6` steps of z-update before y refinement
- **Multi-head**: pass(37), feasibility(1), value(1), halt(1)

### Training

- **Loss**: pass_loss + 0.5*feasibility + 0.3*value + 0.2*halt
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Schedule**: CosineAnnealingLR

### Reward

```python
log_reward = math.log(prev_instruction_count / max(next_instruction_count, 1))
```

---

## 5. Test Results

### Training (20 epochs, 53,407 traces)

| Epoch | Loss | Pass Loss | Value Loss |
|---|---|---|---|
| 1 | 3.64 | 3.59 | 0.14 |
| 10 | 1.50 | 1.50 | 0.004 |
| 20 | 0.02 | 0.02 | 0.003 |

### Rollout (qsort, 2 epochs trained)

```
800 → 503 instructions (37% reduction in 10 steps)
Diverse pass selection: ipconstprop, slp-vectorize, instcombine-2,
loop-rotate, simplifycfg-2, loop-unroll, loop-deletion, gvn
```

### Baselines (Synthetic Environment)

| Algorithm | Mean Reward |
|---|---|
| Random (100 trials) | 0.94 ± 0.20 |
| Greedy | 1.93 |
| TRM (2 epochs) | 0.46 (10 steps) |

---

## 6. Key Papers

| Paper | Year | Relevance |
|---|---|---|
| **TRM** (Jolicoeur-Martineau) | 2025 | Core architecture |
| **CompilerGym** (Cummins et al.) | 2022 | RL environment, cbench dataset |
| **MLIR-RL** (Tirichine et al.) | 2024 | Multi-discrete action space |
| **AutoPhase** (Haj-Ali et al.) | 2020 | DRL for HLS phase ordering |
| **Compiler-R1** (Pan et al.) | 2025 | Agentic RL for compiler tuning |
| **ML for Compiler Optimization** (Wang & O'Boyle) | 2018 | Survey |

---

## 7. Usage Commands

```bash
# Synthetic environment (any platform)
python -m trm_compiler.example --epochs 20

# Real CompilerGym (Linux only)
pip install compiler_gym
python -m trm_compiler.example --compilergym --epochs 20

# Evaluate only
python -m trm_compiler.example --eval

# Full pipeline test
python trm_compiler/test_pipeline.py

# Custom parameters
python -m trm_compiler.example --epochs 30 --episodes 200 --max-steps 40 --lr 5e-4
```
