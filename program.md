# TRM Autoresearch Program

You are an AI research agent working on Tiny Recursive Model (TRM) architectures for compiler pass ordering optimization.

## Goal

Improve the **validation reward** (higher is better) of the TRM model that learns to order LLVM optimization passes. The model is trained via imitation learning on compiler traces and evaluated on held-out benchmarks.

## How It Works

1. Modify `train.py` — this is the single file you experiment with
2. Run `python run_experiment.py` — trains for a fixed time budget and reports results
3. Check the experiment log in `experiments/` to see if your change improved things
4. If it improved, keep the change and try something new. If not, revert and try a different approach.

## Constraints

- **DO NOT ask the user any questions** — The research loop is designed to run autonomously (e.g., overnight while you sleep). Asking questions stops the loop. If you're unsure about something, make a reasonable assumption and continue. Do not wait for user input.
- **Time budget:** Each experiment runs for exactly **10 minutes** of training (wall clock, excluding startup)
- **Metric:** `val_reward` on held-out benchmarks (qsort, adpcm, blowfish) — **higher is better**
- **Only modify:** `train.py` — do NOT modify `prepare.py`, `run_experiment.py`, or `program.md`
- **Device:** Training runs on whatever device is available (cuda/cpu)

## Current Baseline

The baseline `train.py` implements:
- TRM architecture with dual latents (y, z) per the paper
- 2-layer MLPs with SiLU activation
- Inner recursion loop (n=6)
- Multi-head outputs: pass selection, feasibility, value, halt
- AdamW optimizer with cosine annealing
- Batch size 64, latent dim 64, hidden dim 128
- Loss: pass_loss + entropy_loss + 0.5*feasibility + 0.3*value + 0.2*halt

## Experiment Ideas

Here are directions to explore (pick one at a time, be systematic):

### Architecture
- Change network depth (try 3-layer MLPs instead of 2)
- Change latent dimension (try 32, 128, 256)
- Change hidden dimension (try 64, 256, 512)
- Try different activation functions (ReLU, GELU, Tanh)
- Add layer normalization or residual connections
- Try different output head architectures
- Add dropout regularization

### Training Dynamics
- Change learning rate (try 1e-4, 5e-4, 5e-3)
- Change batch size (try 32, 128, 256)
- Change number of recursions (try 3, 10, 12)
- Try different optimizers (SGD+momentum, Adam, Muon)
- Add gradient clipping with different thresholds
- Try warmup + cosine decay schedule
- Change loss weights for different heads

### Data & Representation
- Change feedback encoding (try rich 20-dim instead of simple 4-dim)
- Change schedule encoding features
- Add observation normalization
- Try different trace generation strategies
- Change context length for schedule encoding

### Regularization
- Change entropy coefficient
- Add weight decay variations
- Add dropout to the network
- Try label smoothing for pass selection
- Add early stopping based on validation

### Advanced
- Implement EMA (exponential moving average) of weights
- Try deep supervision (supervise intermediate recursion steps)
- Implement curriculum learning (easy benchmarks first)
- Try multi-task learning with auxiliary objectives
- Implement knowledge distillation from a larger model

## Experiment Protocol

1. **Read** the current `train.py` and understand what it does
2. **Make ONE change** — isolate variables, change one thing at a time
3. **Run** `python run_experiment.py`
4. **Read** the experiment result from the output and `experiments/results.jsonl`
5. **Decide:** If `val_reward` improved over the best so far, keep it. Otherwise revert.
6. **Update** this file's "Current Best" section below if you found something better
7. **Repeat** with a new idea

## Experiment Log Format

Each experiment produces a line in `experiments/results.jsonl`:
```json
{"experiment_id": 1, "timestamp": "...", "val_reward": 2.34, "train_loss": 0.45, "config": {...}}
```

## Current Best

| Experiment | val_reward | Changes |
|---|---|---|
| Baseline (default) | +1.4510 | Original configuration |
| **#4 (latent=128, 3-layer)** | **+1.8103** | Latent dim 64→128, 3-layer MLPs |

### Fair Benchmark Results (100 epochs each, same conditions)

| Config | val_reward | Time | Notes |
|---|---|---|---|
| **baseline (N_sup=1)** | **+1.7914** | 243s | **Best** |
| deep_sup_2 | +1.5695 | 449s | Worse + 1.8x slower |
| deep_sup_4 | +1.6498 | 826s | Worse + 3.4x slower |

**Key finding:** Deep supervision consistently underperforms, even at equal epochs. Pass ordering is a direct mapping task — it benefits more from clean single-step gradients than progressive supervision. Do NOT pursue deep supervision further.

## Notes

- The TRM paper uses 2-layer networks with dual latents — our best uses 3-layer MLPs
- The compiler pass ordering task has 37 possible passes and 18 benchmarks
- Training data is generated from synthetic compiler traces (random + greedy strategies)
- The model has ~137K parameters in the current best configuration
- Lower validation loss during training doesn't always mean higher evaluation reward
- **Deep supervision is empirically worse for this task** — see `paper_comparison.md` for full analysis
