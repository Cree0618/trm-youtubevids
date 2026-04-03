# TRM Paper vs. Our Implementation — Detailed Comparison

## Paper Reference

**"Less is More: Recursive Reasoning with Tiny Networks"** — Alexia Jolicoeur-Martineau (arXiv:2510.04871, Oct 2025)

---

## 1. Architecture

| Aspect | TRM Paper | Our Implementation |
|---|---|---|
| **Networks** | Single tiny 2-layer MLP | Two 3-layer MLPs (net_z + net_y) |
| **Latents** | y (answer) and z (reasoning) | y and z — same |
| **net_z input** | (x, y, z) → z | (x, y, z) → z — same |
| **net_y input** | (x, z) → y | (x, z) → y — same |
| **Inner recursions** | N times inside each supervision step | N_RECURSIONS=6 — same concept |
| **Supervision steps** | N_sup=16 outer steps with deep supervision | **Only 1 supervision step** (single forward pass) |
| **Activation** | GeLU | SiLU |
| **Parameters** | 7M (ARC-AGI task) | ~104K (much smaller task) |

### The Paper's Core TRM Algorithm

```
For each supervision step s = 1..N_sup:
    For inner step i = 1..N:
        z = f_θ(x, y, z)        # recursive reasoning (N times)
    y = g_φ(x, z)               # refine answer
    Compute loss(y, y_target)   # deep supervision
    Detach z from graph         # z carries forward but no BPTT
```

### Our Current Implementation

```
y, z = zeros
For i = 1..N_RECURSIONS:
    z = net_z(x, y, z)
y = net_y(x, z)
Compute multi-head loss (pass, feas, value, halt)
```

**We are doing "TRM with N_sup=1"** — a single supervision step with no deep supervision loop.

---

## 2. Deep Supervision — Missing Entirely

The paper shows deep supervision doubled accuracy (19% → 39%) on ARC-AGI, while recursive reasoning alone was only incremental (35.7% → 39.0%).

Key mechanics:
- Multiple outer steps (N_sup=16)
- At each step, compute loss against the target
- After each step, **detach z from the computational graph** so gradients don't propagate backward through previous steps
- The detached z is carried forward as initialization for the next step
- This emulates a very deep network that would be too memory-expensive to run in one forward pass

**Our implementation has zero deep supervision** — we do a single pass through the recursion loop and compute loss once.

---

## 3. ACT (Adaptive Computation Time) — Partially Present

| Aspect | TRM Paper | Our Implementation |
|---|---|---|
| **Halt head** | Q-head predicts halt/continue | We have `head_halt` |
| **ACT halt loss** | BCE on whether to stop | BCE on (reward ≤ 0) — different target |
| **ACT continue loss** | BCE on whether to continue | Not implemented |
| **Early stopping** | Model learns when to halt | Not used during training |
| **Extra forward pass** | Paper needs extra pass for ACT continue loss | We don't |

The paper's ACT mechanism:
- `ACT_halt`: trains the model to predict whether its current answer is correct (should halt)
- `ACT_continue`: trains the model to predict whether it should take another supervision step
- Together these let the model learn optimal computation time per input

Our halt head uses a different signal: `(reward <= 0)` — predicting whether a pass was unhelpful, not whether reasoning should stop.

---

## 4. Training Objective — Different

| Aspect | TRM Paper | Our Implementation |
|---|---|---|
| **Task** | Supervised: predict answer y | RL-style: predict next pass |
| **Loss** | Cross-entropy on answer + ACT losses | Multi-head: pass CE + feas BCE + value MSE + halt BCE + entropy |
| **Reward weighting** | Not applicable | Reward-weighted pass loss |
| **Gradient flow** | Detach z between supervision steps | Full graph, single step |

---

## 5. What the Paper Does That We Don't

1. **Deep supervision loop**: Multiple outer steps where z is detached and carried forward, computing loss at each step. This is the primary driver of performance gains.

2. **ACT with Q-learning**: Two-head system (halt + continue) that learns optimal computation time per input.

3. **EMA (Exponential Moving Average)**: Weights are averaged over training for more stable evaluation.

4. **No BPTT through recursion**: The 1-step gradient approximation — only the last inner recursion step gets gradients. Earlier steps run under `torch.no_grad()`. This prevents vanishing/exploding gradients through deep recursion.

5. **Attention-free design rationale**: The paper explicitly chooses MLPs over Transformers for small fixed-context tasks.

---

## 6. What We Do That the Paper Doesn't

1. **Multi-head output**: We predict pass selection, feasibility, value, and halt simultaneously — the paper only predicts the answer y.

2. **Reward-weighted loss**: Better passes get higher weight in the cross-entropy loss.

3. **Entropy regularization**: Prevents policy collapse in pass selection.

4. **Value head**: Critic-like signal for reward prediction.

5. **Feasibility head**: Predicts whether a pass will compile successfully.

---

## 7. Key Insights

### The "Less is More" Claim

The paper's main thesis is that a single 2-layer network recursing is better than HRM's complex dual-frequency hierarchy (two networks at different frequencies). Our implementation captures the dual-network recursive structure (net_z + net_y) but:

- We use 3-layer MLPs instead of 2-layer
- We have two separate networks instead of one shared network

### The Deep Supervision Gap

The most significant missing piece is deep supervision. The paper's independent analysis showed:
- Single-step supervision: 19% accuracy
- With deep supervision: 39% accuracy (2x improvement)
- Adding hierarchical recursion on top: only 39.0% (negligible)

This means **our current implementation is using the least impactful component** of the TRM/HRM approach. Adding deep supervision should be the highest-priority improvement.

### The 1-Step Gradient Approximation

The paper runs N-1 inner recursion steps under `torch.no_grad()`, then only the final step gets gradients. This:
- Prevents vanishing/exploding gradients through deep recursion
- Reduces memory usage
- Is justified by the Implicit Function Theorem

We currently backpropagate through all 6 recursion steps, which could cause gradient issues as N increases.

---

## 8. Recommended Experiments (Priority Order)

1. **Add deep supervision loop** — N_sup outer steps with detached z, computing loss at each step. Expected: largest single improvement.

2. **Implement 1-step gradient approximation** — run N-1 inner recursions under `no_grad()`, only gradient through the last one. Expected: enables higher N without instability.

3. **Add ACT continue loss** — complete the adaptive computation time mechanism. Expected: better per-input efficiency.

4. **Add EMA weights** — exponential moving average for evaluation. Expected: more stable val_reward.

5. **Reduce to 2-layer MLPs** — match the paper's architecture exactly. Expected: may reduce overfitting.

6. **Try single shared network** — use one MLP for both net_z and net_y. Expected: fewer parameters, potentially better generalization.
