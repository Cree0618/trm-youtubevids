#!/usr/bin/env python3
"""TRM Compiler Pass Ordering — Train on real LLVM, benchmark vs LLVM -Oz.

Single self-contained script.  Run on Linux / Google Colab:

    pip install compiler_gym torch numpy
    python trm_compiler_real_llvm.py                  # full pipeline
    python trm_compiler_real_llvm.py --eval-only      # benchmark only (loads checkpoint)
    python trm_compiler_real_llvm.py --synthetic      # no CompilerGym needed
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ═══════════════════════════════════════════════════════════════
# 1. PASS DATABASE
# ═══════════════════════════════════════════════════════════════

USEFUL_PASSES = [
    "mem2reg", "simplifycfg", "early-cse", "instcombine", "reassociate",
    "gvn", "newgvn", "sccp", "dce", "adce",
    "licm", "loop-rotate", "indvars", "loop-unswitch", "loop-deletion",
    "loop-idiom", "loop-unroll", "loop-vectorize", "slp-vectorize", "inline",
    "argpromotion", "deadargelim", "globalopt", "globaldce", "ipconstprop",
    "ipsccp", "prune-eh", "strip-dead-prototypes", "constmerge", "sink",
    "sroa", "tailcallelim", "correlated-propagation", "speculative-execution",
    "jump-threading", "simplifycfg-2", "instcombine-2",
]
NUM_PASSES = len(USEFUL_PASSES)

_PASS_INDEX = {name: i for i, name in enumerate(USEFUL_PASSES)}

COMPILERGYM_FLAG = {
    "mem2reg": "-mem2reg", "simplifycfg": "-simplifycfg",
    "early-cse": "-early-cse", "instcombine": "-instcombine",
    "reassociate": "-reassociate", "gvn": "-gvn", "newgvn": "-newgvn",
    "sccp": "-sccp", "dce": "-dce", "adce": "-adce", "licm": "-licm",
    "loop-rotate": "-loop-rotate", "indvars": "-indvars",
    "loop-unswitch": "-loop-unswitch", "loop-deletion": "-loop-deletion",
    "loop-idiom": "-loop-idiom", "loop-unroll": "-loop-unroll",
    "loop-vectorize": "-loop-vectorize", "slp-vectorize": "-slp-vectorize",
    "inline": "-inline", "argpromotion": "-argpromotion",
    "deadargelim": "-deadargelim", "globalopt": "-globalopt",
    "globaldce": "-globaldce", "ipconstprop": "-ipconstprop",
    "ipsccp": "-ipsccp", "prune-eh": "-prune-eh",
    "strip-dead-prototypes": "-strip-dead-prototypes",
    "constmerge": "-constmerge", "sink": "-sink", "sroa": "-sroa",
    "tailcallelim": "-tailcallelim",
    "correlated-propagation": "-correlated-propagation",
    "speculative-execution": "-speculative-execution",
    "jump-threading": "-jump-threading",
    "simplifycfg-2": "-simplifycfg", "instcombine-2": "-instcombine",
}

OZ_PIPELINE = [
    "mem2reg", "simplifycfg", "early-cse", "instcombine", "reassociate",
    "gvn", "sccp", "dce", "adce", "simplifycfg-2", "instcombine-2",
    "inline", "deadargelim", "globaldce", "ipsccp", "prune-eh",
    "loop-rotate", "licm", "indvars", "loop-deletion", "loop-idiom",
    "sink", "sroa", "correlated-propagation", "jump-threading",
    "strip-dead-prototypes", "constmerge",
]

O3_PIPELINE = OZ_PIPELINE + [
    "loop-unroll", "loop-vectorize", "slp-vectorize",
    "speculative-execution", "loop-unswitch",
]

DEFAULT_BENCHMARKS = [
    "adpcm", "blowfish", "bzip2", "dijkstra", "gsm", "ispell",
    "jpeg-c", "lame", "qsort", "rijndael", "sha", "susan",
    "tiff2bw", "tiff2rgba", "tiffdither", "tiffmedian",
]

# ═══════════════════════════════════════════════════════════════
# 2. COMPILERGYM WRAPPER
# ═══════════════════════════════════════════════════════════════

class CompilerEnv:
    """Thin wrapper: uniform API over CompilerGym and synthetic env."""

    def __init__(self, benchmark_id: str, use_compilergym: bool = False, seed: int = 42):
        self.benchmark_id = benchmark_id
        self.use_compilergym = use_compilergym
        self.seed = seed
        self._cg = None
        self._syn = None
        self._action_map: dict[int, int] = {}
        self._initial_inst = 0
        self._current_inst = 0
        self._applied_passes: list[int] = []
        self._recent_rewards: list[float] = []
        self._cum_reward = 0.0
        self._step = 0
        self._done = False

    def _pass_id_to_name(self, pid: int) -> str:
        return USEFUL_PASSES[pid]

    def open(self):
        if self.use_compilergym:
            import compiler_gym
            full_id = self.benchmark_id if "/" in self.benchmark_id else f"cbench-v1/{self.benchmark_id}"
            self._cg = compiler_gym.make(
                "llvm-v0", benchmark=full_id,
                observation_space="Autophase",
                reward_space="IrInstructionCountOz",
            )
            self._cg.reset()
            action_names = list(self._cg.action_space.names)
            self._action_map = {}
            for pid, pname in enumerate(USEFUL_PASSES):
                flag = COMPILERGYM_FLAG.get(pname, f"-{pname}")
                if flag in action_names:
                    self._action_map[pid] = action_names.index(flag)

    def close(self):
        if self._cg is not None:
            self._cg.close()
            self._cg = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *a):
        self.close()

    def reset(self) -> tuple[np.ndarray, int]:
        self._applied_passes = []
        self._recent_rewards = []
        self._cum_reward = 0.0
        self._step = 0
        self._done = False

        if self._cg is not None:
            self._cg.reset()
            self._initial_inst = int(self._cg.observation["IrInstructionCount"])
            self._current_inst = self._initial_inst
            obs = np.array(self._cg.observation["Autophase"], dtype=np.float32)
        else:
            obs = self._synthetic_reset()
        return obs, self._initial_inst

    def step(self, pass_id: int) -> tuple[np.ndarray, list[float], bool, dict]:
        if self._done:
            raise RuntimeError("Episode done, call reset()")
        pname = self._pass_id_to_name(pass_id)
        prev_inst = self._current_inst

        if self._cg is not None:
            if pass_id not in self._action_map:
                return self._compilergym_obs(), [0.0, 1.0, 0.0, 0.0], True, {"err": "unmapped"}
            try:
                _, _, cg_done, _ = self._cg.step(self._action_map[pass_id])
                self._current_inst = int(self._cg.observation["IrInstructionCount"])
                compiled = True
                obs = np.array(self._cg.observation["Autophase"], dtype=np.float32)
                if cg_done:
                    self._done = True
            except Exception:
                compiled = False
                self._done = True
                obs = self._compilergym_obs()
        else:
            obs = self._synthetic_step(pass_id, pname)
            compiled = True

        if prev_inst > 0 and self._current_inst > 0:
            reward = math.log(prev_inst / max(self._current_inst, 1))
        else:
            reward = 0.0

        self._applied_passes.append(pass_id)
        self._recent_rewards.append(reward)
        self._cum_reward += reward
        self._step += 1

        feedback = [
            float(self._current_inst) / 10000.0,
            self._current_inst / max(prev_inst, 1),
            float(compiled),
            reward,
        ]
        info = {"pass_name": pname, "current_inst": self._current_inst,
                "initial_inst": self._initial_inst, "step": self._step}
        return obs, feedback, self._done, info

    @property
    def current_inst_count(self):
        return self._current_inst

    @property
    def initial_inst_count(self):
        return self._initial_inst

    # ---- synthetic helpers ----

    _PASS_EFF = {
        "mem2reg": (0.02, 0.15), "simplifycfg": (0.01, 0.10),
        "early-cse": (0.01, 0.08), "instcombine": (0.05, 0.25),
        "reassociate": (0.01, 0.06), "gvn": (0.03, 0.20),
        "newgvn": (0.03, 0.18), "sccp": (0.01, 0.10),
        "dce": (0.01, 0.08), "adce": (0.01, 0.06),
        "licm": (0.02, 0.12), "loop-rotate": (0.00, 0.05),
        "indvars": (0.00, 0.04), "loop-unswitch": (0.00, 0.06),
        "loop-deletion": (0.00, 0.08), "loop-idiom": (0.00, 0.06),
        "loop-unroll": (0.02, 0.15), "loop-vectorize": (0.03, 0.20),
        "slp-vectorize": (0.02, 0.12), "inline": (0.02, 0.18),
        "argpromotion": (0.00, 0.04), "deadargelim": (0.00, 0.03),
        "globalopt": (0.01, 0.08), "globaldce": (0.00, 0.05),
        "ipconstprop": (0.00, 0.04), "ipsccp": (0.00, 0.05),
        "prune-eh": (0.00, 0.03), "strip-dead-prototypes": (0.00, 0.02),
        "constmerge": (0.00, 0.02), "sink": (0.00, 0.05),
        "sroa": (0.02, 0.12), "tailcallelim": (0.00, 0.04),
        "correlated-propagation": (0.00, 0.04),
        "speculative-execution": (0.00, 0.03),
        "jump-threading": (0.01, 0.06), "simplifycfg-2": (0.00, 0.05),
        "instcombine-2": (0.00, 0.08),
    }
    _SYNERGIES = {
        ("instcombine", "gvn"): 1.4, ("mem2reg", "instcombine"): 1.3,
        ("loop-rotate", "licm"): 1.4, ("loop-rotate", "loop-unroll"): 1.3,
        ("simplifycfg", "instcombine"): 1.3, ("mem2reg", "sroa"): 1.3,
        ("reassociate", "instcombine"): 1.3, ("reassociate", "gvn"): 1.2,
        ("gvn", "dce"): 1.3, ("inline", "instcombine"): 1.4,
        ("loop-unroll", "loop-vectorize"): 1.5, ("loop-unroll", "slp-vectorize"): 1.3,
        ("early-cse", "instcombine"): 1.2, ("gvn", "licm"): 1.2,
    }

    def _synthetic_reset(self):
        base = hash(self.benchmark_id) % 2000 + 500
        self._initial_inst = base
        self._current_inst = base
        self._syn_rng = np.random.RandomState(self.seed)
        return np.random.RandomState(self.seed).rand(56).astype(np.float32) * 5

    def _synthetic_step(self, pass_id, pname):
        lo, hi = self._PASS_EFF.get(pname, (0.0, 0.05))
        eff = self._syn_rng.uniform(lo, hi)
        if self._applied_passes:
            last = self._pass_id_to_name(self._applied_passes[-1])
            eff *= self._SYNERGIES.get((last, pname), 1.0)
        times = self._applied_passes.count(pass_id)
        if times > 0:
            eff *= 0.5 ** times
        reduction = int(self._current_inst * eff)
        self._current_inst = max(self._current_inst - reduction, 1)
        return np.random.RandomState(self._step + self.seed).rand(56).astype(np.float32) * 5

    def _compilergym_obs(self):
        try:
            return np.array(self._cg.observation["Autophase"], dtype=np.float32)
        except Exception:
            return np.zeros(56, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# 3. TRM MODEL
# ═══════════════════════════════════════════════════════════════

class TRMPassOrdering(nn.Module):
    """Tiny Recursive Model for compiler pass ordering (~60K params).

    Dual-network architecture per TRM paper:
      net_z(x,y,z) → z   (inner recursion n times)
      net_y(x,z)   → y   (answer refinement)
    Multi-head output: pass logits, feasibility, value, halt.
    """

    def __init__(self, obs_dim=56, sched_dim=4, fb_dim=4,
                 latent_dim=64, hidden_dim=128,
                 num_passes=NUM_PASSES, n_recursion=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_recursion = n_recursion
        x_dim = obs_dim + sched_dim + fb_dim
        self.net_z = nn.Sequential(
            nn.Linear(x_dim + 2 * latent_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim))
        self.net_y = nn.Sequential(
            nn.Linear(x_dim + latent_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim))
        self.head_pass = nn.Linear(latent_dim, num_passes)
        self.head_feas = nn.Linear(latent_dim, 1)
        self.head_val = nn.Linear(latent_dim, 1)
        self.head_halt = nn.Linear(latent_dim, 1)

    def init_latents(self, bs=1, device="cpu"):
        return (torch.zeros(bs, self.latent_dim, device=device),
                torch.zeros(bs, self.latent_dim, device=device))

    def forward(self, obs, sched, fb, y, z):
        x = torch.cat([obs, sched, fb], -1)
        for _ in range(self.n_recursion):
            z = self.net_z(torch.cat([x, y, z], -1))
        y = self.net_y(torch.cat([x, z], -1))
        return {
            "y": y, "z": z,
            "pass_logits": self.head_pass(z),
            "feasibility": torch.sigmoid(self.head_feas(z).squeeze(-1)),
            "value": self.head_val(z).squeeze(-1),
            "halt_logit": self.head_halt(z).squeeze(-1),
        }


def _encode_schedule(step, max_steps, applied):
    return [
        step / max(max_steps, 1),
        len(applied) / max(max_steps, 1),
        (applied[-1] / NUM_PASSES) if applied else 0.0,
        (sum(applied[-3:]) / (3 * NUM_PASSES)) if applied else 0.0,
    ]


@torch.no_grad()
def rollout_blind(model, obs_np, max_steps=30, temperature=1.0, device="cpu"):
    """Single forward pass that picks an entire pass sequence."""
    y, z = model.init_latents(1, device)
    obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    fb_t = torch.zeros(1, 4, device=device)
    trace, applied = [], []
    for step in range(max_steps):
        sch_t = torch.tensor(_encode_schedule(step, max_steps, applied),
                             dtype=torch.float32, device=device).unsqueeze(0)
        out = model(obs_t, sch_t, fb_t, y, z)
        y, z = out["y"], out["z"]
        halt = torch.sigmoid(out["halt_logit"]).item()
        if halt > 0.5 and step >= 5:
            trace.append({"step": step, "pass_id": -1})
            break
        logits = out["pass_logits"].squeeze(0) / temperature
        if applied:
            for p in applied[-3:]:
                logits[p] -= 2.0
        probs = F.softmax(logits, -1).clamp(min=1e-6)
        probs /= probs.sum(-1, keepdim=True)
        pid = torch.multinomial(probs, 1).item()
        applied.append(pid)
        trace.append({"step": step, "pass_id": pid})
    return trace


@torch.no_grad()
def rollout_closed_loop(model, env, max_steps=30, temperature=1.0, device="cpu"):
    """Closed-loop: real CompilerGym feedback injected each step."""
    y, z = model.init_latents(1, device)
    obs, _ = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    fb_t = torch.zeros(1, 4, device=device)
    trace, applied, total_r = [], [], 0.0
    for step in range(max_steps):
        sch_t = torch.tensor(_encode_schedule(step, max_steps, applied),
                             dtype=torch.float32, device=device).unsqueeze(0)
        out = model(obs_t, sch_t, fb_t, y, z)
        y, z = out["y"], out["z"]
        halt = torch.sigmoid(out["halt_logit"]).item()
        if halt > 0.5 and step >= 5:
            trace.append({"step": step, "pass_id": -1, "total_reward": total_r,
                          "inst_count": env.current_inst_count})
            break
        logits = out["pass_logits"].squeeze(0) / temperature
        if applied:
            for p in applied[-3:]:
                logits[p] -= 2.0
        probs = F.softmax(logits, -1).clamp(min=1e-6)
        probs /= probs.sum(-1, keepdim=True)
        pid = torch.multinomial(probs, 1).item()
        applied.append(pid)
        next_obs, fb, done, info = env.step(pid)
        r = fb[3]
        total_r += r
        trace.append({"step": step, "pass_id": pid, "pass_name": info.get("pass_name", ""),
                       "reward": r, "total_reward": total_r,
                       "inst_count": env.current_inst_count})
        obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        fb_t = torch.tensor(fb, dtype=torch.float32, device=device).unsqueeze(0)
        if done:
            break
    return trace


# ═══════════════════════════════════════════════════════════════
# 4. TRAINING DATA
# ═══════════════════════════════════════════════════════════════

class TraceDataset(Dataset):
    def __init__(self, records):
        self.r = records

    def __len__(self):
        return len(self.r)

    def __getitem__(self, idx):
        rec = self.r[idx]
        obs = torch.tensor(rec["obs"][:56], dtype=torch.float32)
        if len(rec["obs"]) < 56:
            obs = F.pad(obs, (0, 56 - len(rec["obs"])))
        sch = torch.tensor(_encode_schedule(rec["step"], 30, rec["applied"]), dtype=torch.float32)
        fb = torch.tensor(rec["feedback"], dtype=torch.float32)
        pid = torch.tensor(rec["pass_id"], dtype=torch.long)
        rew = torch.tensor(rec["reward"], dtype=torch.float32)
        return {"observation": obs, "schedule": sch, "feedback": fb,
                "pass_id": pid, "reward": rew}


def generate_traces(benchmarks, episodes, max_steps, use_cg, seed=42):
    """Generate training traces from CompilerGym (or synthetic fallback)."""
    records = []
    rng = random.Random(seed)
    for bench in benchmarks:
        for ep in range(episodes):
            env = CompilerEnv(bench, use_compilergym=use_cg, seed=seed + ep)
            try:
                env.open()
                obs, init_inst = env.reset()
                applied, cum_r = [], 0.0
                for step in range(max_steps):
                    # mixed strategy: 50% greedy+noise, 50% random
                    if ep % 2 == 1:
                        best_pid, best_r = rng.randint(0, NUM_PASSES - 1), float("-inf")
                        for _ in range(min(8, NUM_PASSES)):
                            cand = rng.randint(0, NUM_PASSES - 1)
                            te = CompilerEnv(bench, use_compilergym=False, seed=seed + step + cand)
                            te.open()
                            te.reset()
                            for p in applied:
                                te.step(p)
                            _, fbc, _, _ = te.step(cand)
                            te.close()
                            if fbc[3] > best_r:
                                best_r, best_pid = fbc[3], cand
                        if rng.random() < 0.2:
                            pid = rng.randint(0, NUM_PASSES - 1)
                        else:
                            pid = best_pid
                    else:
                        pid = rng.randint(0, NUM_PASSES - 1)

                    next_obs, fb, done, info = env.step(pid)
                    records.append({
                        "obs": obs.tolist(),
                        "step": step,
                        "applied": list(applied),
                        "pass_id": pid,
                        "feedback": fb,
                        "reward": fb[3],
                    })
                    cum_r += fb[3]
                    applied.append(pid)
                    obs = next_obs
                    if done:
                        break
            finally:
                env.close()
    return records


# ═══════════════════════════════════════════════════════════════
# 5. TRAINING
# ═══════════════════════════════════════════════════════════════

def train_step(model, batch, device, entropy_coef=0.05):
    obs = batch["observation"].to(device)
    sch = batch["schedule"].to(device)
    fb = batch["feedback"].to(device)
    target = batch["pass_id"].to(device)
    rew = batch["reward"].to(device)
    bs = obs.shape[0]
    y, z = model.init_latents(bs, device)
    out = model(obs, sch, fb, y, z)

    pass_loss = F.cross_entropy(out["pass_logits"], target)
    probs = F.softmax(out["pass_logits"], -1)
    logp = F.log_softmax(out["pass_logits"], -1)
    entropy = -(probs * logp).sum(-1).mean()
    ent_loss = -entropy_coef * entropy

    comp_tgt = (rew > -2.0).float()
    feas_loss = F.binary_cross_entropy(out["feasibility"], comp_tgt)
    val_loss = F.mse_loss(out["value"], rew)
    halt_tgt = (rew < -1.0).float()
    halt_loss = F.binary_cross_entropy_with_logits(out["halt_logit"], halt_tgt)

    total = pass_loss + ent_loss + 0.5 * feas_loss + 0.3 * val_loss + 0.2 * halt_loss
    return total, {"pass": pass_loss.item(), "entropy": entropy.item(),
                   "feas": feas_loss.item(), "val": val_loss.item(),
                   "halt": halt_loss.item(), "total": total.item()}


def train_model(model, records, epochs, batch_size, lr, device):
    ds = TraceDataset(records)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        t0 = time.time()
        model.train()
        tot = 0.0
        nb = 0
        info = {"pass": 0, "entropy": 0}
        for batch in dl:
            opt.zero_grad()
            loss, info = train_step(model, batch, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += info["total"]
            nb += 1
        sched.step()
        print(f"  epoch {ep+1:3d}/{epochs}  loss={tot/max(nb,1):.4f}  "
              f"pass={info['pass']:.4f}  entropy={info['entropy']:.4f}  "
              f"{time.time()-t0:.1f}s  lr={sched.get_last_lr()[0]:.2e}")


# ═══════════════════════════════════════════════════════════════
# 6. BENCHMARKING
# ═══════════════════════════════════════════════════════════════

def _run_fixed_pipeline(env, pass_names, max_steps):
    env.reset()
    total_r = 0.0
    for pname in pass_names[:max_steps]:
        if pname not in _PASS_INDEX:
            continue
        try:
            _, fb, done, _ = env.step(_PASS_INDEX[pname])
        except Exception:
            break
        total_r += fb[3]
        if done:
            break
    return total_r, env.current_inst_count, env.initial_inst_count


def _run_random(env, max_steps, num_trials, seed):
    best_r, best_inst = float("-inf"), 0
    rng = random.Random(seed)
    for trial in range(num_trials):
        env.reset()
        tr = 0.0
        for _ in range(max_steps):
            pid = rng.randint(0, NUM_PASSES - 1)
            try:
                _, fb, done, _ = env.step(pid)
            except Exception:
                break
            tr += fb[3]
            if done:
                break
        if tr > best_r:
            best_r = tr
            best_inst = env.current_inst_count
    return best_r, best_inst, env.initial_inst_count


def _run_trm(model, env, max_steps, device, closed_loop=True, temperature=0.8):
    if closed_loop:
        trace = rollout_closed_loop(model, env, max_steps, temperature, device)
        if not trace:
            return 0.0, env.current_inst_count, env.initial_inst_count
        last = trace[-1]
        return last["total_reward"], last["inst_count"], env.initial_inst_count
    else:
        env.reset()
        obs = env._compilergym_obs() if env._cg else np.zeros(56, dtype=np.float32)
        if env._cg:
            obs = np.array(env._cg.observation["Autophase"], dtype=np.float32)
        trace = rollout_blind(model, obs, max_steps, temperature, device)
        total_r = 0.0
        for step in trace:
            if step["pass_id"] < 0:
                break
            try:
                _, fb, done, _ = env.step(step["pass_id"])
            except Exception:
                break
            total_r += fb[3]
            if done:
                break
        return total_r, env.current_inst_count, env.initial_inst_count


def bench_single(bench_id, model, max_steps, device, use_cg, seed, num_random):
    """Run all algorithms on one benchmark. Returns dict of results."""
    results = {}

    # LLVM -Oz
    env = CompilerEnv(bench_id, use_compilergym=use_cg, seed=seed)
    try:
        env.open()
        r, fi, ii = _run_fixed_pipeline(env, OZ_PIPELINE, max_steps)
        results["LLVM-Oz"] = {"reward": r, "final": fi, "initial": ii,
                               "reduction": 1 - fi / max(ii, 1)}
    finally:
        env.close()

    # LLVM -O3
    env = CompilerEnv(bench_id, use_compilergym=use_cg, seed=seed)
    try:
        env.open()
        r, fi, ii = _run_fixed_pipeline(env, O3_PIPELINE, max_steps)
        results["LLVM-O3"] = {"reward": r, "final": fi, "initial": ii,
                               "reduction": 1 - fi / max(ii, 1)}
    finally:
        env.close()

    # Random search
    env = CompilerEnv(bench_id, use_compilergym=use_cg, seed=seed)
    try:
        env.open()
        r, fi, ii = _run_random(env, max_steps, num_random, seed)
        results[f"Random({num_random})"] = {"reward": r, "final": fi, "initial": ii,
                                             "reduction": 1 - fi / max(ii, 1)}
    finally:
        env.close()

    if model is not None:
        model.eval()
        # TRM closed-loop
        env = CompilerEnv(bench_id, use_compilergym=use_cg, seed=seed)
        try:
            env.open()
            r, fi, ii = _run_trm(model, env, max_steps, device, closed_loop=True)
            results["TRM-loop"] = {"reward": r, "final": fi, "initial": ii,
                                    "reduction": 1 - fi / max(ii, 1)}
        finally:
            env.close()

        # TRM blind
        env = CompilerEnv(bench_id, use_compilergym=use_cg, seed=seed)
        try:
            env.open()
            r, fi, ii = _run_trm(model, env, max_steps, device, closed_loop=False)
            results["TRM-blind"] = {"reward": r, "final": fi, "initial": ii,
                                     "reduction": 1 - fi / max(ii, 1)}
        finally:
            env.close()

    return results


def print_table(all_results, benchmarks):
    algos = set()
    for br in all_results.values():
        algos.update(br.keys())
    algos = sorted(algos)

    # Reward
    print(f"\n{'Algorithm':<14s}", end="")
    for b in benchmarks:
        print(f" {b:>10s}", end="")
    print(f" {'Mean':>10s}")
    print("-" * (14 + 11 * (len(benchmarks) + 1)))

    for algo in algos:
        print(f"{algo:<14s}", end="")
        vals = []
        for b in benchmarks:
            if b in all_results and algo in all_results[b]:
                v = all_results[b][algo]["reward"]
                vals.append(v)
                print(f" {v:+10.4f}", end="")
            else:
                print(f" {'—':>10s}", end="")
        print(f" {np.mean(vals):+10.4f}" if vals else "")

    # Reduction
    print(f"\n{'Algorithm':<14s}", end="")
    for b in benchmarks:
        print(f" {b:>10s}", end="")
    print(f" {'Mean':>10s}")
    print("-" * (14 + 11 * (len(benchmarks) + 1)))

    for algo in algos:
        print(f"{algo:<14s}", end="")
        vals = []
        for b in benchmarks:
            if b in all_results and algo in all_results[b]:
                v = all_results[b][algo]["reduction"] * 100
                vals.append(v)
                print(f" {v:9.1f}%", end="")
            else:
                print(f" {'—':>10s}", end="")
        print(f" {np.mean(vals):9.1f}%" if vals else "")


# ═══════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TRM Compiler Pass Ordering — Train & benchmark on real LLVM")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=20,
                        help="Trace episodes per benchmark")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-recursion", type=int, default=6)
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["qsort", "adpcm", "blowfish", "bzip2"])
    parser.add_argument("--num-random", type=int, default=100,
                        help="Random search budget")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="trm_output")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic env (no CompilerGym)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, just benchmark existing checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "trm_model.pt")

    use_cg = not args.synthetic
    if use_cg:
        try:
            import compiler_gym
            print(f"[ok] compiler_gym {compiler_gym.__version__}")
        except ImportError as e:
            print(f"[warn] compiler_gym import error: {e}")
            print("[warn] falling back to --synthetic")
            use_cg = False
        except Exception as e:
            print(f"[warn] compiler_gym error: {e}")
            print("[warn] falling back to --synthetic")
            use_cg = False

    print(f"Device: {device}  |  CompilerGym: {use_cg}  |  "
          f"Benchmarks: {args.benchmarks}")

    # ---- Phase 1: Generate traces ----
    if not args.eval_only:
        print(f"\n{'='*60}")
        print("Phase 1: Generating training traces")
        print(f"{'='*60}")
        t0 = time.time()
        records = generate_traces(
            args.benchmarks, args.episodes, args.max_steps,
            use_cg=use_cg, seed=args.seed,
        )
        print(f"  {len(records)} records from {len(args.benchmarks)} benchmarks "
              f"in {time.time()-t0:.1f}s")

    # ---- Phase 2: Create model ----
    print(f"\n{'='*60}")
    print("Phase 2: TRM model")
    print(f"{'='*60}")
    model = TRMPassOrdering(n_recursion=args.n_recursion).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ---- Phase 3: Train ----
    if not args.eval_only:
        print(f"\n{'='*60}")
        print("Phase 3: Training")
        print(f"{'='*60}")
        train_model(model, records, args.epochs, args.batch_size, args.lr, device)
        torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)
        print(f"  Saved checkpoint to {ckpt_path}")
    else:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded checkpoint from {ckpt_path}")
        else:
            print(f"  [warn] No checkpoint at {ckpt_path}, using random weights")

    # ---- Phase 4: Benchmark ----
    print(f"\n{'='*60}")
    print("Phase 4: Benchmarking")
    print(f"{'='*60}")

    all_results = {}
    for bench_id in args.benchmarks:
        print(f"\n  --- {bench_id} ---")
        r = bench_single(bench_id, model, args.max_steps, device,
                         use_cg=use_cg, seed=args.seed, num_random=args.num_random)
        for algo, v in r.items():
            print(f"    {algo:<14s}  reward={v['reward']:+8.4f}  "
                  f"reduction={v['reduction']*100:5.1f}%  final_inst={v['final']}")
        all_results[bench_id] = r

    print_table(all_results, args.benchmarks)
    print()


if __name__ == "__main__":
    main()
