"""TRM Compiler Pass Ordering — Training and Evaluation Example.

Usage:
    python -m trm_compiler.example            # train with heuristic backend
    python -m trm_compiler.example --compilergym  # train with real CompilerGym
    python -m trm_compiler.example --eval     # evaluate only (no training)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .types import BenchmarkSpec
from .model import TinyPassOrderingRefiner, rollout_pass_optimizer
from .data import generate_compiler_traces, CompilerTraceDataset
from .training import train_one_epoch, evaluate_model
from .env_wrapper import NUM_PASSES
from .baselines import random_search, greedy_search, run_optimization_level


def main():
    parser = argparse.ArgumentParser(description="TRM Compiler Pass Ordering")
    parser.add_argument("--compilergym", action="store_true",
                        help="Use real CompilerGym (requires pip install compiler_gym)")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate model only (skip training)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=64,
                        help="Latent dimension")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension")
    parser.add_argument("--n-recursions", type=int, default=6,
                        help="Inner recursion steps (TRM paper: 6)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per benchmark for trace generation")
    parser.add_argument("--max-steps", type=int, default=30,
                        help="Max passes per episode")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full benchmark suite (slower)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="trm_compiler_output",
                        help="Output directory for traces and checkpoints")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # === Generate traces ===
    print(f"\n{'='*60}")
    print("Phase 1: Generating training traces")
    print(f"{'='*60}")

    use_heuristic = not args.compilergym
    traces_path = str(output_dir / "traces.json")

    if Path(traces_path).exists():
        print(f"Loading existing traces from {traces_path}")
        from .data import load_traces
        traces = load_traces(traces_path)
    else:
        print(f"Generating traces (heuristic={use_heuristic})...")
        traces = generate_compiler_traces(
            episodes_per_benchmark=args.episodes,
            max_steps_per_episode=args.max_steps,
            use_heuristic=use_heuristic,
            output_path=traces_path,
            seed=args.seed,
            strategy="mixed",
        )
        print(f"Saved {len(traces)} traces to {traces_path}")

    print(f"Total trace records: {len(traces)}")

    # Create dataset and dataloader
    dataset = CompilerTraceDataset(traces)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # === Initialize model ===
    print(f"\n{'='*60}")
    print("Phase 2: Initializing TRM model")
    print(f"{'='*60}")

    model = TinyPassOrderingRefiner(
        observation_dim=56,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_passes=NUM_PASSES,
        n_recursions=args.n_recursions,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Input: Autophase(56) + Schedule(4) + Feedback(4) + Latent({args.latent_dim})")
    print(f"Output: Pass({NUM_PASSES}) + Feasibility(1) + Value(1) + Halt(1)")

    # === Training ===
    if not args.eval:
        print(f"\n{'='*60}")
        print("Phase 3: Training TRM")
        print(f"{'='*60}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        for epoch in range(args.epochs):
            t0 = time.time()
            losses = train_one_epoch(model, dataloader, optimizer, device)
            scheduler.step()
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"Loss: {losses['total_loss']:.4f} | "
                f"Pass: {losses['pass_loss']:.4f} | "
                f"Entropy: {losses['entropy']:.4f} | "
                f"Value: {losses['value_loss']:.4f} | "
                f"Time: {elapsed:.1f}s | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        # Save model
        ckpt_path = str(output_dir / "trm_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args),
        }, ckpt_path)
        print(f"\nModel saved to {ckpt_path}")

    # === Evaluation ===
    print(f"\n{'='*60}")
    print("Phase 4: Evaluation")
    print(f"{'='*60}")

    if args.benchmark:
        # Full benchmark suite
        from .baselines import run_full_benchmark
        eval_benchmarks = ["qsort", "adpcm", "blowfish", "bzip2", "dijkstra", "sha"]
        run_full_benchmark(model, benchmarks=eval_benchmarks,
                           max_steps=args.max_steps, device=device, seed=args.seed)
    else:
        # Quick evaluation
        from .baselines import random_search, greedy_search, run_optimization_level
        from .env_wrapper import make_compiler_env

        eval_benchmarks = ["qsort", "adpcm", "blowfish"]

        # LLVM levels
        print("\n--- LLVM Optimization Levels ---")
        for level in ["O0", "O2", "Oz"]:
            r = run_optimization_level("qsort", level=level, seed=args.seed)
            print(f"  {level}: reward={r['total_reward']:+.4f}  reduction={r['reduction_pct']*100:.1f}%")

        # Random search
        print("\n--- Random Search (100 trials) ---")
        r = random_search("qsort", max_steps=args.max_steps, num_trials=100, seed=args.seed)
        print(f"  Best: {r['best_reward']:+.4f}  Mean: {r['mean_reward']:+.4f}±{r['std_reward']:.4f}")

        # Greedy search
        print("\n--- Greedy Search ---")
        r = greedy_search("qsort", max_steps=args.max_steps, seed=args.seed)
        print(f"  Reward: {r['total_reward']:+.4f}  Steps: {r['num_steps']}")

        # TRM model
        print("\n--- TRM Model ---")
        model.eval()
        for bench_id in eval_benchmarks:
            env = make_compiler_env(benchmark_id=bench_id, use_compilergym=False)
            obs, _ = env.reset()

            total_reward = 0.0
            for step in range(args.max_steps):
                trace = rollout_pass_optimizer(model, obs, max_steps=1,
                                               temperature=1.0, device=device)
                if trace and trace[0]["pass_id"] >= 0:
                    pass_id = trace[0]["pass_id"]
                    obs, fb, done, info = env.step(pass_id)
                    total_reward += fb.reward
                    if done:
                        break
                else:
                    break

            print(f"  {bench_id:15s} | Reward: {total_reward:+8.4f} | "
                  f"Inst: {env.current_inst_count}/{env.initial_inst_count}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
