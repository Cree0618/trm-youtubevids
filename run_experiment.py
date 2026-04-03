"""TRM Autoresearch — Experiment runner.

This file is FIXED and should NOT be modified by agents.

Orchestrates the full experiment cycle:
1. Load training data (runs prepare.py if needed)
2. Import model/training from train.py
3. Train for a fixed time budget
4. Evaluate on held-out benchmarks
5. Log results to experiments/results.jsonl
6. Compare against best result so far
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from prepare import EVAL_BENCHMARKS

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

TIME_BUDGET = 600  # 10 minutes
BATCH_SIZE = 64
SEED = 42
EXPERIMENTS_DIR = Path("experiments")
RESULTS_FILE = EXPERIMENTS_DIR / "results.jsonl"
BEST_MODEL_DIR = EXPERIMENTS_DIR / "best_model"
DATA_DIR = Path("data")
TRACES_FILE = DATA_DIR / "traces.json"

# ──────────────────────────────────────────────────────────────
# Ensure data exists
# ──────────────────────────────────────────────────────────────

def ensure_data():
    """Generate training data if it doesn't exist."""
    if not TRACES_FILE.exists():
        print("No training data found. Generating...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        from prepare import generate_traces
        generate_traces(output_path=str(TRACES_FILE))
        print(f"Generated traces saved to {TRACES_FILE}")


def load_traces():
    """Load training traces from JSON."""
    from prepare import CompilerTraceRecord
    with open(TRACES_FILE) as f:
        data = json.load(f)
    return [CompilerTraceRecord.from_dict(d) for d in data]


# ──────────────────────────────────────────────────────────────
# Experiment management
# ──────────────────────────────────────────────────────────────

def get_experiment_id():
    """Get next experiment ID."""
    if not RESULTS_FILE.exists():
        return 1
    with open(RESULTS_FILE) as f:
        lines = [l for l in f.readlines() if l.strip()]
    if not lines:
        return 1
    last = json.loads(lines[-1])
    return last.get("experiment_id", 0) + 1


def get_best_result():
    """Get the best val_reward seen so far."""
    if not RESULTS_FILE.exists():
        return None
    best = None
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                if best is None or result.get("val_reward", float("-inf")) > best.get("val_reward", float("-inf")):
                    best = result
            except json.JSONDecodeError:
                continue
    return best


def save_result(result):
    """Append experiment result to JSONL file."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


def save_best_model(train_module, model):
    """Save the best model and its train.py."""
    if BEST_MODEL_DIR.exists():
        shutil.rmtree(BEST_MODEL_DIR)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "latent_dim": model.LATENT_DIM,
            "hidden_dim": model.HIDDEN_DIM,
            "n_recursions": model.N_RECURSIONS,
        },
    }, str(BEST_MODEL_DIR / "model.pt"))

    # Save the train.py that produced this model
    train_path = Path("train.py")
    if train_path.exists():
        shutil.copy2(str(train_path), str(BEST_MODEL_DIR / "train.py"))


def load_best_model(device):
    """Load the best model from disk."""
    model_path = BEST_MODEL_DIR / "model.pt"
    if not model_path.exists():
        return None

    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    from train import TRMPassOrdering
    model = TRMPassOrdering()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


# ──────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────

def run_experiment():
    """Run a single experiment."""
    exp_id = get_experiment_id()
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"\n{'='*60}")
    print(f"Experiment #{exp_id}")
    print(f"{'='*60}")

    # Reload train.py module to pick up any changes
    if "train" in sys.modules:
        importlib.reload(sys.modules["train"])
    import train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading training data from {TRACES_FILE}...")
    traces = load_traces()
    print(f"  {len(traces)} trace records")

    from prepare import TraceDataset
    dataset = TraceDataset(traces)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # Create model
    print(f"\nCreating model...")
    model = train.TRMPassOrdering()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Latent dim: {model.LATENT_DIM}")
    print(f"  Hidden dim: {model.HIDDEN_DIM}")
    print(f"  Recursions: {model.N_RECURSIONS}")

    # Train
    print(f"\nTraining (time budget: {TIME_BUDGET}s)...")
    t0 = time.time()
    try:
        train_result = train.train(
            model,
            dataloader,
            device,
            time_budget=TIME_BUDGET,
        )
    except Exception as e:
        print(f"\nTraining failed: {e}")
        traceback.print_exc()
        result = {
            "experiment_id": exp_id,
            "timestamp": timestamp,
            "status": "failed",
            "error": str(e),
            "config": {
                "latent_dim": model.LATENT_DIM,
                "hidden_dim": model.HIDDEN_DIM,
                "n_recursions": model.N_RECURSIONS,
                "batch_size": BATCH_SIZE,
            },
        }
        save_result(result)
        return result

    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time:.1f}s")
    print(f"  Epochs: {train_result['epochs']}")
    print(f"  Steps: {train_result['steps']}")
    print(f"  Final loss: {train_result['final_loss']:.4f}")

    # Evaluate
    print(f"\nEvaluating on {EVAL_BENCHMARKS}...")
    model.eval()
    eval_result = train.evaluate(model, device=device)

    print(f"  val_reward: {eval_result['val_reward']:+.4f}")
    print(f"  val_reward_std: {eval_result['val_reward_std']:.4f}")
    print(f"  val_reward_min: {eval_result['val_reward_min']:+.4f}")
    print(f"  val_reward_max: {eval_result['val_reward_max']:+.4f}")

    # Compare with best
    best = get_best_result()
    is_new_best = best is None or eval_result["val_reward"] > best.get("val_reward", float("-inf"))

    if is_new_best:
        print(f"\n*** NEW BEST! *** (previous: {best.get('val_reward', 'N/A')})")
        save_best_model(train, model)
    else:
        print(f"\nNo improvement over best (best: {best.get('val_reward', 'N/A'):+.4f})")

    # Save result
    result = {
        "experiment_id": exp_id,
        "timestamp": timestamp,
        "status": "success",
        "val_reward": eval_result["val_reward"],
        "val_reward_std": eval_result["val_reward_std"],
        "val_reward_min": eval_result["val_reward_min"],
        "val_reward_max": eval_result["val_reward_max"],
        "final_loss": train_result["final_loss"],
        "train_epochs": train_result["epochs"],
        "train_steps": train_result["steps"],
        "train_time": train_result["total_time"],
        "is_new_best": is_new_best,
        "config": {
            "latent_dim": model.LATENT_DIM,
            "hidden_dim": model.HIDDEN_DIM,
            "n_recursions": model.N_RECURSIONS,
            "batch_size": BATCH_SIZE,
            "n_params": n_params,
        },
    }
    save_result(result)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Experiment #{exp_id} complete")
    print(f"{'='*60}")
    print(f"  val_reward: {eval_result['val_reward']:+.4f}")
    print(f"  New best: {is_new_best}")
    print(f"  Result logged to {RESULTS_FILE}")

    return result


def show_history():
    """Show experiment history."""
    if not RESULTS_FILE.exists():
        print("No experiments yet.")
        return

    print(f"\n{'='*60}")
    print("Experiment History")
    print(f"{'='*60}")
    print(f"{'ID':>4s} | {'val_reward':>12s} | {'loss':>8s} | {'epochs':>6s} | {'time':>6s} | {'best':>4s}")
    print("-" * 52)

    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                status = r.get("status", "unknown")
                if status == "failed":
                    print(f"{r['experiment_id']:4d} | {'FAILED':>12s} | {'—':>8s} | {'—':>6s} | {'—':>6s} | {'':>4s}")
                    continue

                vr = r.get("val_reward", 0)
                loss = r.get("final_loss", 0)
                epochs = r.get("train_epochs", 0)
                t = r.get("train_time", 0)
                is_best = "*" if r.get("is_new_best", False) else ""
                print(f"{r['experiment_id']:4d} | {vr:+12.4f} | {loss:8.4f} | {epochs:6d} | {t:5.0f}s | {is_best:>4s}")
            except (json.JSONDecodeError, KeyError):
                continue


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--history":
        show_history()
        return

    ensure_data()
    run_experiment()


if __name__ == "__main__":
    main()
