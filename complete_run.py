#!/usr/bin/env python3
"""Complete TRM training pipeline for Google Colab.

Usage (in Colab cell):
    %run complete_run.py

This script runs the entire pipeline in Colab:
1. Clones repo (if needed)
2. Installs dependencies (torch, numpy, compiler_gym)
3. Trains TRM on synthetic LLVM
4. Benchmarks against real CompilerGym (LLVM -Oz, -O3, Random, TRM)
"""
import subprocess
import sys
import os
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = "/content/trm-youtubevids"


def setup_environment():
    """Install all dependencies in Colab environment."""
    print("=" * 60)
    print("STEP 1: Installing dependencies")
    print("=" * 60)
    
    # Fresh install: uninstall existing numpy first
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "numpy", "torch", "-y", "-q"], capture_output=True)
    
    # Install in correct order: numpy first (compatible version)
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy>=1.26.0,<2.0", "-q"], check=True, capture_output=True)
    
    # Then torch
    subprocess.run([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu", "-q"], check=True, capture_output=True)
    
    # Then compiler_gym (it will install its own deps but we pin numpy)
    subprocess.run([sys.executable, "-m", "pip", "install", "compiler_gym", "-q"], check=True, capture_output=True)
    
    # Now pin numpy again to ensure we have the right version
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy>=1.26.0,<2.0", "-q"], capture_output=True)
    
    print("Dependencies installed!")


def clone_repo():
    """Clone or update the repo."""
    print("=" * 60)
    print("STEP 2: Setting up project")
    print("=" * 60)
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["COMPILER_GYM_HOME"] = "/content/compiler_gym"
    
    if not os.path.exists(PROJECT_DIR):
        print("Cloning repo...")
        subprocess.run(["git", "clone", "https://github.com/Cree0618/trm-youtubevids.git", PROJECT_DIR], check=True, capture_output=True)
    else:
        print("Pulling latest...")
        subprocess.run(["git", "-C", PROJECT_DIR, "pull"], check=True, capture_output=True)


def verify_compiler_gym():
    """Verify CompilerGym works."""
    print("=" * 60)
    print("STEP 3: Verifying CompilerGym")
    print("=" * 60)
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["COMPILER_GYM_HOME"] = "/content/compiler_gym"
    
    import compiler_gym
    env = compiler_gym.make("llvm-v0", benchmark="cbench-v1/qsort",
        observation_space="Autophase", reward_space="IrInstructionCountOz")
    obs = env.reset()
    ap = obs["Autophase"]
    ic = obs["IrInstructionCount"]
    print(f"Autophase: {len(ap)} features, Initial inst: {ic}")
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        inst = obs["IrInstructionCount"]
        print(f"  Step {i}: inst={inst} reward={reward:.2f} done={done}")
    env.close()
    print("CompilerGym OK!")


def train_and_benchmark():
    """Train TRM and benchmark with real CompilerGym."""
    print("=" * 60)
    print("STEP 4: Training TRM + Benchmarking with REAL LLVM")
    print("=" * 60)
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["COMPILER_GYM_HOME"] = "/content/compiler_gym"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Train on synthetic, benchmark with real CompilerGym
    sys.argv = [
        "trm_compiler_real_llvm.py",
        "--synthetic",
        "--epochs", "5",
        "--episodes", "5",
        "--benchmarks", "qsort", "adpcm",
        "--max-steps", "20",
        "--batch-size", "64",
        "--num-random", "20",
        "--seed", "42"
    ]
    
    sys.path.insert(0, PROJECT_DIR)
    
    from trm_compiler_real_llvm import main as train_main
    train_main()


def main():
    clone_repo()
    setup_environment()
    verify_compiler_gym()
    train_and_benchmark()
    
    print("=" * 60)
    print("DONE! TRM trained and benchmarked on REAL LLVM.")
    print("=" * 60)


if __name__ == "__main__":
    main()