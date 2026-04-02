#!/usr/bin/env python3
"""Complete TRM training pipeline: setup env + train on real LLVM + benchmark.

Usage (in Colab):
    !python complete_run.py

This script:
1. Creates Python 3.11 venv if needed
2. Installs all dependencies (torch, numpy<2, compiler_gym deps)
3. Verifies CompilerGym works
4. Trains TRM model on real LLVM
5. Benchmarks against LLVM -Oz, -O3, Random, TRM
"""
import subprocess
import sys
import os
import time

VENV_DIR = "/content/trm-env"
PROJECT_DIR = "/content/trm-youtubevids"
VENV_PIP = f"{VENV_DIR}/bin/pip"
VENV_PYTHON = f"{VENV_DIR}/bin/python"


def run(cmd, check=True, capture=True):
    """Run a command."""
    result = subprocess.run(cmd, shell=isinstance(cmd, str), check=check, capture_output=capture, text=True)
    if result and result.stdout:
        print(result.stdout.strip())
    return result


def step1_setup():
    """Step 1: Setup environment."""
    print("=" * 60)
    print("STEP 1: Setting up Python 3.11 venv with all dependencies")
    print("=" * 60)
    
    # Create venv if needed
    if not os.path.exists(VENV_DIR):
        print("Creating venv with Python 3.11...")
        run(f"python3.11 -m venv {VENV_DIR}")
    else:
        print(f"Venv already exists: {VENV_DIR}")
    
    # Always upgrade pip
    print("Upgrading pip...")
    run(f"{VENV_PIP} install --upgrade pip setuptools wheel -q")
    
    # Always reinstall numpy<2 and torch (venv might be corrupted)
    print("Installing torch + numpy<2...")
    run(f"{VENV_PIP} install 'numpy<2.0' torch --index-url https://download.pytorch.org/whl/cpu -q")
    
    # Install all compiler_gym dependencies one by one (continue on error)
    print("Installing compiler_gym dependencies...")
    deps = [
        "grpcio", "pydantic", "protobuf==3.20.3", "requests", "docker",
        "fasteners", "absl-py", "deprecated", "tabulate", "gym==0.21.0",
        "humanize", "six"
    ]
    for dep in deps:
        result = run(f"{VENV_PIP} install {dep} -q", check=False)
        if result.returncode != 0:
            print(f"  Warning: {dep} failed, continuing...")
    
    # Install numpy<2 separately
    print("Ensuring numpy<2...")
    run(f"{VENV_PIP} install 'numpy<2.0' -q", check=False)
    
    # Install compiler_gym (no-deps to avoid grpcio conflict)
    print("Installing compiler_gym...")
    run(f"{VENV_PIP} install compiler_gym --no-deps -q", check=False)
    
    print("Environment setup complete!")


def step2_verify():
    """Step 2: Verify environment."""
    print("=" * 60)
    print("STEP 2: Verifying CompilerGym works")
    print("=" * 60)
    
    # Write test to temp file to avoid shell escaping issues
    test_file = "/tmp/test_compiler_gym.py"
    with open(test_file, "w") as f:
        f.write('''import compiler_gym
env = compiler_gym.make("llvm-v0", benchmark="cbench-v1/qsort",
    observation_space="Autophase", reward_space="IrInstructionCountOz")
obs = env.reset()
ap = env.observation["Autophase"]
ic = env.observation["IrInstructionCount"]
print(f"Autophase: {len(ap)} features, Initial inst: {ic}")
for i in range(3):
    _, reward, done, _ = env.step(env.action_space.sample())
    inst = env.observation["IrInstructionCount"]
    print(f"  Step {i}: inst={inst} reward={reward:.2f} done={done}")
env.close()
print("CompilerGym OK!")
''')
    
    result = run(f"{VENV_PYTHON} {test_file}", check=False)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)


def step3_train():
    """Step 3: Train TRM on real LLVM."""
    print("=" * 60)
    print("STEP 3: Training TRM on REAL LLVM via CompilerGym")
    print("=" * 60)
    
    cmd = [
        VENV_PYTHON,
        f"{PROJECT_DIR}/trm_compiler_real_llvm.py",
        "--epochs", "10",
        "--episodes", "10",
        "--benchmarks", "qsort", "adpcm",
        "--max-steps", "20",
        "--batch-size", "64",
        "--num-random", "20",
        "--seed", "42"
    ]
    
    # Run without --synthetic flag = uses real CompilerGym
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    if result.returncode != 0:
        print(f"Training failed with code {result.returncode}")
        sys.exit(1)


def main():
    # Clone repo if needed
    if not os.path.exists(PROJECT_DIR):
        print("Cloning repo...")
        run("git clone https://github.com/Cree0618/trm-youtubevids.git /content/trm-youtubevids")
    else:
        print("Pulling latest...")
        run(f"git -C {PROJECT_DIR} pull")
    
    # Setup environment
    step1_setup()
    
    # Verify it works
    step2_verify()
    
    # Train on real LLVM
    step3_train()
    
    print("=" * 60)
    print("DONE! TRM trained on REAL LLVM and benchmarked.")
    print("=" * 60)


if __name__ == "__main__":
    main()
