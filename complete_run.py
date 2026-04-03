#!/usr/bin/env python3
"""Complete TRM training pipeline for Google Colab.

Usage (in Colab cell):
    %run complete_run.py

This script runs the compiler pipeline from an already checked-out repo.
It is safe for Colab because it does not self-clone the repo or uninstall core
packages. By default it runs the synthetic/heuristic path.
"""
import subprocess
import sys
import os
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = os.environ.get("TRM_PROJECT_DIR", os.getcwd())


def setup_environment():
    """Lightweight Colab-safe environment setup."""
    print("=" * 60)
    print("STEP 1: Environment setup")
    print("=" * 60)
    
    # CRITICAL: Downgrade setuptools and wheel to fix gym install
    # See: https://stackoverflow.com/questions/76129688
    print("Fixing setuptools/wheel for gym compatibility...")
    subprocess.run([sys.executable, "-m", "pip", "install", "setuptools==65.5.0", "wheel<0.40.0", "-q"], capture_output=True)
    
    # Install compiler_gym (it depends on old gym which needs fixed setuptools)
    result = subprocess.run([sys.executable, "-m", "pip", "install", "compiler_gym", "-q"], capture_output=True)
    if result.returncode != 0:
        print(f"compiler_gym install warning: {result.stderr.decode()[:300] if result.stderr else 'unknown'}")
    
    # CRITICAL: Downgrade protobuf to fix compiler_gym compatibility
    # Must be done AFTER compiler_gym install to override its dependencies
    # See: https://developers.google.com/protocol-buffers/docs/news/2022-05-06
    print("Fixing protobuf for compiler_gym compatibility...")
    subprocess.run([sys.executable, "-m", "pip", "install", "protobuf>=3.20.0,<4.0.0", "--force-reinstall", "-q"], capture_output=True)
    
    # CRITICAL: Downgrade pydantic to fix compiler_gym compatibility
    # compiler_gym uses deprecated `regex` parameter (removed in pydantic 2.4+)
    print("Fixing pydantic for compiler_gym compatibility...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pydantic>=2.0,<2.3", "--force-reinstall", "-q"], capture_output=True)
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["COMPILER_GYM_HOME"] = "/content/compiler_gym"
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    print(f"PROJECT_DIR={PROJECT_DIR}")


def verify_compiler_gym():
    """Verify CompilerGym works."""
    print("=" * 60)
    print("STEP 3: Verifying CompilerGym")
    print("=" * 60)
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["COMPILER_GYM_HOME"] = "/content/compiler_gym"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    
    try:
        import compiler_gym
    except ImportError as e:
        print(f"compiler_gym not available: {e}")
        print("Using synthetic mode for training")
        return False
    
    try:
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
        return True
    except Exception as e:
        print(f"CompilerGym error: {e}")
        print("Using synthetic mode for training")
        return False


def train_and_benchmark(use_synthetic=True):
    """Train TRM and benchmark."""
    print("=" * 60)
    print("STEP 4: Training TRM + Benchmarking")
    print("=" * 60)
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["COMPILER_GYM_HOME"] = "/content/compiler_gym"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    
    # Build args
    sys.argv = [
        "trm_compiler_real_llvm.py",
        "--epochs", "5",
        "--episodes", "5",
        "--benchmarks", "qsort", "adpcm",
        "--max-steps", "20",
        "--batch-size", "64",
        "--num-random", "20",
        "--seed", "42"
    ]
    
    # Add synthetic flag if needed
    if use_synthetic:
        sys.argv.append("--synthetic")
    
    sys.path.insert(0, PROJECT_DIR)
    
    from trm_compiler_real_llvm import main as train_main
    train_main()


def main():
    setup_environment()
    
    # Check if CompilerGym works
    cg_ok = verify_compiler_gym()
    
    # Train (use synthetic mode if CompilerGym failed)
    train_and_benchmark(use_synthetic=not cg_ok)
    
    print("=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
