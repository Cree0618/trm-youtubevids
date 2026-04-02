#!/usr/bin/env python3
"""Test TRM compiler on real CompilerGym (Linux only).

Usage:
    pip install compiler_gym torch numpy
    python trm_compiler/test_compilergym.py
"""
import sys
import time


def test_compilergym_available():
    """Check if CompilerGym is installed."""
    try:
        import compiler_gym
        print(f"[OK] compiler_gym version: {compiler_gym.__version__}")
        return True
    except ImportError:
        print("[SKIP] compiler_gym not installed. Install with: pip install compiler_gym")
        return False


def test_env_basic():
    """Test basic CompilerGym environment."""
    import compiler_gym

    env = compiler_gym.make(
        "llvm-v0",
        benchmark="cbench-v1/qsort",
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
    )
    obs = env.reset()

    print(f"[OK] Observation type: {type(obs)}")
    print(f"[OK] Autophase shape: {len(env.observation['Autophase'])}")
    print(f"[OK] IrInstructionCount: {env.observation['IrInstructionCount']}")
    print(f"[OK] Action space size: {env.action_space.n}")
    print(f"[OK] Action names (first 10): {list(env.action_space.names)[:10]}")

    # Try a few actions
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        inst = env.observation["IrInstructionCount"]
        print(f"  Step {i}: action={action} reward={reward:.4f} inst={inst} done={done}")
        if done:
            break

    env.close()
    print("[OK] Environment test passed")
    return True


def test_wrapper():
    """Test our CompilerGymWrapper."""
    from trm_compiler.env_wrapper import CompilerGymWrapper, pass_id_to_name

    wrapper = CompilerGymWrapper(benchmark_id="cbench-v1/qsort")
    wrapper.open()
    obs, initial_inst = wrapper.reset()

    print(f"[OK] Initial inst: {initial_inst}")
    print(f"[OK] Observation shape: {obs.shape}")
    print(f"[OK] Action map size: {len(wrapper._action_map)}")

    total_reward = 0.0
    for step in range(10):
        # Try mem2reg (pass_id=0)
        pass_id = 0
        obs, feedback, done, info = wrapper.step(pass_id)
        total_reward += feedback.reward
        print(f"  Step {step}: {pass_id_to_name(pass_id):15s} "
              f"reward={feedback.reward:+.4f} inst={feedback.instruction_count} "
              f"compiled={feedback.compiled}")
        if done:
            break

    wrapper.close()
    print(f"[OK] Total reward: {total_reward:.4f}")
    return True


def test_trace_generation():
    """Generate traces using real CompilerGym."""
    from trm_compiler.data import generate_compiler_traces

    t0 = time.time()
    traces = generate_compiler_traces(
        benchmarks=["qsort", "adpcm"],
        episodes_per_benchmark=3,
        max_steps_per_episode=15,
        use_heuristic=False,  # use real CompilerGym
        seed=42,
    )
    elapsed = time.time() - t0

    print(f"[OK] Generated {len(traces)} traces in {elapsed:.1f}s")
    if traces:
        sample = traces[0]
        print(f"[OK] Sample observation dim: {len(sample.observation)}")
        print(f"[OK] Sample feedback: inst={sample.feedback.instruction_count} "
              f"reward={sample.feedback.reward:.4f}")
    return True


def test_model_inference():
    """Test model forward pass with CompilerGym data."""
    import torch
    from trm_compiler.model import TinyPassOrderingRefiner, rollout_closed_loop
    from trm_compiler.env_wrapper import CompilerGymWrapper

    model = TinyPassOrderingRefiner()
    model.eval()

    wrapper = CompilerGymWrapper(benchmark_id="cbench-v1/qsort")
    wrapper.open()

    trace = rollout_closed_loop(model, wrapper, max_steps=10, temperature=1.0)
    total_reward = sum(s.get("real_reward", 0) for s in trace)

    print(f"[OK] Model rollout: {len(trace)} steps, reward={total_reward:.4f}")
    for s in trace[:5]:
        from trm_compiler.env_wrapper import pass_id_to_name
        if s["pass_id"] >= 0:
            print(f"  Step {s['step']}: {pass_id_to_name(s['pass_id']):15s} "
                  f"reward={s.get('real_reward', 0):+.4f}")

    wrapper.close()
    return True


def test_training():
    """Quick training test."""
    import torch
    from torch.utils.data import DataLoader
    from trm_compiler.model import TinyPassOrderingRefiner
    from trm_compiler.data import generate_compiler_traces, CompilerTraceDataset
    from trm_compiler.training import train_one_epoch

    traces = generate_compiler_traces(
        benchmarks=["qsort"],
        episodes_per_benchmark=5,
        max_steps_per_episode=15,
        use_heuristic=False,
        seed=42,
    )

    dataset = CompilerTraceDataset(traces)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    model = TinyPassOrderingRefiner()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        losses = train_one_epoch(model, loader, optimizer, "cpu")
        print(f"  Epoch {epoch+1}: loss={losses['total_loss']:.4f} "
              f"pass={losses['pass_loss']:.4f} entropy={losses['entropy']:.4f}")

    print("[OK] Training test passed")
    return True


def test_full_benchmark():
    """Run full benchmark with real CompilerGym."""
    import torch
    from trm_compiler.model import TinyPassOrderingRefiner
    from trm_compiler.baselines import run_full_benchmark

    model = TinyPassOrderingRefiner()

    # Train briefly on real CompilerGym traces
    from torch.utils.data import DataLoader
    from trm_compiler.data import generate_compiler_traces, CompilerTraceDataset
    from trm_compiler.training import train_one_epoch

    print("Generating training traces from CompilerGym...")
    traces = generate_compiler_traces(
        benchmarks=["qsort", "adpcm"],
        episodes_per_benchmark=10,
        max_steps_per_episode=20,
        use_heuristic=False,
        seed=42,
    )

    dataset = CompilerTraceDataset(traces)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training...")
    for epoch in range(10):
        losses = train_one_epoch(model, loader, optimizer, "cpu")
        print(f"  Epoch {epoch+1}: loss={losses['total_loss']:.4f}")

    # Run benchmark
    print("\nRunning benchmark...")
    results = run_full_benchmark(
        model,
        benchmarks=["qsort", "adpcm"],
        max_steps=20,
        device="cpu",
    )

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("TRM Compiler — CompilerGym Integration Test")
    print("=" * 60)

    if not test_compilergym_available():
        sys.exit(1)

    tests = [
        ("Basic environment", test_env_basic),
        ("Wrapper", test_wrapper),
        ("Trace generation", test_trace_generation),
        ("Model inference", test_model_inference),
        ("Training", test_training),
        # ("Full benchmark", test_full_benchmark),  # uncomment for full test
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                print(f"[FAIL] {name}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
