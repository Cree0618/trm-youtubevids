"""Quick end-to-end test for trm_compiler."""
from trm_compiler import (
    generate_compiler_traces, CompilerTraceDataset,
    TinyPassOrderingRefiner, compute_compiler_losses,
    rollout_pass_optimizer, NUM_PASSES, make_compiler_env,
    pass_id_to_name
)
import torch
import numpy as np
from torch.utils.data import DataLoader
from trm_compiler.training import train_one_epoch

print("=== TRM Compiler Pass Ordering - Full Pipeline Test ===")
print()

# 1. Generate traces
print("1. Generating traces...")
traces = generate_compiler_traces(
    benchmarks=["qsort", "adpcm"],
    episodes_per_benchmark=20,
    max_steps_per_episode=20,
    use_heuristic=True,
    seed=42,
)
print(f"   Generated {len(traces)} trace records")

# 2. Create dataset
print("2. Creating dataset...")
dataset = CompilerTraceDataset(traces)
sample = dataset[0]
print(f"   Observation shape: {sample['observation'].shape}")
print(f"   Schedule shape: {sample['schedule'].shape}")
print(f"   Feedback shape: {sample['feedback'].shape}")
print(f"   Pass ID: {sample['pass_id']}")
print(f"   Reward: {sample['reward']:.4f}")

# 3. Initialize model
print("3. Initializing model...")
model = TinyPassOrderingRefiner()
n_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {n_params:,}")

# 4. Quick forward pass
print("4. Testing forward pass...")
obs = sample["observation"].unsqueeze(0)
sch = sample["schedule"].unsqueeze(0)
fb = sample["feedback"].unsqueeze(0)
y, z = model.init_latent(1, "cpu")
outputs = model(obs, sch, fb, y, z)
print(f"   Pass logits shape: {outputs['pass_logits'].shape}")
print(f"   Feasibility: {outputs['feasibility'].item():.4f}")
print(f"   Value: {outputs['value'].item():.4f}")
print(f"   Halt logit: {outputs['halt_logit'].item():.4f}")

# 5. Quick training (2 epochs)
print("5. Quick training (2 epochs)...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
for epoch in range(2):
    losses = train_one_epoch(model, loader, optimizer, "cpu")
    print(f"   Epoch {epoch+1}: loss={losses['total_loss']:.4f} pass={losses['pass_loss']:.4f}")

# 6. Rollout
print("6. Testing rollout...")
model.eval()
env = make_compiler_env("qsort", use_compilergym=False)
obs, _ = env.reset()
print(f"   Initial inst: {env.current_inst_count}")

total_reward = 0.0
for step in range(10):
    trace = rollout_pass_optimizer(model, obs, max_steps=1, temperature=1.0, device="cpu")
    if trace and trace[0]["pass_id"] >= 0:
        pass_id = trace[0]["pass_id"]
        obs, fb, done, info = env.step(pass_id)
        total_reward += fb.reward
        print(
            f"   Step {step}: {pass_id_to_name(pass_id):25s} "
            f"reward={fb.reward:+.4f} inst={env.current_inst_count}"
        )
        if done:
            break
    else:
        print(f"   Step {step}: halt")
        break

print(f"   Total reward: {total_reward:.4f}")
print()
print("=== All tests passed! ===")
