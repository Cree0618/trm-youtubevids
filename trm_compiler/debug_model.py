"""Debug TRM model with 20-dim feedback."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch
from trm_compiler.model import TinyPassOrderingRefiner, rollout_blind, rollout_closed_loop
from trm_compiler.env_wrapper import make_compiler_env, pass_id_to_name
from trm_compiler.types import CompilerFeedback

model = TinyPassOrderingRefiner()
ckpt = torch.load('trm_compiler_output/trm_model.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Check raw logits with 20-dim feedback
env = make_compiler_env('qsort', use_compilergym=False)
obs, _ = env.reset()
obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
sch = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32).unsqueeze(0)
fb = torch.tensor(CompilerFeedback.zero().encode(), dtype=torch.float32).unsqueeze(0)
y, z = model.init_latent(1, 'cpu')
outputs = model(obs_tensor, sch, fb, y, z)
logits = outputs['pass_logits'].squeeze(0)
probs = torch.softmax(logits, dim=-1)
print(f"Feedback dim: {fb.shape[1]}")
print(f"Pass logits range: [{logits.min():.1f}, {logits.max():.1f}]")
print(f"Top 5 passes:")
top5 = torch.topk(probs, 5)
for i, (idx, val) in enumerate(zip(top5.indices, top5.values)):
    print(f"  {i+1}. {pass_id_to_name(idx.item()):25s} prob={val.item():.4f}")

# Run blind rollout
print(f"\nBlind rollout on qsort:")
env = make_compiler_env('qsort', use_compilergym=False)
obs, _ = env.reset()
trace = rollout_blind(model, obs, max_steps=15, temperature=1.0, device='cpu')
for s in trace:
    if s['pass_id'] >= 0:
        print(f"  Step {s['step']:2d}: {pass_id_to_name(s['pass_id']):25s}")
    else:
        print(f"  Step {s['step']:2d}: halt")

# Run closed-loop rollout
print(f"\nClosed-loop rollout on qsort:")
env = make_compiler_env('qsort', use_compilergym=False)
trace = rollout_closed_loop(model, env, max_steps=15, temperature=1.0, device='cpu')
total_r = 0
for s in trace:
    if s['pass_id'] >= 0:
        total_r += s.get('real_reward', 0)
        print(f"  Step {s['step']:2d}: {pass_id_to_name(s['pass_id']):25s} "
              f"reward={s.get('real_reward', 0):+.4f} inst={s.get('inst_count', '?')}")
    else:
        print(f"  Step {s['step']:2d}: halt")
print(f"Total reward: {total_r:.4f}")
