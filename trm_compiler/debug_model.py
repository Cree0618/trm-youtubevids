"""Debug TRM model pass selection after 50 epochs."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch
from trm_compiler.model import TinyPassOrderingRefiner, rollout_pass_optimizer
from trm_compiler.env_wrapper import make_compiler_env, pass_id_to_name

model = TinyPassOrderingRefiner()
ckpt = torch.load('trm_compiler_output/trm_model.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Check raw logits
env = make_compiler_env('qsort', use_compilergym=False)
obs, _ = env.reset()
obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
sch = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32).unsqueeze(0)
fb = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32).unsqueeze(0)
y, z = model.init_latent(1, 'cpu')
outputs = model(obs_tensor, sch, fb, y, z)
logits = outputs['pass_logits'].squeeze(0)
probs = torch.softmax(logits, dim=-1)
print(f"Pass logits range: [{logits.min():.1f}, {logits.max():.1f}]")
print(f"Pass probs range: [{probs.min():.4f}, {probs.max():.4f}]")
print(f"Top 5 passes by probability:")
top5 = torch.topk(probs, 5)
for i, (idx, val) in enumerate(zip(top5.indices, top5.values)):
    print(f"  {i+1}. {pass_id_to_name(idx.item()):25s} prob={val.item():.4f}  logit={logits[idx].item():.1f}")

# Run rollout
print(f"\nRollout on qsort:")
env = make_compiler_env('qsort', use_compilergym=False)
obs, _ = env.reset()
print(f"Initial inst: {env.current_inst_count}")
total_reward = 0.0
for step in range(15):
    trace = rollout_pass_optimizer(model, obs, max_steps=1, temperature=1.0, device='cpu')
    if trace and trace[0]['pass_id'] >= 0:
        pass_id = trace[0]['pass_id']
        obs, fb, done, info = env.step(pass_id)
        total_reward += fb.reward
        print(f"  Step {step:2d}: {pass_id_to_name(pass_id):25s} reward={fb.reward:+.4f} inst={env.current_inst_count}")
        if done:
            break
    else:
        prob = trace[0]['halt_prob']
        print(f"  Step {step:2d}: halt (prob={prob:.3f})")
        break
print(f"Total reward: {total_reward:.4f}")
