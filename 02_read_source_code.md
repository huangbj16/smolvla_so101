# 02 — Read the SmolVLA source (it's your W2/W3/W4 in production)

SmolVLA is the same architecture family as the mini-π₀ you built: a VLM producing context tokens,
an action expert that **cross-attends** to them, a **flow-matching** head over an **action chunk**,
with a **sinusoidal timestep embedding** and **per-dim action normalization**. This tour maps every
piece back to your code so the production version reads like annotated notes.

Source (your editable install):
```
D:\SOARM101-Testing\lerobot\src\lerobot\policies\smolvla\
├── configuration_smolvla.py     # SmolVLAConfig — all hyperparameters
├── modeling_smolvla.py          # SmolVLAPolicy + VLAFlowMatching (the FM head)  ← start here
└── smolvlm_with_expert.py       # SmolVLMWithExpertModel — the VLM + action-expert attention
```

## The mapping (your file → SmolVLA)

| You built (Wx) | SmolVLA location |
|---|---|
| W2 `MultiHeadSelfAttention` | `smolvlm_with_expert.py :: forward_attn_layer` (line ~209) |
| W2 `CrossAttention` (Q=actions, KV=context) | `smolvlm_with_expert.py :: forward_cross_attn_layer` (~286) |
| W4 `TimestepEmbed` (sinusoidal) | `modeling_smolvla.py :: create_sinusoidal_pos_embedding` (~82) |
| W4 `action_in` + pos + timestep add | `VLAFlowMatching.embed_suffix` (~731) |
| W4 `context` = VLM features (K/V) | `VLAFlowMatching.embed_prefix` (~637) |
| W3/W4 sample `x0` (`randn_like`) | `VLAFlowMatching.sample_noise` (~621) |
| W3/W4 sample `t` (`rand`) | `VLAFlowMatching.sample_time` (~631) |
| W3/W4 `fm_loss` (interpolant + MSE velocity) | `VLAFlowMatching.forward` (~774) |
| W3/W4 `sample_actions` (Euler loop) | `VLAFlowMatching.sample_actions` (~812) |
| W3/W4 one Euler step (`x += v·dt`) | `VLAFlowMatching.denoise_step` (~883) |
| T4 per-dim action **normalization** | `normalize` / `unnormalize` (~172/176), `pad_vector` (~158) |
| chunking + receding horizon (execute + replan) | `SmolVLAPolicy.select_action` (~325) + action queue |

## Suggested reading order (≈45 min)

1. **`configuration_smolvla.py` → `SmolVLAConfig`** (5 min). Skim the hyperparameters and find the
   real numbers behind your toy: chunk size (`chunk_size` / n action steps), number of denoising
   steps (`num_steps` — your `n_steps=10`), action/state dims, VLM name. This grounds the two-time-
   axes picture (denoising steps vs chunk length) in concrete values.

2. **`VLAFlowMatching.forward`** (~774) — **the training loss = your `fm_loss`.** Find, in order:
   the noise sample (`sample_noise`), the time sample (`sample_time`), the **interpolant**
   `x_t = (1-t)·noise + t·actions` (same line you wrote — check their t-direction convention!),
   the **target velocity** `u = actions - noise`, and the MSE against the model's predicted
   velocity. Confirm it's the exact math from W3, just with a VLM in the loop.

3. **`embed_prefix` (~637) and `embed_suffix` (~731)** — how tokens are built.
   - `embed_prefix`: images + language + state → the **prefix/context tokens** (your `context`,
     the K/V of cross-attention). This is where the VLM plugs in (your T4 answer #1).
   - `embed_suffix`: noisy actions → embedded, **+ timestep embedding** (your `action_in` +
     `TimestepEmbed` add). Note how they add the timestep — compare to your `temb.unsqueeze(1)`.

4. **`sample_actions` (~812) + `denoise_step` (~883)** — inference = your Euler sampler. Watch the
   loop integrate from noise over `num_steps`, calling `denoise_step` (one `x += v·dt`). Same loop
   you wrote in `sample_actions`, now conditioned on the VLM prefix.

5. **`smolvlm_with_expert.py`** — `forward_attn_layer` (self-attn) and `forward_cross_attn_layer`
   (cross-attn). Skim to see your scaled-dot-product attention + Q/K/V projections at production
   scale, and how `attention_mode` switches between the VLM's self-attention and the action
   expert's cross-attention to the prefix.

6. **`SmolVLAPolicy.select_action` (~325)** — deployment. See the **action queue / chunking**:
   it samples a chunk, executes actions from it, and re-plans — your receding-horizon note made real.
   Also see `normalize`/`unnormalize` wrapping the actions (your T4 normalization answer).

## Run it in Python to watch the shapes (optional, ~1 GB download)

From a `conda activate lerobot` shell, in this folder:
```python
# inspect_smolvla.py
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")   # downloads ~450M
policy.eval()
n = sum(p.numel() for p in policy.parameters())
print(f"{n/1e6:.0f}M params")                      # ~450M — the whole VLA
print(policy.config.chunk_size, "action steps per chunk")
print("VLM:", policy.config.vlm_model_name if hasattr(policy.config,'vlm_model_name') else policy.config)
```
Run: `python inspect_smolvla.py`. Then set breakpoints (or add prints) inside `VLAFlowMatching.forward`
and `sample_actions` to print tensor shapes — you'll recognize `(B, chunk, action_dim)` and the
`(B,)` timestep, exactly your W4 shapes.

## What to take away (interview gold)

You can now say: *"SmolVLA and π₀ are the same recipe I implemented — a VLM produces prefix tokens,
an action expert cross-attends to them, and a flow-matching head denoises an action chunk over ~10
Euler steps; I've read the `VLAFlowMatching` class and it's the `fm_loss` + `sample_actions` I wrote,
with per-dim action normalization and a receding-horizon action queue on top."*

Next: **[03_collect_data.md](03_collect_data.md)** to record your own demonstrations.
