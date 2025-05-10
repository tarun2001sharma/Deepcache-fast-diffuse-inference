# Install deps (if needed)
# !pip install -q diffusers transformers DeepCache datasets scikit-image matplotlib

import torch, gc
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from DeepCache import DeepCacheSDHelper
from skimage.metrics import structural_similarity as ssim

# 1) Cleanup & device
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to(device)
pipe.enable_attention_slicing()

# 3) Prepare prompts
prompts = [
    "A futuristic cityscape bathed in neon lights at dusk",
    "An ancient castle perched atop misty mountains",
    "An astronaut drifting in space against a swirling nebula"
]
num_steps = 50

# 4) Hook into the U₂ block
last_feat = None
def hook_fn(module, inp, outp):
    global last_feat
    last_feat = outp.detach().cpu()

handle = pipe.unet.up_blocks[2].resnets[0].conv2.register_forward_hook(hook_fn)

# 5) Capture feature maps
all_features = []
for prompt in prompts:
    features = []
    def cb(step_idx, timestep, latents):
        # append the latest feature map
        features.append(last_feat.squeeze(0))
    generator = torch.Generator(device=device).manual_seed(42)
    pipe(
        prompt,
        num_inference_steps=num_steps,
        generator=generator,
        callback_on_step_end=cb
    )
    all_features.append(features)

# remove hook & free memory
handle.remove()
gc.collect()
torch.cuda.empty_cache()

# 6) Panel (a): show orig + features at steps [20,19,1,0]
steps_to_show = [20, 19, 1, 0]
fig, axes = plt.subplots(len(prompts), len(steps_to_show)+1, 
                         figsize=(3*(len(steps_to_show)+1), 3*len(prompts)))
for i, prompt in enumerate(prompts):
    # regenerate original for context
    gen = torch.Generator(device=device).manual_seed(42)
    orig = pipe(prompt, num_inference_steps=num_steps, generator=gen).images[0]
    axes[i,0].imshow(orig); axes[i,0].set_title("Original"); axes[i,0].axis("off")
    feats = all_features[i]
    for j, s in enumerate(steps_to_show, start=1):
        fmap = feats[s]  # this now exists
        heat = fmap.mean(0).numpy()
        hm = (heat - heat.min()) / (heat.max() - heat.min())
        axes[i,j].imshow(hm, cmap="coolwarm", vmin=0, vmax=1)
        axes[i,j].set_title(f"Step {s}"); axes[i,j].axis("off")
plt.suptitle("Feature Maps in U₂ (Steps 20,19,1,0)")
plt.tight_layout()
plt.show()

# 7) Panel (b): cosine-similarity heatmap on last prompt
f = all_features[-1]
T = len(f)
vecs = [feat.reshape(-1) for feat in f]
norms = [v/np.linalg.norm(v) for v in vecs]
sim = np.stack([[float(np.dot(norms[a], norms[b])) for b in range(T)] 
                for a in range(T)])
plt.figure(figsize=(5,5))
plt.imshow(sim, cmap="magma", vmin=0, vmax=1)
plt.colorbar(label="Cosine Similarity")
plt.title("Step-by-Step Similarity (Last Prompt)")
plt.xlabel("Step"); plt.ylabel("Step")
plt.tight_layout()
plt.show()

# 8) Panel (c): % of prior steps with sim > .95
th = 0.95
ratios = [(np.sum(sim[i,:i] > th) / max(1,i))*100 for i in range(T)]
plt.figure(figsize=(5,4))
plt.plot(np.linspace(0,100,T), ratios, '-o')
plt.xlabel("Denoising Progress (%)"); plt.ylabel("% sim > 0.95")
plt.title("High-Similarity Step Ratio")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
