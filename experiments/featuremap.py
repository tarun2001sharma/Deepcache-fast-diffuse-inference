# Install dependencies (run once)
!pip install -q diffusers transformers DeepCache datasets scikit-image matplotlib

import torch, gc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline
from DeepCache import DeepCacheSDHelper
from skimage.metrics import structural_similarity as ssim

# 1) Setup & memory cleanup
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Load SD1.5 pipeline & scheduler
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to(device)
pipe.enable_attention_slicing()

# 3) Prepare prompts
prompts = [
    "A large teddy bear with a heart is in the garbage",
    "A green plate filled with rice and a mixture of sauce on top of it",
    "A very ornate, three layered wedding cake in a banquet room"
]

# 4) Hook into U2 = up_blocks[2]
features = None    # will hold list of feature maps
last_feat = None   # temp storage inside hook

def hook_fn(module, inp, outp):
    global last_feat
    # outp shape: [batch, C, H, W]
    last_feat = outp.detach().cpu()

# attach hook
handle = pipe.unet.up_blocks[2].resnets[0].conv2.register_forward_hook(hook_fn)

# 5) Capture features for each prompt
all_features = []  # will be list of [T x C x H x W] per prompt
num_steps = 50
for prompt in prompts:
    # initialize buffer
    features = [None] * num_steps
    
    # callback to store last_feat into features[step]
    def cb(step, timestep, latents):
        features[step] = last_feat.squeeze(0)  # remove batch dim
    
    # run
    generator = torch.Generator(device=device).manual_seed(42)
    pipe(
        prompt,
        num_inference_steps=num_steps,
        generator=generator,
        callback=cb,
        callback_steps=1,
    )
    all_features.append(features)

# remove hook & clear VRAM
handle.remove()
gc.collect()
torch.cuda.empty_cache()

# 6) Panel (a): visualize feature maps at steps [20,19,1,0]
steps_to_show = [20, 19, 1, 0]
fig, axes = plt.subplots(len(prompts), len(steps_to_show)+1, figsize=(3*(len(steps_to_show)+1), 3*len(prompts)))
for i, prompt in enumerate(prompts):
    # generate original image again for display
    gen = torch.Generator(device=device).manual_seed(42)
    orig = pipe(prompt, num_inference_steps=num_steps, generator=gen).images[0]
    axes[i,0].imshow(orig)
    axes[i,0].set_title("Original")
    axes[i,0].axis("off")
    
    feats = all_features[i]
    for j, s in enumerate(steps_to_show, start=1):
        fmap = feats[s]  # [C,H,W]
        # average across channels for visualization
        heat = fmap.mean(0).numpy()
        # normalize to [0,1]
        hm = (heat - heat.min()) / (heat.max() - heat.min())
        axes[i,j].imshow(hm, cmap="coolwarm", vmin=0, vmax=1)
        axes[i,j].set_title(f"Step{s}")
        axes[i,j].axis("off")

plt.suptitle("Figure 2(a): Examples of Feature Maps in U₂", fontsize=14)
plt.tight_layout()
plt.show()

# 7) Panel (b): heatmap of similarity between all pairs of steps (for the last prompt)
# compute cosine similarity matrix of shape [T x T]
f = all_features[-1]  # last prompt
T = len(f)
# flatten & normalize
vecs = [feat.reshape(-1) for feat in f]
norms = [v / np.linalg.norm(v) for v in vecs]
sim_matrix = np.zeros((T, T), dtype=np.float32)
for a in range(T):
    for b in range(T):
        sim_matrix[a,b] = float(np.dot(norms[a], norms[b]))

plt.figure(figsize=(6,5))
plt.imshow(sim_matrix, cmap="magma", vmin=0, vmax=1)
plt.colorbar(label="Cosine Similarity")
plt.title("Figure 2(b): Heatmap of U₂ Feature Similarity (Last Prompt)")
plt.xlabel("Step index")
plt.ylabel("Step index")
plt.tight_layout()
plt.show()

# 8) Panel (c): % of previous steps with similarity > 0.95
threshold = 0.95
ratios = []
for i in range(T):
    if i == 0:
        ratios.append(1.0)
    else:
        count = np.sum(sim_matrix[i, :i] > threshold)
        ratios.append(count / i * 100)

plt.figure(figsize=(6,4))
plt.plot(np.linspace(0, 100, T), ratios, '-o')
plt.xlabel("Denoising Progress (%)")
plt.ylabel("% Steps with Similarity > 0.95")
plt.title("Figure 2(c): Ratio of Highly Similar Steps (Last Prompt)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

