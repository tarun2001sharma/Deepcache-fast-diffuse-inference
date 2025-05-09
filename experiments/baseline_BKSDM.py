# Install/upgrade deps
!pip install -q diffusers transformers accelerate DeepCache

import torch, time, gc
import matplotlib.pyplot as plt
from diffusers import (
    StableDiffusionPipeline,
    LMSDiscreteScheduler
)
from DeepCache import DeepCacheSDHelper

# 1) Cleanup & device
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Prompts & seeds
prompts = [
    "A person in a helmet is riding a skateboard",
    "There are three vases made of clay on a table",
    "A bicycle is standing next to a bed in a room",
    "A kitten that is sitting down by a door",
    "A serene mountain landscape with a flowing river, lush greenery",
    "A delicate floral arrangement with soft, pastel colors and light",
    "A magical winter wonderland at night. Envision a landscape",
    "A photograph of an abandoned house at the edge of a forest",
    # (you can add "A man holding a surfboard walking on a beach next to the ocean")
]
# one seed per prompt
seeds = [111,222,333,444,555,666,777,888]

# 3) Helper to set PLMS scheduler
def set_plms(p: StableDiffusionPipeline):
    plms = LMSDiscreteScheduler.from_config(p.scheduler.config)
    p.scheduler = plms

# 4) Load pipelines
# 4.1 Baseline SD v1.5
pipe_base = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to(device)
pipe_base.enable_attention_slicing()
set_plms(pipe_base)

# 4.2 BK-SDM-Tiny (architecturally compressed) :contentReference[oaicite:0]{index=0}
pipe_bksdm = StableDiffusionPipeline.from_pretrained(
    "nota-ai/bk-sdm-tiny", torch_dtype=torch.float16
).to(device)
pipe_bksdm.enable_attention_slicing()
set_plms(pipe_bksdm)

# 4.3 DeepCache “Ours” (wrap baseline)
dc = DeepCacheSDHelper(pipe_base)
dc.set_params(cache_interval=5, cache_branch_id=0)

# 5) Run and collect
results = {"base":[], "bksdm":[], "dc": []}
imgs    = {"base":[], "bksdm":[], "dc": []}

for prompt, seed in zip(prompts, seeds):
    gen = torch.Generator(device=device).manual_seed(seed)

    # Baseline
    t0 = time.time()
    img0 = pipe_base(prompt, num_inference_steps=50, generator=gen).images[0]
    t_base = time.time()-t0

    # BK-SDM-Tiny
    gen = torch.Generator(device=device).manual_seed(seed)
    t1 = time.time()
    img1 = pipe_bksdm(prompt, num_inference_steps=50, generator=gen).images[0]
    t_bksdm = time.time()-t1

    # DeepCache
    gen = torch.Generator(device=device).manual_seed(seed)
    dc.enable()
    t2 = time.time()
    img2 = pipe_base(prompt, num_inference_steps=50, generator=gen).images[0]
    dc.disable()
    t_dc = time.time()-t2

    # store
    results["base"].append(t_base)
    results["bksdm"].append(t_bksdm)
    results["dc"].append(t_dc)
    imgs["base"].append(img0)
    imgs["bksdm"].append(img1)
    imgs["dc"].append(img2)

# 6) Plot grid
n = len(prompts)
fig, axes = plt.subplots(3, n, figsize=(2.5*n, 8))
row_titles = ["Stable Diffusion v1.5", "BK-SDM-Tiny", "DeepCache (N=5)"]
for r, key in enumerate(["base","bksdm","dc"]):
    for c in range(n):
        axes[r,c].imshow(imgs[key][c])
        axes[r,c].set_title(f"Time: {results[key][c]:.3f}s", fontsize=8)
        axes[r,c].axis("off")
    axes[r,0].set_ylabel(row_titles[r], rotation=90, size=12)

# Column captions (prompts)
for c, prompt in enumerate(prompts):
    fig.text(0.5/n + c*(1/n), 0.96, prompt, ha='center', va='bottom', fontsize=9)

plt.tight_layout(rect=[0,0,1,0.95])
plt.suptitle("Figure 5 Reproduction: Baseline vs BK-SDM-Tiny vs DeepCache", fontsize=14)
plt.show()
