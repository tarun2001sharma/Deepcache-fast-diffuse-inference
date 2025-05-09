# 1) Install / upgrade dependencies (uncomment if needed)
# !pip install -q diffusers transformers accelerate DeepCache matplotlib

import torch, time, gc
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from DeepCache import DeepCacheSDHelper

# 2) Cleanup and device setup
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3) Load SD v1.5 pipeline (FP16 + safetensors)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True
).to(device)
pipe.enable_attention_slicing()  # reduce peak VRAM

# 4) Wrap with DeepCache helper (we'll set interval in the loop)
dc = DeepCacheSDHelper(pipe)
dc.set_params(cache_branch_id=0)  # always branch 0

# 5) Prompts, seeds, and intervals
prompts = [
    "A cat standing on the edge of a sink drink water",
    "A child riding a skateboard on a city street"
]
seeds    = [2025, 4242]        # fixed seeds for reproducibility
intervals = [None, 2, 3, 4, 5, 6, 7, 8]  # None → baseline (no caching)

# 6) Generate all images + record timings
results_imgs  = [[None]*len(intervals) for _ in prompts]
results_times = [[0.0]*len(intervals) for _ in prompts]

for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
    for j, N in enumerate(intervals):
        gen = torch.Generator(device=device).manual_seed(seed)
        
        # enable/disable DeepCache
        if N is None:
            dc.disable()
        else:
            dc.set_params(cache_interval=N, cache_branch_id=0)
            dc.enable()
        
        # inference timing
        t0 = time.time()
        img = pipe(prompt, num_inference_steps=50, generator=gen).images[0]
        elapsed = time.time() - t0
        
        if N is not None:
            dc.disable()
        
        results_imgs[i][j]  = img
        results_times[i][j] = elapsed

# 7) Plot 2×8 grid
n_rows, n_cols = len(prompts), len(intervals)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))

# Column titles
col_titles = ["Original"] + [f"N={n}" for n in intervals[1:]]
for ax, title in zip(axes[0], col_titles):
    ax.set_title(title, fontsize=12)

# Row label positions (vertical)
row_labels = ["Prompt 1", "Prompt 2"]
for i, row_label in enumerate(row_labels):
    axes[i][0].set_ylabel(row_label, rotation=90, size=14, labelpad=10)

# Populate images and time captions
for i in range(n_rows):
    for j in range(n_cols):
        axes[i][j].imshow(results_imgs[i][j])
        axes[i][j].axis("off")
        axes[i][j].text(
            0.5, -0.1,
            f"Time: {results_times[i][j]:.3f}s",
            ha="center", va="top", transform=axes[i][j].transAxes,
            fontsize=10
        )

plt.suptitle("Evolution of Generated Images with Increasing cache_interval N", fontsize=16)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()
