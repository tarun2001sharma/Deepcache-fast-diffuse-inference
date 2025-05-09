# DeepCache Experiment Suite

This repository provides a set of **production‑quality** Python scripts to reproduce and extend the experiments from the [DeepCache: Accelerating Diffusion Models for Free](https://arxiv.org/abs/2312.00858) paper. Each script is designed to run independently: simply install dependencies, pass the required arguments, and inspect the results.

---

## Table of Contents
1. [Setup](#setup)
2. [Experiment Scripts](#experiment-scripts)
   - [DDPM Benchmarks](#ddpm-benchmarks)
   - [Stable Diffusion (SD) Workflows](#stable-diffusion-sd-workflows)
   - [Latent Diffusion (LDM) Benchmarks](#latent-diffusion-ldm-benchmarks)
   - [Ablation Studies](#ablation-studies)
   - [Layer‑Wise Caching Ablation](#layer-wise-caching-ablation)
   - [Diversity & Per‑Prompt Variance](#diversity--per-prompt-variance)
   - [Scheduler Ablation](#scheduler-ablation)
   - [Result Aggregation](#result-aggregation)
3. [MACs (FLOPs) Profiling](#macs-flops-profiling)
4. [Contributing](#contributing)

---

## Setup

1. **Clone** this repository and navigate into it:
   ```bash
   git clone https://github.com/horseee/DeepCache.git
   cd DeepCache
   ```
2. **Create** a virtual environment and activate it:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install** the common dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. For **DDPM** and **LDM-4-G** experiments, you may need to download additional model checkpoints or datasets—consult each script’s `--help` for details.

---

## Experiment Scripts

All scripts are located under the `experiments/` directory (or the repo root for core routines). Use `python <script>.py --help` to see detailed options.

### DDPM Benchmarks

- **Script:** `experiments/ddpm/ddpm_experiments.py`  
- **Purpose:** Apply DeepCache to DDPM pipelines on CIFAR‑10, LSUN Bedroom, and LSUN Church. Generate samples and compute FID.  
- **Example:**
  ```bash
  cd experiments/ddpm
  accelerate launch ddpm_experiments.py \
    --dataset cifar10 \
    --timesteps 100 \
    --cache_interval 5 \
    --branch 2 \
    --output_dir ../../results/ddpm_cifar10
  ```

### Stable Diffusion (SD) Workflows

1. **Text‑to‑Image, Img2Img, Inpainting**
   - **Script:** `generate.py`  
   - **Demo:**
     ```bash
     python generate.py \
       --model runwayml/stable-diffusion-v1-5 \
       --prompt_file data/prompts.txt \
       --cache_interval 3 \
       --cache_branch_id 0 \
       --steps 50 \
       --batch_size 8 \
       --output_dir results/sd_text2img
     ```
2. **BK‑SDM‑Tiny Baseline**
   - **Script:** `baseline_BKSDM.py`  
   - **Demo:**
     ```bash
     python baseline_BKSDM.py \
       --model runwayml/stable-diffusion-v1-5 \
       --prompts data/prompts.txt \
       --steps 50 \
       --output_dir results/bksdm_vs_deepcache
     ```
3. **Cache‑Interval Evolution (Fig 7)**
   - **Script:** `cacheinterval_evolution.py`  
   - **Demo:**
     ```bash
     python cacheinterval_evolution.py \
       --model runwayml/stable-diffusion-v1-5 \
       --prompt "A cat standing on a sink" \
       --intervals 2 3 4 5 6 7 8 \
       --steps 50 \
       --output_dir results/interval_evolution
     ```
4. **CLIP Score Evaluation**
   - **Script:** `clip_score.py`  
   - **Purpose:** Compute CLIP similarity of generated images against their prompts.  
   - **Usage:**
     ```bash
     python clip_score.py --images_dir results/sd_text2img --prompt_file data/prompts.txt
     ```
5. **Feature‐Map Similarity (Fig 2)**
   - **Script:** `featuremap.py`  
   - **Purpose:** Hook into the U‑Net up‑sampling block, extract feature maps across timesteps, and plot similarity heatmaps.

### Latent Diffusion (LDM) Benchmarks

- **Scripts:** `experiments/ldm/sampling.py` & `experiments/ldm/main.py`  
- **Purpose:** Class‑conditional generation on ImageNet with LDM‑4‑G; compute FID, sFID, IS, precision, and recall (reproduce Table 1).  
- **Demo:**
  ```bash
  cd experiments/ldm
  python main.py \
    --imagenet_val /path/to/imagenet/val \
    --config configs/ldm4-g-imagenet.yaml \
    --checkpoint checkpoints/ldm4-g-imagenet.ckpt \
    --batch_size 500 \
    --steps 250 \
    --output_csv ../../results/table1_imagenet.csv
  ```

### Ablation Studies

1. **Cache Branch & Scheduler Ablation**
   - **Script:** `experiments/ablations.py`  
   - **Purpose:** Sweep `cache_branch_id` and compare PLMS, DDIM, PNDM schedulers, measuring latency and CLIP.  
   - **Demo:**
     ```bash
     python experiments/ablations.py \
       --model_id runwayml/stable-diffusion-v1-5 \
       --prompt "A mountain lake at sunrise" \
       --intervals 3 5 \
       --branches 0 1 2 3
     ```
2. **Layer‑Wise Caching Ablation**
   - **Script:** `layer_wise_ablation.py`  
   - **Purpose:** Evaluate every possible `cache_branch_id` at a fixed interval; log speedup, CLIP, and VRAM overhead.  
   - **Demo:**
     ```bash
     python layer_wise_ablation.py \
       --prompt "A sunset over the ocean" \
       --interval 3 \
       --output_csv results/layerwise.csv
     ```
3. **Diversity & Per‑Prompt Variance**
   - **Script:** `diversity_per_prompt_variance.py`  
   - **Purpose:** Generate multiple seeds per prompt, compute LPIPS and pixel MSE diversity, and CLIP score variance.  
   - **Demo:**
     ```bash
     python diversity_per_prompt_variance.py \
       --prompts data/prompts.txt \
       --num_seeds 5 \
       --cache_interval 3 \
       --cache_branch_id 0 \
       --output_csv results/diversity.csv
     ```
4. **Results Aggregation**
   - **Script:** `method_results.py`  
   - **Purpose:** Collect outputs from all experiments and produce a unified summary (CSV or Markdown).  

---

## MACs (FLOPs) Profiling

Profiling FLOPs is built into `sampling.py` via the `compute_macs()` function (using `fvcore`). To include MACs logging in any pipeline:

```python
from sampling import compute_macs
macs = compute_macs(model, device, resolution=256)
print(f"Total MACs: {macs:.2f} G")
```

For detailed per‑module MACs, enable the `layer_wise=True` flag in `count_ops_and_params` (see the original README snippet).

---

## Contributing

We welcome improvements, new experiments, and bug fixes. Please open issues or submit pull requests with descriptive titles and thorough documentation.
