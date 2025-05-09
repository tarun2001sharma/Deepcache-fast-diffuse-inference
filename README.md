# DeepCache Course Project Submission

This repository contains **production-grade** experiment code built on top of the **DeepCache** framework (Ma et al., CVPR 2024). It was developed as a graduate class project to reproduce, extend, and deeply analyze the key contributions of DeepCache, including new ablation studies, diversity analyses, and benchmark reproductions.

---

## Overview

The original **DeepCache** work introduces a training-free mechanism to accelerate diffusion models by caching high-level features in the U-Net architecture, yielding up to **2.3×** speedup on Stable Diffusion v1.5 and **4.1×** on LDM-4-G with minimal quality loss. This project builds on that foundation by adding:

- **Branch Ablation**: Systematic sweep of `cache_branch_id` to locate the most effective caching layer.  
- **Scheduler Ablation**: Comparison across PLMS, DDIM, PNDM schedulers with and without caching.  
- **Layer‐Wise Caching Ablation**: Per-U-Net-block evaluation of speedup, CLIP fidelity, and memory overhead.  
- **Diversity & Variance**: Multi‐seed sampling per prompt to measure LPIPS/MSE diversity and CLIP variance under caching.  
- **Cache‐Interval Evolution**: Visualization of output changes as `cache_interval` increases (Figure 7 recreation).  
- **Reproduction of Table 1**: Class‐conditional ImageNet benchmarks with LDM‑4‑G (FID, IS, precision, recall).  

All experiments are implemented as standalone Python scripts for clarity and reproducibility.

---

## Setup

1. **Clone**:
   ```bash
   git clone https://github.com/username/deepcache-course-project.git
   cd deepcache-course-project
   ```
2. **Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Prepare data**:
   - For DDPM: download CIFAR-10 / LSUN datasets.  
   - For LDM-4-G: place ImageNet validation images under `data/imagenet_val/`.  

---

## Experiment Scripts

Each script lives in the root or under `experiments/`. All accept `--help` for detailed options.

### 1. DDPM Benchmarks
- **File**: `experiments/ddpm/ddpm_experiments.py`  
- **What**: Apply DeepCache to DDPM on CIFAR-10, LSUN. Compute FID.  
- **Run**:
  ```bash
  cd experiments/ddpm
  accelerate launch ddpm_experiments.py \
    --dataset cifar10 \
    --timesteps 100 \
    --cache_interval 5 \
    --branch 2 \
    --output_dir ../../results/ddpm_cifar10
  ```

### 2. Stable Diffusion Workflows
- **Text2Img, Img2Img, Inpainting**: `generate.py`  
- **Baseline vs BK-SDM**: `baseline_BKSDM.py`  
- **CacheInterval Evolution (Fig 7)**: `cacheinterval_evolution.py`  
- **CLIP Scoring**: `clip_score.py`  
- **FeatureMap Heatmaps (Fig 2)**: `featuremap.py`  

**Example** (Text2Img with caching):
```bash
python generate.py \
  --model runwayml/stable-diffusion-v1-5 \
  --prompt_file prompts.txt \
  --cache_interval 3 \
  --cache_branch_id 0 \
  --steps 50 \
  --batch_size 8 \
  --output_dir results/sd_text2img
```

### 3. Latent Diffusion (ImageNet) Benchmarks
- **Files**: `experiments/ldm/sampling.py`, `experiments/ldm/main.py`  
- **What**: Reproduce Table 1 on ImageNet using LDM‑4‑G (FID, IS, precision, recall).  
- **Run**:
  ```bash
  cd experiments/ldm
  python main.py \
    --imagenet_val data/imagenet_val \
    --config configs/ldm4-g-imagenet.yaml \
    --checkpoint checkpoints/ldm4-g-imagenet.ckpt \
    --batch_size 500 \
    --steps 250 \
    --output_csv ../../results/table1_imagenet.csv
  ```

### 4. Ablation Studies

#### 4.1 Branch & Scheduler Ablation
- **File**: `experiments/ablations.py`  
- **What**: Sweep `cache_branch_id` and test PLMS/DDIM/PNDM schedulers; record latency & CLIP fidelity.  
- **Run**:
  ```bash
  python experiments/ablations.py \
    --model_id runwayml/stable-diffusion-v1-5 \
    --prompt "A mountain lake at sunrise" \
    --intervals 3 5 \
    --branches 0 1 2 3
  ```

#### 4.2 Layer-Wise Caching Ablation
- **File**: `layer_wise_ablation.py`  
- **What**: Evaluate each U-Net skip path (`cache_branch_id`) with fixed `cache_interval`; measure speedup, CLIP, memory overhead.  
- **Run**:
  ```bash
  python layer_wise_ablation.py \
    --prompt "A sunset over the ocean" \
    --interval 3 \
    --output_csv results/layerwise.csv
  ```

#### 4.3 Diversity & Variance
- **File**: `diversity_per_prompt_variance.py`  
- **What**: Multi-seed sampling per prompt; compute LPIPS & MSE diversity and CLIP score variance.  
- **Run**:
  ```bash
  python diversity_per_prompt_variance.py \
    --prompts prompts.txt \
    --num_seeds 5 \
    --cache_interval 3 \
    --cache_branch_id 0 \
    --output_csv results/diversity.csv
  ```

### 5. Results Aggregation
- **File**: `method_results.py`  
- **What**: Collect outputs from all experiments into a unified summary table (CSV/Markdown).

---

## MACs (FLOPs) Profiling

Use the helper in `sampling.py`:
```python
from sampling import compute_macs
macs = compute_macs(model, device, resolution=256)
print(f"Total MACs: {macs:.2f} G")
```
Enable `layer_wise=True` in `count_ops_and_params` to see per-module FLOPs.

---

## Original DeepCache Reference

> **DeepCache: Accelerating Diffusion Models for Free**  
> **Xinyin Ma**, Gongfan Fang, Xinchao Wang (NUS Learning & Vision Lab)  
> [[ArXiv]](https://arxiv.org/abs/2312.00858) [[Project]](https://horseee.github.io/Diffusion_DeepCache/)

---

## BibTeX
```bibtex
@inproceedings{ma2023deepcache,
  title={DeepCache: Accelerating Diffusion Models for Free},
  author={Ma, Xinyin and Fang, Gongfan and Wang, Xinchao},
  booktitle={CVPR},
  year={2024}
}
```

*This README has been adapted for a graduate course submission, emphasizing reproducibility, clarity, and extensibility.*
