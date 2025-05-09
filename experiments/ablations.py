# deepcache_ablation_studies.py
"""
Ablation Studies for DeepCache

This script performs additional experiments to analyze the impact of various DeepCache hyperparameters and settings,
including cache_branch_id variations, different sampling schedulers, and memory usage profiling.
Results are logged to CSV files for later analysis.

Usage:
    python deepcache_ablation_studies.py --output_dir ./ablations
"""
import os
import time
import argparse
import torch
import psutil
import pandas as pd
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler
)
from transformers import CLIPProcessor, CLIPModel
from DeepCache import DeepCacheSDHelper
from PIL import Image

def get_gpu_memory():
    """Returns current GPU memory allocated in MB"""
    return torch.cuda.memory_allocated() / 1024**2

def measure_latency_and_clip(
    pipe, helper, prompt, seed, steps, scheduler_name
):
    """Generate an image and return (latency, clip_score)."""
    # prepare CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(pipe.device)
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # baseline or cached
    if helper is not None:
        helper.enable()
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    start = time.time()
    out = pipe(
        prompt,
        num_inference_steps=steps,
        generator=generator,
        output_type="pil"
    )
    latency = time.time() - start
    if helper is not None:
        helper.disable()

    # compute CLIP
    img = out.images[0]
    inputs = clip_proc(text=[prompt], images=[img], return_tensors="pt").to(pipe.device)
    with torch.no_grad():
        img_emb = clip_model.get_image_features(**{"pixel_values": inputs.pixel_values})
        txt_emb = clip_model.get_text_features(**{"input_ids": inputs.input_ids})
        clip_score = torch.cosine_similarity(img_emb, txt_emb).item()

    return latency, clip_score


def run_branch_ablation(
    pipe, prompt, seed, steps, interval,
    branches, scheduler_name, output_dir
):
    """Ablation over cache_branch_id for fixed interval"""
    results = []
    for branch in branches:
        # setup helper
        helper = DeepCacheSDHelper(pipe)
        helper.set_params(cache_interval=interval, cache_branch_id=branch)
        # profile memory before
        mem_before = get_gpu_memory()
        latency, clip = measure_latency_and_clip(pipe, helper, prompt, seed, steps, scheduler_name)
        mem_after = get_gpu_memory()
        results.append({
            "branch": branch,
            "latency_s": latency,
            "clip_score": clip,
            "mem_overhead_MB": mem_after - mem_before,
            "cache_interval": interval,
            "scheduler": scheduler_name,
        })
        print(f"Interval={interval}, Branch={branch}: latency={latency:.2f}s, clip={clip:.4f}, mem_overhead={mem_after-mem_before:.1f}MB")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, f"branch_ablation_I{interval}_{scheduler_name}.csv"), index=False)


def run_scheduler_ablation(
    model_id, prompt, seed, steps, interval, branch,
    schedulers, output_dir
):
    """Ablation over scheduler choice with & without DeepCache"""
    results = []
    for sched_name, sched_cls in schedulers.items():
        # load pipeline with specific scheduler
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to("cuda")
        pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing()

        # baseline run
        lat_base, clip_base = measure_latency_and_clip(pipe, None, prompt, seed, steps, sched_name)
        # DeepCache run
        helper = DeepCacheSDHelper(pipe)
        helper.set_params(cache_interval=interval, cache_branch_id=branch)
        lat_dc, clip_dc = measure_latency_and_clip(pipe, helper, prompt, seed, steps, sched_name)

        results.append({
            "scheduler": sched_name,
            "latency_baseline_s": lat_base,
            "clip_baseline": clip_base,
            "latency_deepcache_s": lat_dc,
            "clip_deepcache": clip_dc,
            "cache_interval": interval,
            "cache_branch": branch,
        })
        print(f"Scheduler={sched_name}: base={lat_base:.2f}s ({clip_base:.4f}), dc={lat_dc:.2f}s ({clip_dc:.4f})")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, f"scheduler_ablation_I{interval}_B{branch}.csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./ablations")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt", type=str, default="A serene mountain landscape with river and trees")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--intervals", nargs="+", type=int, default=[3,5,10])
    parser.add_argument("--branches", nargs="+", type=int, default=[0,1,2,3,4])
    return parser.parse_args()

if __name__ == "__main__":
    import argparse
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # choose schedulers to test
    schedulers = {
        "DDIM": DDIMScheduler,
        "PNDM": PNDMScheduler,
        "PLMS": LMSDiscreteScheduler
    }

    # For branch ablation: same scheduler (PLMS)
    # Load pipeline once
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_attention_slicing()

    for I in args.intervals:
        run_branch_ablation(
            pipe,
            prompt=args.prompt,
            seed=args.seed,
            steps=args.steps,
            interval=I,
            branches=args.branches,
            scheduler_name="PLMS",
            output_dir=args.output_dir
        )

    # Scheduler ablation for a fixed I and branch
    for I in args.intervals:
        for B in args.branches[:2]:  # test first two branch choices
            run_scheduler_ablation(
                model_id=args.model_id,
                prompt=args.prompt,
                seed=args.seed,
                steps=args.steps,
                interval=I,
                branch=B,
                schedulers=schedulers,
                output_dir=args.output_dir
            )
