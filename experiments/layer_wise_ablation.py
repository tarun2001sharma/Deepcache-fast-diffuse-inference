# layer_wise_ablation.py
"""
Layer-Wise Caching Ablation for DeepCache

evaluates DeepCache by sweeping the cache_branch_id parameter across
all skip-connection layers (i.e., U-Net branches) for a fixed cache_interval.
It measures inference latency, CLIP score, and GPU memory overhead for each
branch, and outputs the results to a CSV file.
"""
import os
import time
import argparse
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from DeepCache import DeepCacheSDHelper

def get_gpu_memory_mb():
    """Return current GPU memory allocated in megabytes."""
    return torch.cuda.memory_allocated() / (1024 ** 2)


def measure_metrics(pipe, helper, prompt, seed, steps):
    """
    Generate an image with the given pipeline and helper (None for baseline),
    returning (latency_s, clip_score, memory_overhead_mb).
    """
    # Setup CLIP for fidelity measurement
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(pipe.device)
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Measure memory before
    mem_before = get_gpu_memory_mb()

    # Optionally enable DeepCache
    if helper is not None:
        helper.enable()

    # Time inference
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    start = time.time()
    result = pipe(prompt, num_inference_steps=steps, generator=generator, output_type="pil")
    latency = time.time() - start

    # Disable DeepCache
    if helper is not None:
        helper.disable()

    # Measure memory after
    mem_after = get_gpu_memory_mb()
    mem_overhead = mem_after - mem_before

    # Compute CLIP score
    img = result.images[0]
    inputs = clip_proc(text=[prompt], images=[img], return_tensors="pt").to(pipe.device)
    with torch.no_grad():
        img_emb = clip_model.get_image_features(pixel_values=inputs.pixel_values)
        txt_emb = clip_model.get_text_features(input_ids=inputs.input_ids)
        clip_score = torch.cosine_similarity(img_emb, txt_emb).item()

    return latency, clip_score, mem_overhead


def main():
    parser = argparse.ArgumentParser(description="Layer-Wise Caching Ablation for DeepCache")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Hugging Face model ID for Stable Diffusion")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--interval", type=int, default=3,
                        help="cache_interval for DeepCache")
    parser.add_argument("--output_csv", type=str, default="layer_wise_results.csv",
                        help="Output CSV file for results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pipeline
    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16
    ).to(device)
    pipe.enable_attention_slicing()

    # Determine number of branches from UNet skip connections
    num_branches = len(pipe.unet.down_blocks)
    print(f"Detected {num_branches} U-Net branches (skip connections). Sweeping branch_id in [0..{num_branches-1}].")

    # Baseline measurement (no caching)
    print("Running baseline (no DeepCache)...")
    baseline_latency, baseline_clip, baseline_mem = measure_metrics(
        pipe, helper=None,
        prompt=args.prompt, seed=args.seed, steps=args.steps
    )
    print(f"Baseline: latency={baseline_latency:.3f}s, clip={baseline_clip:.4f}, mem={baseline_mem:.1f}MB")

    # Sweep branch_id values
    records = []
    for branch_id in range(num_branches):
        print(f"Testing branch_id={branch_id} (cache_interval={args.interval})...")
        helper = DeepCacheSDHelper(pipe)
        helper.set_params(cache_interval=args.interval, cache_branch_id=branch_id)

        lat, clip, mem = measure_metrics(
            pipe, helper=helper,
            prompt=args.prompt, seed=args.seed, steps=args.steps
        )
        speedup = baseline_latency / lat
        records.append({
            "branch_id": branch_id,
            "cache_interval": args.interval,
            "latency_s": lat,
            "speedup": speedup,
            "clip_score": clip,
            "mem_overhead_MB": mem
        })
        print(f" -> latency={lat:.3f}s, speedup={speedup:.2f}x, clip={clip:.4f}, mem_ov={mem:.1f}MB")

    # Compile results
    df = pd.DataFrame(records)
    df.insert(0, "baseline_latency_s", baseline_latency)
    df.insert(1, "baseline_clip_score", baseline_clip)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved layer-wise ablation results to {args.output_csv}")

if __name__ == "__main__":
    main()
