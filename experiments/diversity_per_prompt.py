
"""
Diversity & Per-Prompt Variance Analysis for DeepCache

This script evaluates how DeepCache affects sample diversity and per-prompt output variance.
For each prompt, it generates multiple images under:
    1) Baseline (no cache)
    2) DeepCache (uniform interval)

It computes metrics:
    - Average pairwise LPIPS distance (higher = more diversity)
    - Average pairwise MSE (pixel-level diversity)
    - CLIP score mean and standard deviation across samples
    - Average inference time per image

"""
import argparse
import os
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from diffusers import StableDiffusionPipeline
from DeepCache import DeepCacheSDHelper
from transformers import CLIPProcessor, CLIPModel
import lpips
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompts", type=str, required=True,
                        help="Path to a newline-separated prompts text file")
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="Number of samples per prompt")
    parser.add_argument("--cache_interval", type=int, default=3)
    parser.add_argument("--cache_branch_id", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--output_csv", type=str, default="diversity_results.csv")
    return parser.parse_args()


def load_pipeline(model_id, device):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    pipe.enable_attention_slicing()
    return pipe


def generate_samples(pipe, prompt, seeds, helper, steps, device):
    images = []
    times = []
    for seed in seeds:
        gen = torch.Generator(device=device).manual_seed(seed)
        if helper:
            helper.enable()
        start = time.time()
        out = pipe(prompt, num_inference_steps=steps, generator=gen, output_type="pil")
        latency = time.time() - start
        if helper:
            helper.disable()
        images.append(out.images[0])
        times.append(latency)
    return images, times


def compute_diversity_metrics(images, device, lpips_model):
    # Prepare transforms
    to_tensor = transforms.ToTensor()
    n = len(images)
    # Compute pairwise LPIPS and MSE
    lpips_vals = []
    mse_vals = []
    for i in range(n):
        for j in range(i+1, n):
            # convert to tensors in [-1,1]
            ti = to_tensor(images[i]).to(device) * 2 - 1
            tj = to_tensor(images[j]).to(device) * 2 - 1
            with torch.no_grad():
                d = lpips_model(ti.unsqueeze(0), tj.unsqueeze(0))
            lpips_vals.append(d.item())
            arr_i = np.array(images[i], dtype=np.float32)
            arr_j = np.array(images[j], dtype=np.float32)
            mse = np.mean((arr_i - arr_j)**2)
            mse_vals.append(mse)
    avg_lpips = float(np.mean(lpips_vals)) if lpips_vals else 0.0
    avg_mse = float(np.mean(mse_vals)) if mse_vals else 0.0
    return avg_lpips, avg_mse


def compute_clip_stats(images, prompt, clip_processor, clip_model, device):
    clip_scores = []
    for img in images:
        inputs = clip_processor(text=[prompt], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            im_emb = clip_model.get_image_features(pixel_values=inputs.pixel_values)
            txt_emb = clip_model.get_text_features(input_ids=inputs.input_ids)
            score = torch.cosine_similarity(im_emb, txt_emb).item()
        clip_scores.append(score)
    mean_clip = float(np.mean(clip_scores))
    std_clip = float(np.std(clip_scores))
    return mean_clip, std_clip


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load prompts
    with open(args.prompts, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Seeds for diversity
    seeds = list(range(args.num_seeds))

    # Load models
    pipe = load_pipeline(args.model_id, device)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    lpips_model = lpips.LPIPS(net='alex').to(device)

    # Setup DeepCache helper
    helper = DeepCacheSDHelper(pipe)
    helper.set_params(cache_interval=args.cache_interval,
                      cache_branch_id=args.cache_branch_id)

    # Collect results
    records = []
    for prompt in prompts:
        print(f"Processing prompt: {prompt}")
        # Baseline
        imgs_base, times_base = generate_samples(
            pipe, prompt, seeds, helper=None,
            steps=args.num_steps, device=device
        )
        avg_time_base = float(np.mean(times_base))
        lpips_base, mse_base = compute_diversity_metrics(imgs_base, device, lpips_model)
        clip_mean_base, clip_std_base = compute_clip_stats(imgs_base, prompt, clip_processor, clip_model, device)

        # DeepCache
        imgs_dc, times_dc = generate_samples(
            pipe, prompt, seeds, helper=helper,
            steps=args.num_steps, device=device
        )
        avg_time_dc = float(np.mean(times_dc))
        lpips_dc, mse_dc = compute_diversity_metrics(imgs_dc, device, lpips_model)
        clip_mean_dc, clip_std_dc = compute_clip_stats(imgs_dc, prompt, clip_processor, clip_model, device)

        records.append({
            'prompt': prompt,
            'method': 'baseline',
            'avg_time_s': avg_time_base,
            'avg_lpips': lpips_base,
            'avg_mse': mse_base,
            'clip_mean': clip_mean_base,
            'clip_std': clip_std_base
        })
        records.append({
            'prompt': prompt,
            'method': f'deepcache_I{args.cache_interval}_B{args.cache_branch_id}',
            'avg_time_s': avg_time_dc,
            'avg_lpips': lpips_dc,
            'avg_mse': mse_dc,
            'clip_mean': clip_mean_dc,
            'clip_std': clip_std_dc
        })

    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved diversity & variance results to {args.output_csv}")

if __name__ == '__main__':
    main()
