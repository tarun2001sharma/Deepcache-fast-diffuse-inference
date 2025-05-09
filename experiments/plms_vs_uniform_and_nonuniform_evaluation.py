# Install prerequisites (run once)
!pip install -q diffusers transformers accelerate DeepCache datasets scikit-image matplotlib

# Imports
import time, torch, numpy as np, matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from DeepCache import DeepCacheSDHelper
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset

# 1) Setup device, CLIP, and SD pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLIP for quality metric
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Stable Diffusion v1.5 pipeline (fp16 + slicing for memory)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to(device)
pipe.enable_attention_slicing()

# 2) Load prompts: MS-COCO 2017 captions (first 200 for speed)
ds = load_dataset("coco_captions", "2017", split="validation")
coco_prompts = [cap["caption"] for example in ds for cap in example["captions"]][:200]

# (Optionally, load your PartiPrompt file similarly)
# with open("prompts_partiprompt.txt") as f:
#     partiprompts = [l.strip() for l in f if l.strip()][:200]

# 3) Runner function
def run_experiment(prompts, mode="baseline", cache_interval=None, nonuniform_steps=None):
    """
    mode: "baseline", "uniform", or "nonuniform"
    cache_interval: integer for uniform caching
    nonuniform_steps: list of step-indices at which to refresh cache
    """
    # Setup helper if needed
    helper = None
    if mode in ("uniform", "nonuniform"):
        helper = DeepCacheSDHelper(pipe)
        helper.set_params(cache_interval=cache_interval or 1, cache_branch_id=0)
    
    times, clips = [], []
    for prompt in prompts:
        gen = torch.Generator(device=device).manual_seed(42)
        
        # Baseline
        if mode=="baseline":
            start = time.time()
            out = pipe(prompt, num_inference_steps=50, generator=gen)
            elapsed = time.time() - start
            img = out.images[0]
        
        # Uniform 1:N
        elif mode=="uniform":
            helper.enable()
            start = time.time()
            out = pipe(prompt, num_inference_steps=50, generator=gen)
            elapsed = time.time() - start
            helper.disable()
            img = out.images[0]
        
        # Non-uniform 1:N: toggle caching inside a callback
        else:  # nonuniform
            def cb(step, timestep, latents):
                # enable only if this step is in our refresh list
                if step in nonuniform_steps:
                    helper.enable()
                else:
                    helper.disable()
            helper.enable()  # start enabled on step 0
            start = time.time()
            out = pipe(
                prompt,
                num_inference_steps=50,
                generator=gen,
                callback=cb,
                callback_steps=1,
                output_type="pil"
            )
            elapsed = time.time() - start
            helper.disable()
            img = out.images[0]
        
        # CLIP score
        clip_inputs = clip_proc(text=[prompt], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            im_emb = clip_model.get_image_features(**clip_inputs)
            txt_emb = clip_model.get_text_features(input_ids=clip_inputs.input_ids)
            score = torch.cosine_similarity(im_emb, txt_emb).item()
        
        times.append(elapsed)
        clips.append(score)
    
    return np.mean(times), np.mean(clips)


# 4) Run baseline
t_base, c_base = run_experiment(coco_prompts, mode="baseline")
print(f"Baseline →   time: {t_base:.2f}s,  CLIP: {c_base:.4f}")

# 5) Uniform 1:N for various N
uniform_results = []
for N in [2, 3, 5, 10]:
    t_u, c_u = run_experiment(coco_prompts, mode="uniform", cache_interval=N)
    uniform_results.append((N, t_u, c_u))
    print(f"Uniform I={N} → time: {t_u:.2f}s, CLIP: {c_u:.4f}")

# 6) Non-uniform schedule (example: refresh more in first/last 10 steps)
# For 50 steps: refresh at steps [0,5,10,20,30,40,49]
nu_steps = [0,5,10,20,30,40,49]
t_nu, c_nu = run_experiment(coco_prompts, mode="nonuniform", nonuniform_steps=nu_steps)
print(f"NonUniform → time: {t_nu:.2f}s, CLIP: {c_nu:.4f}")

# 7) Prepare data for plotting
# Speedup = baseline_time / run_time
speedups       = [t_base / u[1] for u in uniform_results]
clip_uniform   = [u[2] for u in uniform_results]
speedup_nu     = t_base / t_nu
clip_nonuniform= c_nu

# 8) Plotting
plt.figure(figsize=(6,4))
plt.plot([1.0] + speedups, [c_base] + clip_uniform, 's-', label="Uniform 1:N")
plt.plot([1.0, speedup_nu], [c_base, clip_nonuniform], 'g^-', label="Non-Uniform 1:N")
plt.plot([1.0, *speedups], [c_base, *clip_uniform], ' ', label="")  # to align markers
# Baseline as a horizontal marker
plt.plot(1.0, c_base, 'bo', label="PLMS (baseline)")
plt.xlabel("Speedup Ratio")
plt.ylabel("CLIP Score")
plt.title("MS-COCO 2017 (reproduced)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
