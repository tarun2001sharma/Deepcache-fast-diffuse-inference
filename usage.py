import torch

# Loading the original pipeline
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16).to("cuda:0")

# Import the DeepCacheSDHelper
from DeepCache import DeepCacheSDHelper
helper = DeepCacheSDHelper(pipe=pipe)
helper.set_params(
    cache_interval=3,
    cache_branch_id=0,
)
helper.enable()

# Generate Image
deepcache_image = pipe(
        prompt,
        output_type='pt'
).images[0]
helper.disable()