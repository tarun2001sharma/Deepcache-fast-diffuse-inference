
import os, time, torch
from diffusers import LMSDiscreteScheduler, DDIMScheduler
from fvcore.nn import FlopCountAnalysis
from DeepCache import DeepCacheLDMHelper
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import DDIMSampler


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Instantiate LDM-4-G from config and checkpoint."""
    cfg = OmegaConf.load(config_path)
    model = instantiate_from_config(cfg.model)
    state = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    scheduler = DDIMSampler(model)
    return cfg, model, scheduler


def sample_class_conditional(
    model, scheduler, imagenet_ds, indices, 
    cache_interval=None, nonuniform_steps=None,
    steps: int = 250, device: torch.device = torch.device('cpu')
):
    """
    Generate one image per index in `indices`, returning list of PIL images and elapsed seconds.
    If cache_interval is set, applies uniform 1:N caching via DeepCacheLDMHelper.
    If nonuniform_steps is set, applies non-uniform updates.
    """
    helper = None
    if cache_interval is not None or nonuniform_steps is not None:
        helper = DeepCacheLDMHelper(model)
        if cache_interval:
            helper.set_params(cache_interval=cache_interval)
    images = []
    start_all = time.time()
    for idx in indices:
        # load label
        _, label = imagenet_ds[idx]
        cond = torch.zeros(1, model.num_classes, device=device)
        cond[0, label] = 1.0

        # cache control
        if cache_interval is None and nonuniform_steps is None:
            pass
        elif nonuniform_steps is not None:
            for step in range(steps):
                if step in nonuniform_steps:
                    helper.enable()
                else:
                    helper.disable()
            helper.disable()
        else:
            helper.enable()

        # sampling
        gen = torch.Generator(device=device).manual_seed(idx)
        samples, _ = scheduler.sample(
            S=steps,
            conditioning=cond,
            batch_size=1,
            shape=(cfg.model.params.ddpm.z_channels, cfg.model.params.ddpm.image_size//32, cfg.model.params.ddpm.image_size//32),
            generator=gen,
            verbose=False
        )
        if cache_interval is not None:
            helper.disable()
        decoded = model.decode_first_stage(samples)
        img = ((decoded[0].cpu().clamp(-1,1) + 1) * 127.5).to(torch.uint8)
        images.append(img.permute(1,2,0).numpy())
    total_time = time.time() - start_all
    return images, total_time


def compute_macs(model, device: torch.device, resolution: int = 256):
    """Compute MACs (in G) for a random forward pass."""
    c = model.num_classes
    dummy_cond = torch.zeros(1, c, device=device)
    dummy_latent = torch.randn(1, 4, resolution//32, resolution//32, device=device)
    flops = FlopCountAnalysis(model, (dummy_latent, dummy_cond))
    return flops.total() / 1e9
