"""
Warmup script: load all Qwen Edit models into VRAM in parallel before handler accepts jobs.
Runs as a background process during container boot.

Bypasses ComfyUI's sequential node execution — loads UNET, text encoder,
VAE, and LoRA concurrently using threads so cold start is faster.
"""
import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='[warmup] %(message)s')
log = logging.getLogger(__name__)

MODELS_DIR = "/comfyui/models"


def load_unet():
    """Load Qwen Image Edit diffusion model (fp8mixed) into GPU"""
    t0 = time.time()
    import torch
    from safetensors.torch import load_file
    path = os.path.join(MODELS_DIR, "diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors")
    sd = load_file(path, device="cuda")
    log.info(f"UNET: {len(sd)} keys loaded in {time.time()-t0:.1f}s")
    del sd
    torch.cuda.empty_cache()


def load_text_encoder():
    """Load Qwen 2.5 VL 7B text encoder (fp8)"""
    t0 = time.time()
    import torch
    from safetensors.torch import load_file
    path = os.path.join(MODELS_DIR, "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors")
    sd = load_file(path, device="cuda")
    log.info(f"Text encoder: {len(sd)} keys loaded in {time.time()-t0:.1f}s")
    del sd
    torch.cuda.empty_cache()


def load_vae():
    """Load Qwen Image VAE"""
    t0 = time.time()
    import torch
    from safetensors.torch import load_file
    path = os.path.join(MODELS_DIR, "vae/qwen_image_vae.safetensors")
    sd = load_file(path, device="cuda")
    log.info(f"VAE: {len(sd)} keys loaded in {time.time()-t0:.1f}s")
    del sd
    torch.cuda.empty_cache()


def load_lora():
    """Load Lightning LoRA (4-step, bf16)"""
    t0 = time.time()
    import torch
    from safetensors.torch import load_file
    path = os.path.join(MODELS_DIR, "loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
    sd = load_file(path, device="cuda")
    log.info(f"LoRA: {len(sd)} keys loaded in {time.time()-t0:.1f}s")
    del sd
    torch.cuda.empty_cache()


def main():
    t_start = time.time()
    log.info("Starting parallel model warmup (Qwen Edit)...")

    loaders = [
        ("unet", load_unet),
        ("text_encoder", load_text_encoder),
        ("vae", load_vae),
        ("lora", load_lora),
    ]

    with ThreadPoolExecutor(max_workers=len(loaders)) as pool:
        futures = {pool.submit(fn): name for name, fn in loaders}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                log.error(f"{name} failed: {e}")

    log.info(f"All models warmed up in {time.time()-t_start:.1f}s (parallel)")


if __name__ == "__main__":
    main()
