#!/usr/bin/env bash
# Patch ComfyUI core loaders for faster cold-start via P6 parallel prefetch.
#
# P6: monkey-patch UNETLoader, CLIPLoader, VAELoader, LoraLoaderModelOnly so
# that the first invocation triggers ALL loaders to run in parallel threads.
# Total cold model load time ≈ max(individual) instead of sum, overlapping disk
# I/O for UNET (~7 GB), CLIP (~7.9 GB), VAE (~250 MB), and the two LoRAs.
#
# Ported from tornadocrazy/runpod-comfyui-flux-pulid (patch_pulid.sh).
set -e

NODES_FILE="/comfyui/nodes.py"

if [ ! -f "$NODES_FILE" ]; then
    echo "[patch_qwen] $NODES_FILE not found — aborting" >&2
    exit 1
fi

# Append the parallel prefetch monkey-patch to nodes.py.
# We touch nodes.py because UNETLoader / CLIPLoader / VAELoader /
# LoraLoaderModelOnly all live there and are already imported when the handler
# starts.
cat >> "$NODES_FILE" << 'PARALLEL_PATCH'

# ── P6: Parallel model prefetch (Qwen Edit loaders) ─────────────────────────
# First invocation of any wrapped loader triggers ALL loaders in parallel.
# Cold wall-clock ≈ max(individual) instead of sum — overlaps disk I/O for
# UNET, CLIP, VAE, Lightning LoRA, and HRP_20 LoRA.
import logging as _p6_logging
import threading as _p6_threading
from concurrent.futures import ThreadPoolExecutor as _p6_TPE
import time as _p6_time

_p6_lock = _p6_threading.Lock()
_p6_futures = {}
_p6_cache = {}
_p6_triggered = False
_p6_pool = None

# Qwen Edit model filenames — must match the workflow JSON.
_P6_UNET_NAME = "qwen_image_edit_2511_fp8mixed.safetensors"
_P6_CLIP_NAME = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
_P6_CLIP_TYPE = "qwen_image"
_P6_VAE_NAME = "qwen_image_vae.safetensors"
_P6_LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
_P6_HRP_LORA = "HRP_20.safetensors"

# Save originals (lazy lookup because class bodies may still be executing)
_p6_UNETLoader = None
_p6_CLIPLoader = None
_p6_VAELoader = None
_p6_LoraLoaderModelOnly = None
_p6_orig_load_unet = None
_p6_orig_load_clip = None
_p6_orig_load_vae = None
_p6_orig_load_lora_model_only = None


def _p6_resolve():
    """Resolve loader classes from the module globals."""
    global _p6_UNETLoader, _p6_CLIPLoader, _p6_VAELoader, _p6_LoraLoaderModelOnly
    global _p6_orig_load_unet, _p6_orig_load_clip, _p6_orig_load_vae, _p6_orig_load_lora_model_only
    g = globals()
    _p6_UNETLoader = g.get("UNETLoader")
    _p6_CLIPLoader = g.get("CLIPLoader")
    _p6_VAELoader = g.get("VAELoader")
    _p6_LoraLoaderModelOnly = g.get("LoraLoaderModelOnly")
    if _p6_UNETLoader and hasattr(_p6_UNETLoader, "load_unet"):
        _p6_orig_load_unet = _p6_UNETLoader.load_unet
    if _p6_CLIPLoader and hasattr(_p6_CLIPLoader, "load_clip"):
        _p6_orig_load_clip = _p6_CLIPLoader.load_clip
    if _p6_VAELoader and hasattr(_p6_VAELoader, "load_vae"):
        _p6_orig_load_vae = _p6_VAELoader.load_vae
    if _p6_LoraLoaderModelOnly and hasattr(_p6_LoraLoaderModelOnly, "load_lora_model_only"):
        _p6_orig_load_lora_model_only = _p6_LoraLoaderModelOnly.load_lora_model_only


def _p6_get_pool():
    global _p6_pool
    if _p6_pool is None:
        _p6_pool = _p6_TPE(max_workers=5)
    return _p6_pool


def _p6_trigger_all():
    """Kick off all 5 loads in parallel. Idempotent."""
    global _p6_triggered
    with _p6_lock:
        if _p6_triggered:
            return
        _p6_triggered = True
    _p6_start = _p6_time.time()
    _p6_logging.info("P6: Starting parallel prefetch of Qwen Edit loaders")
    pool = _p6_get_pool()
    if _p6_orig_load_unet:
        _p6_futures["unet"] = pool.submit(
            _p6_orig_load_unet, _p6_UNETLoader(), _P6_UNET_NAME, "default")
    if _p6_orig_load_clip:
        _p6_futures["clip"] = pool.submit(
            _p6_orig_load_clip, _p6_CLIPLoader(), _P6_CLIP_NAME, _P6_CLIP_TYPE, "default")
    if _p6_orig_load_vae:
        _p6_futures["vae"] = pool.submit(
            _p6_orig_load_vae, _p6_VAELoader(), _P6_VAE_NAME)
    # LoRA loads need a model input; we can't parallelize the LoRA APPLICATION
    # (it depends on the UNET output), but we CAN pre-read the safetensors into
    # page cache by calling load_file in parallel. This overlaps the LoRA disk
    # read with UNET/CLIP/VAE reads.
    def _prefetch_lora(name):
        import os
        from safetensors.torch import load_file
        import folder_paths
        path = folder_paths.get_full_path("loras", name)
        if path and os.path.exists(path):
            _t = _p6_time.time()
            sd = load_file(path, device="cpu")
            _p6_logging.info(f"P6: prefetched LoRA {name} ({len(sd)} keys) in {_p6_time.time()-_t:.1f}s")
            del sd
    _p6_futures["lightning_lora"] = pool.submit(_prefetch_lora, _P6_LIGHTNING_LORA)
    _p6_futures["hrp_lora"] = pool.submit(_prefetch_lora, _P6_HRP_LORA)
    _p6_logging.info(
        f"P6: All loaders submitted to thread pool in {_p6_time.time()-_p6_start:.2f}s")


def _p6_cleanup():
    """Release futures and shut down pool after all loads complete."""
    global _p6_pool
    _p6_futures.clear()
    if _p6_pool is not None:
        _p6_pool.shutdown(wait=False)
        _p6_pool = None


def _p6_wait(key):
    """Wait for a specific prefetch to complete and cache the result."""
    with _p6_lock:
        if key in _p6_cache:
            return _p6_cache[key]
    t0 = _p6_time.time()
    result = _p6_futures[key].result()
    with _p6_lock:
        _p6_cache[key] = result
        if len(_p6_cache) >= 5:
            _p6_cleanup()
    _p6_logging.info(f"P6: {key} ready (waited {_p6_time.time()-t0:.1f}s)")
    return result


def _p6_parallel_load_unet(self, unet_name, weight_dtype="default"):
    # Only hijack the canonical Qwen UNET — other models bypass parallelism.
    if unet_name != _P6_UNET_NAME:
        return _p6_orig_load_unet(self, unet_name, weight_dtype)
    _p6_trigger_all()
    return _p6_wait("unet")


def _p6_parallel_load_clip(self, clip_name, type, device="default"):
    if clip_name != _P6_CLIP_NAME:
        return _p6_orig_load_clip(self, clip_name, type, device)
    _p6_trigger_all()
    return _p6_wait("clip")


def _p6_parallel_load_vae(self, vae_name):
    if vae_name != _P6_VAE_NAME:
        return _p6_orig_load_vae(self, vae_name)
    _p6_trigger_all()
    return _p6_wait("vae")


def _p6_parallel_load_lora_model_only(self, model, lora_name, strength_model):
    # LoRA application still has to run synchronously (depends on UNET model),
    # but the safetensors file is already in page cache from the parallel
    # prefetch above — disk read time is eliminated.
    if lora_name in (_P6_LIGHTNING_LORA, _P6_HRP_LORA):
        _p6_trigger_all()
        # Block until our prefetch for this LoRA is done (file in page cache).
        key = "lightning_lora" if lora_name == _P6_LIGHTNING_LORA else "hrp_lora"
        if key in _p6_futures:
            _p6_wait(key)
    return _p6_orig_load_lora_model_only(self, model, lora_name, strength_model)


# Apply monkey-patches
_p6_resolve()
if _p6_orig_load_unet:
    _p6_UNETLoader.load_unet = _p6_parallel_load_unet
    _p6_logging.info("P6: Patched UNETLoader.load_unet")
if _p6_orig_load_clip:
    _p6_CLIPLoader.load_clip = _p6_parallel_load_clip
    _p6_logging.info("P6: Patched CLIPLoader.load_clip")
if _p6_orig_load_vae:
    _p6_VAELoader.load_vae = _p6_parallel_load_vae
    _p6_logging.info("P6: Patched VAELoader.load_vae")
if _p6_orig_load_lora_model_only:
    _p6_LoraLoaderModelOnly.load_lora_model_only = _p6_parallel_load_lora_model_only
    _p6_logging.info("P6: Patched LoraLoaderModelOnly.load_lora_model_only")
_p6_logging.info("P6: Qwen Edit parallel prefetch patches applied")
# ── End P6 ──────────────────────────────────────────────────────────────────
PARALLEL_PATCH

echo "[patch_qwen] P6 parallel prefetch patches appended to $NODES_FILE"
