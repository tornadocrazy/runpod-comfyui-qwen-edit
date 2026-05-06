#!/usr/bin/env bash
# Symlink model files from RunPod's HF Model Cache into ComfyUI's expected
# paths. Requires the endpoint to have Model Caching enabled with HF model
# ID `techwavelaps/qwen-ai-inpainter`.
#
# Why: keeping 30 GB of models inside the docker image bloated the GHCR
# build to 25 GB and made worker image-pulls slow + flaky. Moving model
# files to RunPod's Model Cache means the image stays at ~3 GB and workers
# pick up the models from a host-local overlayfs mount instead.
#
# Cache layout (per RunPod docs):
#   /runpod-volume/huggingface-cache/hub/models--<org>--<name>/
#       refs/main                    -> contains current snapshot hash
#       snapshots/<hash>/<file>      -> the actual files (overlay-mounted)
#
# Snapshot hash changes whenever the HF repo is updated, so we resolve it
# dynamically from refs/main rather than hardcoding.

set -e

CACHE_ROOT=/runpod-volume/huggingface-cache/hub/models--techwavelaps--qwen-ai-inpainter
REFS_FILE="$CACHE_ROOT/refs/main"

echo "[link_models] Waiting for HF model cache to be ready..."
WAIT=0
TIMEOUT=300
while [ ! -f "$REFS_FILE" ]; do
    if [ $WAIT -ge $TIMEOUT ]; then
        echo "[link_models] ERROR: HF model cache not ready after ${TIMEOUT}s." >&2
        echo "[link_models] Is the endpoint configured with Model Caching enabled" >&2
        echo "[link_models] for 'techwavelaps/qwen-ai-inpainter'?" >&2
        exit 1
    fi
    sleep 2
    WAIT=$((WAIT + 2))
done
echo "[link_models] Cache mount detected after ${WAIT}s"

HASH=$(cat "$REFS_FILE")
SRC="$CACHE_ROOT/snapshots/$HASH"
echo "[link_models] Snapshot: $HASH"

if [ ! -d "$SRC" ]; then
    echo "[link_models] ERROR: snapshot directory missing: $SRC" >&2
    exit 1
fi

mkdir -p \
    /comfyui/models/diffusion_models \
    /comfyui/models/text_encoders \
    /comfyui/models/vae \
    /comfyui/models/loras

# UNet
ln -sf "$SRC/qwen_image_edit_2511_fp8mixed.safetensors" \
       /comfyui/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors

# Text encoder
ln -sf "$SRC/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
       /comfyui/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors

# VAE
ln -sf "$SRC/qwen_image_vae.safetensors" \
       /comfyui/models/vae/qwen_image_vae.safetensors

# LoRAs
ln -sf "$SRC/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors" \
       /comfyui/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
ln -sf "$SRC/HRP_20.safetensors" \
       /comfyui/models/loras/HRP_20.safetensors
ln -sf "$SRC/qwen_image_edit_inpainting.safetensors" \
       /comfyui/models/loras/qwen_image_edit_inpainting.safetensors

# Verify all symlinks resolve to actual files (not dangling)
MISSING=0
for f in \
    /comfyui/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
    /comfyui/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
    /comfyui/models/vae/qwen_image_vae.safetensors \
    /comfyui/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
    /comfyui/models/loras/HRP_20.safetensors \
    /comfyui/models/loras/qwen_image_edit_inpainting.safetensors
do
    if [ ! -e "$f" ]; then
        echo "[link_models] MISSING: $f" >&2
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "[link_models] ERROR: one or more model files unavailable in cache." >&2
    exit 1
fi

echo "[link_models] All 6 model files linked successfully."
