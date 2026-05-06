#!/usr/bin/env bash
set -e

# All models (including Union ControlNet + Fusion LoRAs) are baked into the
# image — no runtime downloads needed.

# ─────────────────────────────────────────────────────────────────────────────
# Disk-read-early: warm OS page cache for big model files in parallel with
# ComfyUI startup. When ComfyUI later loads these files from disk, the read
# is from RAM (page cache) instead of network volume → 5-9s saved on cold
# start. The reads happen in background; ComfyUI startup proceeds normally.
# ─────────────────────────────────────────────────────────────────────────────
(
    cat /comfyui/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors > /dev/null 2>&1 || true
    cat /comfyui/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors > /dev/null 2>&1 || true
    cat /comfyui/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors > /dev/null 2>&1 || true
    cat /comfyui/models/loras/HRP_20.safetensors > /dev/null 2>&1 || true
    cat /comfyui/models/loras/qwen_image_edit_inpainting.safetensors > /dev/null 2>&1 || true
    cat /comfyui/models/vae/qwen_image_vae.safetensors > /dev/null 2>&1 || true
) &
echo "worker-comfyui: disk-read-early started (PID $!) — warming page cache for big models"

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po 'libtcmalloc.so.\d' | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Ensure ComfyUI-Manager runs in offline network mode inside the container
comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

echo "worker-comfyui: Starting ComfyUI (Qwen Edit)"

# Default to highvram to keep models in VRAM between requests.
# COMFY_LOG_LEVEL=INFO suppresses the per-module "lowvram: loaded module
# regularly" + "Backend eager selected for dequantize_per_tensor_fp8" spam
# (~120 lines per cold start) — it's DEBUG noise, not actual lowvram mode.
: "${COMFY_LOG_LEVEL:=INFO}"
: "${COMFY_EXTRA_ARGS:=--highvram}"

if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout ${COMFY_EXTRA_ARGS} &

    # Pre-warm GFPGAN/facexlib CUDA kernels in background.
    # First GFPGAN inference triggers ~5s CUDA kernel compilation —
    # running a dummy pass at startup shifts this off the first request.
    python -u /warmup_gfpgan.py &

    # Pre-warm InsightFace (buffalo_l + inswapper) ONNX CUDA sessions.
    # First ReActor face-swap triggers ~30s of CUDA kernel compile.
    python -u /warmup_insightface.py &

    # Pre-warm Qwen Image Edit pipeline on GPU via a 1-step dummy workflow.
    # Block handler startup until warmup finishes so the first real request
    # is guaranteed to find models already on GPU (avoids race with handler
    # uploads for the ComfyUI /prompt queue slot).
    python -u /warmup_qwen.py &
    WARMUP_QWEN_PID=$!

    echo "worker-comfyui: Waiting for Qwen warmup before starting handler..."
    wait "${WARMUP_QWEN_PID}" || echo "worker-comfyui - warmup_qwen exited non-zero (continuing anyway)" >&2

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --verbose "${COMFY_LOG_LEVEL}" --log-stdout ${COMFY_EXTRA_ARGS} &

    # Pre-warm GFPGAN/facexlib CUDA kernels in background.
    python -u /warmup_gfpgan.py &

    # Pre-warm InsightFace (buffalo_l + inswapper) ONNX CUDA sessions.
    python -u /warmup_insightface.py &

    # Pre-warm Qwen Image Edit pipeline on GPU via a 1-step dummy workflow.
    # Block handler startup until warmup finishes (see comment above).
    python -u /warmup_qwen.py &
    WARMUP_QWEN_PID=$!

    echo "worker-comfyui: Waiting for Qwen warmup before starting handler..."
    wait "${WARMUP_QWEN_PID}" || echo "worker-comfyui - warmup_qwen exited non-zero (continuing anyway)" >&2

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py
fi
