#!/usr/bin/env bash
set -e

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po 'libtcmalloc.so.\d' | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Ensure ComfyUI-Manager runs in offline network mode inside the container
comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

echo "worker-comfyui: Starting ComfyUI (Qwen Edit)"

# Default to highvram to keep models in VRAM between requests
: "${COMFY_LOG_LEVEL:=DEBUG}"
: "${COMFY_EXTRA_ARGS:=--highvram}"

if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout ${COMFY_EXTRA_ARGS} &

    # Pre-warm GFPGAN/facexlib CUDA kernels in background.
    # First GFPGAN inference triggers ~5s CUDA kernel compilation —
    # running a dummy pass at startup shifts this off the first request.
    python -u /warmup_gfpgan.py &

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

    # Pre-warm Qwen Image Edit pipeline on GPU via a 1-step dummy workflow.
    # Block handler startup until warmup finishes (see comment above).
    python -u /warmup_qwen.py &
    WARMUP_QWEN_PID=$!

    echo "worker-comfyui: Waiting for Qwen warmup before starting handler..."
    wait "${WARMUP_QWEN_PID}" || echo "worker-comfyui - warmup_qwen exited non-zero (continuing anyway)" >&2

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py
fi
