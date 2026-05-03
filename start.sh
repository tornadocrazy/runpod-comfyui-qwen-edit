#!/usr/bin/env bash
set -e

# ── Runtime installs (kept out of Dockerfile to stay under 30-min build limit) ──

# controlnet_aux — DWPose/OpenPose skeleton extraction for ControlNet workflows
CONTROLNET_AUX=/comfyui/custom_nodes/comfyui_controlnet_aux
if [ ! -d "$CONTROLNET_AUX" ]; then
    echo "worker-comfyui: Installing comfyui_controlnet_aux..."
    git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux "$CONTROLNET_AUX"
    pip install --no-cache-dir -r "$CONTROLNET_AUX/requirements.txt"
fi

# Union ControlNet LoRA (~944 MB) — downloaded once on first start
UNION_LORA=/comfyui/models/loras/qwen_image_union_diffsynth_lora.safetensors
if [ ! -f "$UNION_LORA" ]; then
    echo "worker-comfyui: Downloading union ControlNet LoRA (~944 MB)..."
    wget -q \
        https://huggingface.co/Comfy-Org/Qwen-Image-DiffSynth-ControlNets/resolve/main/split_files/loras/qwen_image_union_diffsynth_lora.safetensors \
        -O "$UNION_LORA"
fi

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
